# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Structured JSON logging, Intent vs. Outcome capture, and OTel tracing (Rubric 4.1-4.4).

Components:
    - **Privacy-by-default span content**: pins ``ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS``
      to ``"false"`` locally so raw unredacted prompts/responses are opt-in
      rather than opt-out (matching Terraform's ``service.tf``).
    - **Structured JSON Logging (4.1)**: uses ``google-cloud-logging``'s
      ``StructuredLogHandler`` (already declared in ``pyproject.toml``), driven
      by ``LOG_LEVEL`` and scrubbed via ``RedactingLogFilter`` (``app/redaction.py``).
    - **Distributed Tracing (4.3)**: configures OpenTelemetry ``TracerProvider``
      with ``TRACE_EXPORTER=console`` locally (no GCP project required) and
      ``TRACE_EXPORTER=cloud_trace`` for Cloud Trace, correlating ``trace_id``
      and ``span_id`` with every JSON log entry.
    - **Intent vs. Outcome + Custom Spans (4.2 & 4.3)**: ``ObservabilityPlugin``
      registers ahead of ``PolicyPlugin`` on ``App`` so every tool call —
      including ``PolicyPlugin`` short-circuits and HITL approve/reject stops —
      emits matched ``intent`` and ``outcome`` JSON events sharing the same
      ``invocation_id``, plus nested custom OTel spans for the prep flow, every
      workflow stage, and every tool call with PII-redacted attributes.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, TextIO

# Privacy-by-default (Rubric 4.4): ADK 2.15 defaults
# ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS to "true" in local environments, which
# writes raw unredacted prompts and responses into OpenTelemetry spans. Setting
# it to "false" via setdefault makes message-content capture strictly opt-in
# rather than opt-out, matching Terraform's production configuration.
os.environ.setdefault("ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS", "false")

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.cloud.logging.handlers import StructuredLogHandler
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Span, TracerProvider as SDKTracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.trace import SpanContext, Status, StatusCode

from app.config import MODEL_ROUTING_TABLE
from app.redaction import redact

LOGGER_NAME = "meeting_prep_concierge"
TRACER_NAME = "meeting_prep_concierge"
TRACER_VERSION = "0.1.0"

_WORKFLOW_STAGES: frozenset[str] = frozenset(
    {
        "gather_step",
        "attendee_agent",
        "history_agent",
        "brief_agent",
        "critic_agent",
        "approval_gate_agent",
        "presenter_agent",
    }
)
_PARALLEL_GATHER_BRANCHES: frozenset[str] = frozenset(
    {"attendee_agent", "history_agent"}
)
_HITL_TOOLS: frozenset[str] = frozenset(
    {"request_brief_approval", "send_brief_to_attendees"}
)

_TRACING_CONFIGURED = False


# ---------------------------------------------------------------------------
# 4.1 & 4.4 — Structured JSON Logging with Active PII Redaction
# ---------------------------------------------------------------------------


class RedactingLogFilter(logging.Filter):
    """Logging filter that scrubs PII from log messages and structured json_fields."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = redact(record.args)
            elif isinstance(record.args, tuple):
                record.args = tuple(redact(arg) for arg in record.args)
        json_fields = getattr(record, "json_fields", None)
        if isinstance(json_fields, dict):
            record.json_fields = redact(json_fields)
        return True


def _resolve_log_level(raw_level: str | None = None) -> int:
    name = (raw_level or os.getenv("LOG_LEVEL", "INFO")).strip().upper()
    mapping = logging.getLevelNamesMapping()
    return mapping.get(name, logging.INFO)


def configure_structured_logging(
    *,
    stream: TextIO | None = None,
    level: str | None = None,
    force: bool = False,
) -> logging.Logger:
    """Configure the ``meeting_prep_concierge`` logger with GCP StructuredLogHandler.

    Emits single-line JSON records carrying ``severity``, ``json_fields`` payload,
    ``logging.googleapis.com/trace``, and ``logging.googleapis.com/spanId``,
    with ``RedactingLogFilter`` scrubbing PII prior to serialization.
    """
    logger = logging.getLogger(LOGGER_NAME)
    resolved_level = _resolve_log_level(level)
    logger.setLevel(resolved_level)
    logger.propagate = False

    if force:
        logger.handlers.clear()

    if not any(isinstance(h, StructuredLogHandler) for h in logger.handlers):
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or None
        handler = StructuredLogHandler(
            stream=stream or sys.stderr,
            project_id=project_id,
        )
        handler.setLevel(resolved_level)
        handler.addFilter(RedactingLogFilter())
        logger.addHandler(handler)
    else:
        for handler in logger.handlers:
            handler.setLevel(resolved_level)

    return logger


def get_structured_logger() -> logging.Logger:
    """Return the configured structured JSON logger for the application."""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        configure_structured_logging()
    return logger


def _format_trace_fields(span_context: SpanContext | None) -> tuple[str, str, str]:
    """Return (trace_id_hex, span_id_hex, gcp_trace_resource) for log-trace correlation."""
    if span_context is None or not span_context.is_valid:
        current = trace.get_current_span().get_span_context()
        span_context = current if current.is_valid else None
    if span_context is None:
        return "", "", ""

    trace_id_hex = f"{span_context.trace_id:032x}"
    span_id_hex = f"{span_context.span_id:016x}"
    project = os.getenv("GOOGLE_CLOUD_PROJECT") or "local"
    gcp_trace = f"projects/{project}/traces/{trace_id_hex}"
    return trace_id_hex, span_id_hex, gcp_trace


def emit_structured_log(
    logger: logging.Logger,
    level: int,
    message: str,
    *,
    json_fields: dict[str, Any],
    span_context: SpanContext | None = None,
) -> None:
    """Emit a structured JSON log line correlated with the active OpenTelemetry trace."""
    trace_id_hex, span_id_hex, gcp_trace = _format_trace_fields(span_context)
    enriched_fields = dict(json_fields)
    if trace_id_hex:
        enriched_fields["trace_id"] = trace_id_hex
        enriched_fields["span_id"] = span_id_hex

    extra: dict[str, Any] = {"json_fields": enriched_fields}
    if gcp_trace:
        extra["trace"] = gcp_trace
        extra["span_id"] = span_id_hex
        extra["trace_sampled"] = True

    logger.log(level, message, extra=extra)


# ---------------------------------------------------------------------------
# 4.3 — Distributed Tracing Configuration (Console Local / Cloud Trace GCP)
# ---------------------------------------------------------------------------


def _resolve_trace_exporter_mode() -> str:
    explicit = os.getenv("TRACE_EXPORTER", "").strip().lower()
    if explicit:
        return explicit
    cloud_enabled = os.getenv(
        "GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY", ""
    ).strip().lower() in ("true", "1")
    if cloud_enabled and os.getenv("GOOGLE_CLOUD_PROJECT"):
        return "cloud_trace"
    return "console"


def configure_tracing(
    *,
    exporter_mode: str | None = None,
    custom_exporter: SpanExporter | None = None,
    console_stream: TextIO | None = None,
) -> SDKTracerProvider:
    """Configure OpenTelemetry tracing with console export locally and Cloud Trace on GCP.

    Safe to call alongside ADK's ``otel_to_cloud=True``: reuses any existing
    ``SDKTracerProvider`` instead of attempting a disallowed provider override.
    """
    global _TRACING_CONFIGURED

    existing = trace.get_tracer_provider()
    if isinstance(existing, SDKTracerProvider):
        provider = existing
    else:
        service_name = os.getenv("OTEL_SERVICE_NAME", "meeting-prep-concierge")
        provider = SDKTracerProvider(
            resource=Resource.create(
                {
                    "service.name": service_name,
                    "service.version": os.getenv("AGENT_VERSION", TRACER_VERSION),
                }
            )
        )
        trace.set_tracer_provider(provider)

    if custom_exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(custom_exporter))
        _TRACING_CONFIGURED = True
        return provider

    if _TRACING_CONFIGURED:
        return provider

    mode = (exporter_mode or _resolve_trace_exporter_mode()).strip().lower()
    if mode == "console":
        provider.add_span_processor(
            SimpleSpanProcessor(ConsoleSpanExporter(out=console_stream or sys.stderr))
        )
    elif mode == "cloud_trace":
        from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

        project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or None
        provider.add_span_processor(
            BatchSpanProcessor(CloudTraceSpanExporter(project_id=project_id))
        )
    elif mode == "none":
        pass
    else:
        raise ValueError(
            f"Unsupported TRACE_EXPORTER={mode!r}. Expected 'console', 'cloud_trace', or 'none'."
        )

    _TRACING_CONFIGURED = True
    return provider


def get_tracer() -> trace.Tracer:
    """Return the application OpenTelemetry tracer."""
    return trace.get_tracer(TRACER_NAME, TRACER_VERSION)


def _safe_span_attr(value: Any, *, max_len: int = 500) -> str | int | float | bool:
    """Redact PII and coerce any value into an OpenTelemetry-safe primitive attribute."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    redacted = redact(value)
    if isinstance(redacted, str):
        text = redacted
    else:
        try:
            text = json.dumps(redacted, default=str, sort_keys=True)
        except Exception:
            text = redact(str(redacted))
    return text[:max_len]


def _set_redacted_attributes(span: Span, attributes: dict[str, Any]) -> None:
    """Attach PII-redacted attributes to a span."""
    for key, value in attributes.items():
        if value is None:
            continue
        if key in ("meeting_prep.invocation_id", "meeting_prep.session_id", "meeting_prep.function_call_id"):
            span.set_attribute(key, str(value))
        else:
            span.set_attribute(key, _safe_span_attr(value))


# ---------------------------------------------------------------------------
# 4.2 & 4.3 — ObservabilityPlugin (Intent vs. Outcome + Custom Span Tree)
# ---------------------------------------------------------------------------


def _summarize_tool_result(result: Any) -> dict[str, Any]:
    """Extract compact, safe summary fields from a tool result dict for outcome logs."""
    if not isinstance(result, dict):
        return {"value": redact(str(result))[:200]}
    summary: dict[str, Any] = {}
    for key in (
        "status",
        "policy_violation",
        "meeting_id",
        "draft_id",
        "delivery_id",
        "outcome_id",
        "approval_status",
        "sent",
        "total_matches",
        "total_open",
        "error",
        "error_message",
    ):
        if key in result:
            summary[key] = redact(result[key])
    return summary


def _classify_tool_outcome(
    tool_name: str, result: Any
) -> tuple[str, str | None, str | None]:
    """Classify tool outcome status, optional policy_violation, and optional hitl_decision.

    Returns:
        ``(outcome_status, policy_violation, hitl_decision)`` where
        ``outcome_status`` is one of ``policy_blocked``,
        ``hitl_confirmation_requested``, ``hitl_rejected``, ``hitl_approved``,
        ``error``, or ``ok``.
    """
    if isinstance(result, dict):
        if violation := result.get("policy_violation"):
            return "policy_blocked", str(violation), None

        raw_error = str(result.get("error", "")).lower()
        if "requires confirmation" in raw_error:
            return "hitl_confirmation_requested", None, "confirmation_requested"
        if "tool call is rejected" in raw_error:
            return "hitl_rejected", None, "rejected"

        if tool_name in _HITL_TOOLS and result.get("status") == "ok":
            approval_state = str(result.get("approval_status", "")).lower()
            if approval_state == "rejected":
                return "hitl_rejected", None, "rejected"
            return "hitl_approved", None, "approved"

        if result.get("status") == "error" or "error" in result:
            return "error", None, None

    return "ok", None, None


class ObservabilityPlugin(BasePlugin):
    """ADK BasePlugin capturing Intent vs. Outcome logs and custom OTel spans.

    Register AHEAD of ``PolicyPlugin`` in ``App(plugins=[ObservabilityPlugin(), PolicyPlugin()])``
    so every tool call records its ``intent`` before ``PolicyPlugin`` can
    short-circuit it, and ``after_tool_callback`` records the correlated
    ``outcome`` (including ``policy_blocked``, ``hitl_confirmation_requested``,
    ``hitl_approved``, and ``hitl_rejected``).
    """

    def __init__(
        self,
        *,
        name: str = "meeting_prep_observability_plugin",
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(name=name)
        self._logger = logger or get_structured_logger()
        # Active spans keyed by invocation_id / (invocation_id, name_or_call_id).
        # Explicit parent-context linking avoids cross-context ContextVar errors
        # across ParallelAgent branches and HITL pause/resume turns.
        self._request_spans: dict[str, Span] = {}
        self._workflow_spans: dict[str, Span] = {}
        self._stage_spans: dict[tuple[str, str], Span] = {}
        self._tool_spans: dict[tuple[str, str], Span] = {}
        self._tool_start_times: dict[tuple[str, str], float] = {}
        self._stage_start_times: dict[tuple[str, str], float] = {}

    def _active_parent_context(
        self, invocation_id: str, *, agent_name: str | None = None
    ) -> Any:
        """Resolve the most specific active span context for parenting child spans."""
        if agent_name and (invocation_id, agent_name) in self._stage_spans:
            return trace.set_span_in_context(
                self._stage_spans[(invocation_id, agent_name)]
            )
        if agent_name in _PARALLEL_GATHER_BRANCHES and (
            invocation_id,
            "gather_step",
        ) in self._stage_spans:
            return trace.set_span_in_context(
                self._stage_spans[(invocation_id, "gather_step")]
            )
        if invocation_id in self._workflow_spans:
            return trace.set_span_in_context(self._workflow_spans[invocation_id])
        if invocation_id in self._request_spans:
            return trace.set_span_in_context(self._request_spans[invocation_id])
        return None

    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        """Open a custom root span for this invocation inheriting ADK's trace_id."""
        inv_id = invocation_context.invocation_id
        session = invocation_context.session
        agent_name = getattr(invocation_context.agent, "name", "unknown")

        span = get_tracer().start_span("meeting_prep.request")
        _set_redacted_attributes(
            span,
            {
                "meeting_prep.invocation_id": inv_id,
                "meeting_prep.session_id": getattr(session, "id", ""),
                "meeting_prep.app_name": invocation_context.app_name,
                "meeting_prep.entry_agent": agent_name,
            },
        )
        self._request_spans[inv_id] = span
        return None

    async def before_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> None:
        """Open custom spans wrapping prep_workflow and each workflow stage."""
        inv_id = callback_context.invocation_id
        session_id = getattr(callback_context.session, "id", "")
        agent_name = agent.name

        if agent_name == "prep_workflow":
            parent_ctx = self._active_parent_context(inv_id)
            span = get_tracer().start_span("meeting_prep.prep_flow", context=parent_ctx)
            _set_redacted_attributes(
                span,
                {
                    "meeting_prep.invocation_id": inv_id,
                    "meeting_prep.session_id": session_id,
                    "meeting_prep.workflow": agent_name,
                    "meeting_prep.pattern": "SequentialAgent+ParallelAgent",
                },
            )
            self._workflow_spans[inv_id] = span
            self._stage_start_times[(inv_id, agent_name)] = time.perf_counter()
            emit_structured_log(
                self._logger,
                logging.INFO,
                "Starting meeting prep workflow",
                json_fields={
                    "event_type": "workflow_start",
                    "invocation_id": inv_id,
                    "session_id": session_id,
                    "workflow": agent_name,
                },
                span_context=span.get_span_context(),
            )
            return None

        if agent_name in _WORKFLOW_STAGES or agent_name == "prep_coordinator":
            parent_ctx = self._active_parent_context(inv_id, agent_name=agent_name)
            span = get_tracer().start_span(
                f"meeting_prep.stage.{agent_name}", context=parent_ctx
            )
            routing = MODEL_ROUTING_TABLE.get(agent_name, {})
            _set_redacted_attributes(
                span,
                {
                    "meeting_prep.invocation_id": inv_id,
                    "meeting_prep.session_id": session_id,
                    "meeting_prep.stage": agent_name,
                    "meeting_prep.model_tier": routing.get("tier", "orchestrator"),
                    "meeting_prep.model": routing.get("model", "n/a"),
                },
            )
            self._stage_spans[(inv_id, agent_name)] = span
            self._stage_start_times[(inv_id, agent_name)] = time.perf_counter()

        return None

    async def after_agent_callback(
        self, *, agent: BaseAgent, callback_context: CallbackContext
    ) -> None:
        """Close workflow and stage spans with duration and completion metadata."""
        inv_id = callback_context.invocation_id
        agent_name = agent.name
        start_ts = self._stage_start_times.pop((inv_id, agent_name), None)
        duration_ms = (
            round((time.perf_counter() - start_ts) * 1000, 2)
            if start_ts is not None
            else None
        )

        if agent_name == "prep_workflow":
            span = self._workflow_spans.pop(inv_id, None)
            if span is not None:
                _set_redacted_attributes(
                    span,
                    {
                        "meeting_prep.status": "completed",
                        "meeting_prep.duration_ms": duration_ms,
                    },
                )
                span.set_status(Status(StatusCode.OK))
                span.end()
            return None

        span = self._stage_spans.pop((inv_id, agent_name), None)
        if span is not None:
            _set_redacted_attributes(
                span,
                {
                    "meeting_prep.status": "completed",
                    "meeting_prep.duration_ms": duration_ms,
                },
            )
            span.set_status(Status(StatusCode.OK))
            span.end()
        return None

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any] | None:
        """Capture structured pre-execution ``intent`` event and open tool span."""
        inv_id = tool_context.invocation_id
        call_id = tool_context.function_call_id or f"{tool.name}:{time.monotonic_ns()}"
        key = (inv_id, call_id)
        agent_name = getattr(tool_context, "agent_name", "unknown")
        session_id = getattr(tool_context.session, "id", "")

        self._tool_start_times[key] = time.perf_counter()

        confirmation = getattr(tool_context, "tool_confirmation", None)
        if confirmation is None:
            hitl_state = "unconfirmed" if tool.name in _HITL_TOOLS else "not_applicable"
        elif getattr(confirmation, "confirmed", False):
            hitl_state = "human_approved"
        else:
            hitl_state = "human_rejected"

        redacted_args = redact(tool_args)
        parent_ctx = self._active_parent_context(inv_id, agent_name=agent_name)
        span = get_tracer().start_span(
            f"meeting_prep.tool.{tool.name}", context=parent_ctx
        )
        _set_redacted_attributes(
            span,
            {
                "meeting_prep.invocation_id": inv_id,
                "meeting_prep.session_id": session_id,
                "meeting_prep.function_call_id": call_id,
                "meeting_prep.agent": agent_name,
                "meeting_prep.tool": tool.name,
                "meeting_prep.intent": f"invoke {tool.name}",
                "meeting_prep.tool_args": redacted_args,
                "meeting_prep.hitl_state": hitl_state,
            },
        )
        self._tool_spans[key] = span

        # Record literal "intent" structured JSON log prior to execution.
        emit_structured_log(
            self._logger,
            logging.INFO,
            f"Tool intent: {agent_name} intends to execute {tool.name}",
            json_fields={
                "event_type": "tool_intent",
                "invocation_id": inv_id,
                "function_call_id": call_id,
                "session_id": session_id,
                "agent_name": agent_name,
                "tool_name": tool.name,
                "intent": {
                    "action": tool.name,
                    "args": redacted_args,
                    "hitl_state": hitl_state,
                },
            },
            span_context=span.get_span_context(),
        )
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Capture correlated post-execution ``outcome`` event with duration and status."""
        inv_id = tool_context.invocation_id
        call_id = tool_context.function_call_id or ""
        key = (inv_id, call_id)
        if key not in self._tool_start_times:
            # Fallback if function_call_id was synthesized in before_tool_callback
            match = next(
                (k for k in self._tool_start_times if k[0] == inv_id and k[1].startswith(f"{tool.name}:")),
                None,
            )
            if match is not None:
                key = match

        start_ts = self._tool_start_times.pop(key, None)
        duration_ms = (
            round((time.perf_counter() - start_ts) * 1000, 2)
            if start_ts is not None
            else 0.0
        )
        agent_name = getattr(tool_context, "agent_name", "unknown")
        session_id = getattr(tool_context.session, "id", "")

        outcome_status, policy_violation, hitl_decision = _classify_tool_outcome(
            tool.name, result
        )
        result_summary = _summarize_tool_result(result)

        span = self._tool_spans.pop(key, None)
        span_ctx = span.get_span_context() if span is not None else None
        if span is not None:
            _set_redacted_attributes(
                span,
                {
                    "meeting_prep.outcome": outcome_status,
                    "meeting_prep.duration_ms": duration_ms,
                    "meeting_prep.policy_violation": policy_violation,
                    "meeting_prep.hitl_decision": hitl_decision,
                },
            )
            if outcome_status in ("policy_blocked", "hitl_rejected", "error"):
                span.set_status(Status(StatusCode.ERROR, outcome_status))
            else:
                span.set_status(Status(StatusCode.OK))
            span.end()

        log_level = (
            logging.WARNING
            if outcome_status in ("policy_blocked", "hitl_rejected", "error")
            else logging.INFO
        )
        # Record literal "outcome" structured JSON log correlated by invocation_id.
        emit_structured_log(
            self._logger,
            log_level,
            f"Tool outcome: {tool.name} finished with status={outcome_status}",
            json_fields={
                "event_type": "tool_outcome",
                "invocation_id": inv_id,
                "function_call_id": call_id or key[1],
                "session_id": session_id,
                "agent_name": agent_name,
                "tool_name": tool.name,
                "duration_ms": duration_ms,
                "status": outcome_status,
                "outcome": {
                    "status": outcome_status,
                    "duration_ms": duration_ms,
                    "policy_violation": policy_violation,
                    "hitl_decision": hitl_decision,
                    "summary": result_summary,
                },
            },
            span_context=span_ctx,
        )
        return None

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> dict[str, Any] | None:
        """Capture structured ``outcome`` event if a tool raises an unexpected exception."""
        inv_id = tool_context.invocation_id
        call_id = tool_context.function_call_id or ""
        key = (inv_id, call_id)
        start_ts = self._tool_start_times.pop(key, None)
        duration_ms = (
            round((time.perf_counter() - start_ts) * 1000, 2)
            if start_ts is not None
            else 0.0
        )
        agent_name = getattr(tool_context, "agent_name", "unknown")
        session_id = getattr(tool_context.session, "id", "")
        redacted_err = redact(f"{type(error).__name__}: {error}")

        span = self._tool_spans.pop(key, None)
        span_ctx = span.get_span_context() if span is not None else None
        if span is not None:
            _set_redacted_attributes(
                span,
                {
                    "meeting_prep.outcome": "exception",
                    "meeting_prep.duration_ms": duration_ms,
                    "meeting_prep.error": redacted_err,
                },
            )
            span.set_status(Status(StatusCode.ERROR, redacted_err))
            span.end()

        emit_structured_log(
            self._logger,
            logging.ERROR,
            f"Tool outcome: {tool.name} raised exception",
            json_fields={
                "event_type": "tool_outcome",
                "invocation_id": inv_id,
                "function_call_id": call_id,
                "session_id": session_id,
                "agent_name": agent_name,
                "tool_name": tool.name,
                "duration_ms": duration_ms,
                "status": "exception",
                "outcome": {
                    "status": "exception",
                    "duration_ms": duration_ms,
                    "error": redacted_err,
                },
            },
            span_context=span_ctx,
        )
        return None

    async def after_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        """Cleanly close any remaining spans (including HITL-paused workflow spans)."""
        inv_id = invocation_context.invocation_id

        for key in [k for k in self._tool_spans if k[0] == inv_id]:
            span = self._tool_spans.pop(key)
            self._tool_start_times.pop(key, None)
            span.set_attribute("meeting_prep.paused_for_hitl", True)
            span.end()

        for key in [k for k in self._stage_spans if k[0] == inv_id]:
            span = self._stage_spans.pop(key)
            self._stage_start_times.pop(key, None)
            span.set_attribute("meeting_prep.paused_for_hitl", True)
            span.end()

        wf_span = self._workflow_spans.pop(inv_id, None)
        if wf_span is not None:
            self._stage_start_times.pop((inv_id, "prep_workflow"), None)
            wf_span.set_attribute("meeting_prep.paused_for_hitl", True)
            wf_span.end()

        req_span = self._request_spans.pop(inv_id, None)
        if req_span is not None:
            req_span.set_status(Status(StatusCode.OK))
            req_span.end()
        return None
