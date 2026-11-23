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

"""Model I/O guardrail callbacks for the Meeting Prep Concierge (Rubric 3.3).

Three-layer guardrail architecture:
    1. ``PolicyPlugin`` (``app/plugins.py``) — App-level ``BasePlugin`` enforcing
       deterministic pre-tool policies (malformed IDs, brief size, invite list).
    2. Model callbacks (this file) — ``before_model_callback`` screens user
       inputs for prompt injection before calling Gemini; ``after_model_callback``
       validates model output shape and hygiene before returning to callers.
    3. Self-evaluation pass (``critic_agent`` in ``app/agent.py``) — dedicated
       LLM critic auditing every drafted brief against grounded source dossiers
       and account history prior to the human approval gate.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types

from app.config import HISTORY_WINDOW

MAX_MODEL_OUTPUT_CHARS = 15_000

# Strong references to fire-and-forget background memory ingestion tasks so
# Python's garbage collector never cancels an in-flight task mid-consolidation.
_BACKGROUND_MEMORY_TASKS: set[asyncio.Task[Any]] = set()

# ---------------------------------------------------------------------------
# Policy 1 — Prompt Injection & Approval-Bypass Input Screening
# Enforces: User messages must not attempt instruction overrides, system-prompt
#           exfiltration, jailbreak roleplay, or bypassing human approval gates.
# Why:      Stops adversarial prompts *before* they consume LLM tokens or steer
#           the model into calling sensitive tools with attacker-controlled args.
# ---------------------------------------------------------------------------
_PROMPT_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        re.compile(
            r"\b(ignore|disregard|forget|override)\s+(all\s+|any\s+)?"
            r"(previous|prior|above|system|earlier)\s+"
            r"(instructions|rules|prompts?|constraints?|directions?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "system_prompt_exfiltration",
        re.compile(
            r"\b(reveal|print|dump|show|output|repeat)\s+(me\s+)?(your\s+|the\s+)?"
            r"(system\s+prompt|system\s+instructions|hidden\s+instructions|developer\s+prompt)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "approval_or_policy_bypass",
        re.compile(
            r"\b(bypass|skip|disable|circumvent|ignore)\s+(the\s+)?"
            r"(human\s+)?(approval|confirmation|policy|guardrail|safety\s+check)\b"
            r"|\bsend\b.{0,40}\bwithout\s+(human\s+|user\s+)?(approval|confirmation)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "roleplay_jailbreak",
        re.compile(
            r"\b(you\s+are\s+now|pretend\s+to\s+be|act\s+as)\s+"
            r"(dan|an?\s+unrestricted|jailbroken)\b"
            r"|<\s*/?\s*system\s*>|\[SYSTEM\s+OVERRIDE\]",
            re.IGNORECASE,
        ),
    ),
)

# ---------------------------------------------------------------------------
# Policy 2 — Model Output Shape & Hygiene Validation
# Enforces: Non-tool model responses must be non-empty, bounded within
#           ``MAX_MODEL_OUTPUT_CHARS``, and free of raw Python tracebacks or
#           leaked internal prompt delimiters.
# Why:      Prevents empty/broken UI turns, runaway generation, and accidental
#           disclosure of internal stack traces or prompt scaffolding.
# ---------------------------------------------------------------------------
_FORBIDDEN_OUTPUT_MARKERS: tuple[str, ...] = (
    "Traceback (most recent call last):",
    "=== SOURCE ATTENDEE DOSSIERS ===",
    "=== SOURCE ACCOUNT HISTORY ===",
)


def _extract_latest_user_text(llm_request: LlmRequest) -> str:
    """Return text from the most recent user-authored text turn (excluding tool responses)."""
    if not llm_request.contents:
        return ""
    for content in reversed(llm_request.contents):
        if content.role != "user" or not content.parts:
            continue
        text_parts = [
            part.text
            for part in content.parts
            if getattr(part, "text", None) and not getattr(part, "function_response", None)
        ]
        if text_parts:
            return "\n".join(text_parts)
    return ""


def screen_prompt_injection_before_model(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> LlmResponse | None:
    """Screen the latest user message for prompt injection before calling Gemini.

    Returns a refusal ``LlmResponse`` to short-circuit the model call when an
    injection pattern matches; returns ``None`` to allow normal execution.
    """
    latest_user_text = _extract_latest_user_text(llm_request)
    if not latest_user_text:
        return None

    for rule_name, pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern.search(latest_user_text):
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part.from_text(
                            text=(
                                "I cannot process that request because it triggered a "
                                f"security guardrail (`{rule_name}`). I must follow "
                                "Northwind Telecom's meeting-preparation policies, including "
                                "requiring explicit human confirmation before sending any "
                                "brief. Please rephrase your meeting-prep question normally."
                            )
                        )
                    ],
                )
            )

    return None


def validate_output_shape_after_model(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> LlmResponse | None:
    """Validate model output shape and hygiene before returning to the caller.

    Passes tool-call turns and streaming partials through untouched. Returns a
    sanitized replacement ``LlmResponse`` only if the text output is empty,
    leaks forbidden internal markers, or exceeds ``MAX_MODEL_OUTPUT_CHARS``.
    """
    if llm_response.error_code or llm_response.partial:
        return None

    content = llm_response.content
    parts = list(content.parts) if content and content.parts else []

    # Function-call turns (tools or agent transfers) are structurally valid.
    if any(getattr(part, "function_call", None) is not None for part in parts):
        return None

    text_chunks = [part.text for part in parts if getattr(part, "text", None)]
    combined_text = "\n".join(text_chunks).strip()

    if not combined_text:
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[
                    types.Part.from_text(
                        text=(
                            "I generated an empty response while processing your request. "
                            "Please retry or specify a Northwind Telecom meeting ID (e.g. M-005)."
                        )
                    )
                ],
            )
        )

    for marker in _FORBIDDEN_OUTPUT_MARKERS:
        if marker in combined_text:
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part.from_text(
                            text=(
                                "A draft response was withheld by the output-hygiene "
                                "guardrail because it contained internal diagnostic markers. "
                                "Please retry your request."
                            )
                        )
                    ],
                )
            )

    if len(combined_text) > MAX_MODEL_OUTPUT_CHARS:
        truncated = (
            combined_text[: MAX_MODEL_OUTPUT_CHARS - 80].rstrip()
            + "\n\n*[Output truncated by output-shape guardrail to stay within length limits.]*"
        )
        return LlmResponse(
            content=types.Content(
                role="model",
                parts=[types.Part.from_text(text=truncated)],
            )
        )

    return None


# ---------------------------------------------------------------------------
# Policy 3 — Complementary Sliding-Window Context Guard (Rubric 2.2)
# Enforces: Bounds ``llm_request.contents`` to the most recent ``HISTORY_WINDOW``
#           user turns (default 12), always cutting cleanly on a user-text turn
#           boundary so function_call / function_response pairs stay intact.
# Why:      Complements ADK's post-invocation ``EventsCompactionConfig`` by
#           capping prompt context growth immediately on every model call.
#           IMPORTANT: This only trims the ephemeral model request window and
#           NEVER mutates or drops durable ``user:``-scoped state stored in
#           ``DatabaseSessionService``.
# ---------------------------------------------------------------------------
def _is_user_text_turn(content: types.Content) -> bool:
    """Return True when a content entry starts a real user text turn (not a tool response)."""
    if content.role != "user" or not content.parts:
        return False
    has_text = any(getattr(part, "text", None) for part in content.parts)
    has_func_response = any(
        getattr(part, "function_response", None) is not None
        for part in content.parts
    )
    return has_text and not has_func_response


def enforce_sliding_history_window_before_model(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> LlmResponse | None:
    """Trim ephemeral model-request history to the last HISTORY_WINDOW user turns.

    Compaction and this sliding window bound only the conversational messages
    sent to Gemini; durable ``user:``-scoped preferences and context remain
    untouched in the session store.
    """
    if not llm_request.contents:
        return None

    try:
        window = max(1, int(os.getenv("HISTORY_WINDOW", str(HISTORY_WINDOW))))
    except ValueError:
        window = max(1, HISTORY_WINDOW)

    user_turn_indices = [
        idx
        for idx, content in enumerate(llm_request.contents)
        if _is_user_text_turn(content)
    ]
    if len(user_turn_indices) > window:
        cutoff = user_turn_indices[-window]
        llm_request.contents = list(llm_request.contents[cutoff:])

    return None


# ---------------------------------------------------------------------------
# Async Fire-and-Forget Memory Ingestion (Rubric 2.4)
# Enforces: Completed sessions are ingested into ADK's configured MemoryService
#           on a background asyncio task after the reply is produced.
# Why:      Memory extraction, indexing, and Vertex AI Memory Bank consolidation
#           are expensive I/O/LLM operations. Scheduling them with
#           ``asyncio.create_task`` and deliberately NOT awaiting them guarantees
#           that memory consolidation never blocks or adds latency to user replies.
# ---------------------------------------------------------------------------
async def _ingest_session_into_memory_background(
    memory_service: Any,
    session_snapshot: Any,
) -> None:
    """Background coroutine ingesting a snapshot of a finished session into MemoryService."""
    try:
        await memory_service.add_session_to_memory(session_snapshot)
    except Exception:
        # Background consolidation is best-effort and must never surface an
        # unhandled exception into the foreground request loop.
        return


def schedule_background_memory_ingestion_after_agent(
    callback_context: CallbackContext,
) -> types.Content | None:
    """Fire-and-forget after_agent_callback scheduling async memory ingestion.

    Intentionally synchronous and deliberately does NOT await ingestion:
    ``asyncio.create_task(...)`` queues ``memory_service.add_session_to_memory``
    on the running event loop after the agent has finished composing its reply,
    so user-facing response latency is never blocked by memory consolidation.
    """
    invocation_context = getattr(callback_context, "_invocation_context", None)
    memory_service = getattr(invocation_context, "memory_service", None)
    session = getattr(invocation_context, "session", None)
    if memory_service is None or session is None:
        return None

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None

    # Snapshot the session before handing it to the background task so quick
    # follow-up turns cannot mutate event lists while ingestion reads them.
    session_snapshot = (
        session.model_copy(deep=True)
        if hasattr(session, "model_copy")
        else session
    )

    # Deliberately NOT awaited: fire-and-forget so the reply returns immediately
    # without waiting on memory indexing or Vertex AI Memory Bank consolidation.
    task = loop.create_task(
        _ingest_session_into_memory_background(memory_service, session_snapshot),
        name=f"memory-ingest-{getattr(session, 'id', 'unknown')}",
    )
    _BACKGROUND_MEMORY_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_MEMORY_TASKS.discard)
    return None


async def flush_background_memory_tasks() -> None:
    """Await any in-flight background memory tasks (for deterministic tests/shutdown)."""
    if _BACKGROUND_MEMORY_TASKS:
        await asyncio.gather(*list(_BACKGROUND_MEMORY_TASKS), return_exceptions=True)
