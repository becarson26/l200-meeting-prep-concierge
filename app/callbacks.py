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

import re

from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types

MAX_MODEL_OUTPUT_CHARS = 15_000

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
