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

"""PII and sensitive-data scrubbing utilities applied before storage (Rubric 4.4).

``redact()`` masks email addresses, phone numbers, and long digit sequences
(including SSNs, payment card numbers, and bare 8+ digit identifiers) while
preserving operational domain IDs (``M-005``, ``AI-002``, ``DRF-...``), ISO
dates (``2026-11-25``), and distributed-tracing correlation keys.

Applied across all three observability/storage egress points:
    1. Structured JSON log pipeline (``RedactingLogFilter`` in ``app/observability.py``).
    2. Async background memory ingestion (``schedule_background_memory_ingestion_after_agent``
       in ``app/callbacks.py``).
    3. Custom OpenTelemetry span attributes (``ObservabilityPlugin`` in ``app/observability.py``).
"""

from __future__ import annotations

import re
from typing import Any

# Placeholder tokens emitted by redact().
REDACTED_EMAIL = "[REDACTED_EMAIL]"
REDACTED_PHONE = "[REDACTED_PHONE]"
REDACTED_NUMBER = "[REDACTED_NUMBER]"

# Correlation identifiers that must remain intact so logs and traces join cleanly.
PRESERVED_CORRELATION_KEYS: frozenset[str] = frozenset(
    {
        "trace_id",
        "span_id",
        "parent_span_id",
        "invocation_id",
        "session_id",
        "function_call_id",
        "trace",
        "logging.googleapis.com/trace",
        "logging.googleapis.com/spanId",
    }
)

# 1. Email addresses (including internal @northwind.example addresses).
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# 2. Structured sensitive numbers (evaluated before phone numbers so 16-digit
#    card numbers and 9-digit SSNs are never partially matched as phone numbers).
_CREDIT_CARD_RE = re.compile(r"\b(?:\d{4}[-\s]){3}\d{4}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# 3. Phone numbers in international, parenthesized, or 3-3-4 separated formats.
#    Carefully excludes ISO dates (YYYY-MM-DD) and short codes.
_PHONE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # +1-555-234-5678, +44 20 7946 0958, +1 (555) 234-5678
    re.compile(r"(?<!\w)\+\d{1,3}[\s.-](?:\(?\d{1,4}\)?[\s.-]){2,4}\d{2,4}\b"),
    # +15552345678
    re.compile(r"(?<!\w)\+\d{9,15}\b"),
    # (555) 234-5678, (555)-234-5678
    re.compile(r"(?<!\w)\(\d{3}\)[\s.-]?\d{3}[\s.-]\d{4}\b"),
    # 555-234-5678, 555.234.5678, 555 234 5678
    re.compile(r"(?<!\w)\d{3}[\s.-]\d{3}[\s.-]\d{4}\b"),
)

# 4. Bare long digit strings (8+ consecutive digits not adjacent to hex/word chars,
#    preserving mixed-alphanumeric IDs like DRF-301514F3 or 32-char hex trace IDs).
_LONG_DIGITS_RE = re.compile(r"(?<![A-Za-z0-9])\d{8,}(?![A-Za-z0-9])")


def redact_string(text: str) -> str:
    """Scrub emails, phone numbers, and long digit strings from a single string."""
    if not text:
        return text
    scrubbed = _EMAIL_RE.sub(REDACTED_EMAIL, text)
    scrubbed = _CREDIT_CARD_RE.sub(REDACTED_NUMBER, scrubbed)
    scrubbed = _SSN_RE.sub(REDACTED_NUMBER, scrubbed)
    for pattern in _PHONE_PATTERNS:
        scrubbed = pattern.sub(REDACTED_PHONE, scrubbed)
    scrubbed = _LONG_DIGITS_RE.sub(REDACTED_NUMBER, scrubbed)
    return scrubbed


def redact(value: Any) -> Any:
    """Recursively mask PII in strings, dicts, lists, and tuples.

    Preserves non-string primitives and explicit distributed-tracing correlation
    dictionary keys (``trace_id``, ``span_id``, ``invocation_id``, etc.).
    """
    if isinstance(value, str):
        return redact_string(value)
    if isinstance(value, dict):
        redacted_dict: dict[Any, Any] = {}
        for k, v in value.items():
            if isinstance(k, str) and k in PRESERVED_CORRELATION_KEYS:
                redacted_dict[k] = v
            else:
                redacted_dict[k] = redact(v)
        return redacted_dict
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact(item) for item in value)
    return value


def redact_session_for_memory(session: Any) -> Any:
    """Return a deep-copied session snapshot with PII scrubbed from all events.

    Called on the async memory-write path so sensitive emails, phone numbers,
    and account/card numbers are actively masked before ``MemoryService``
    persists or indexes session events, without mutating the live session.
    """
    snapshot = (
        session.model_copy(deep=True) if hasattr(session, "model_copy") else session
    )
    events = getattr(snapshot, "events", None) or []
    for event in events:
        content = getattr(event, "content", None)
        parts = getattr(content, "parts", None) if content is not None else None
        if not parts:
            continue
        for part in parts:
            if getattr(part, "text", None):
                part.text = redact_string(part.text)
            fc = getattr(part, "function_call", None)
            if fc is not None and isinstance(getattr(fc, "args", None), dict):
                fc.args = redact(fc.args)
            fr = getattr(part, "function_response", None)
            if fr is not None and isinstance(getattr(fr, "response", None), dict):
                fr.response = redact(fr.response)
    return snapshot
