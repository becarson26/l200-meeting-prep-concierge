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

"""Unit tests for PII redaction utilities (Rubric 4.4)."""

from __future__ import annotations

from google.adk.events import Event
from google.adk.sessions import Session
from google.genai import types

from app.redaction import (
    REDACTED_EMAIL,
    REDACTED_NUMBER,
    REDACTED_PHONE,
    redact,
    redact_session_for_memory,
)


def test_redact_masks_emails() -> None:
    text = "Contact nadia.okafor@northwind.example or external.user+tag@corp.co.uk ASAP."
    result = redact(text)
    assert "nadia.okafor@northwind.example" not in result
    assert "external.user+tag@corp.co.uk" not in result
    assert result.count(REDACTED_EMAIL) == 2


def test_redact_masks_phone_numbers() -> None:
    samples = [
        "Call +1-555-234-5678 for bridge details",
        "UK desk: +44 20 7946 0958",
        "Office (555) 234-5678 ext 4",
        "Direct 555.234.5678 or 555-234-5678",
        "Compact +15552345678",
    ]
    for sample in samples:
        scrubbed = redact(sample)
        assert REDACTED_PHONE in scrubbed, f"Failed to mask phone in: {sample!r} -> {scrubbed!r}"
        assert "555" not in scrubbed and "7946" not in scrubbed


def test_redact_masks_long_digit_strings_ssns_and_cards() -> None:
    text = "Account 987654321012, card 4111-2222-3333-4444, SSN 123-45-6789, pin 87654321."
    scrubbed = redact(text)
    for sensitive in ("987654321012", "4111-2222-3333-4444", "123-45-6789", "87654321"):
        assert sensitive not in scrubbed
    assert scrubbed.count(REDACTED_NUMBER) == 4


def test_redact_preserves_domain_ids_iso_dates_and_correlation_keys() -> None:
    payload = {
        "trace_id": "12345678901234567890123456789012",
        "span_id": "1234567890123456",
        "invocation_id": "e-12345678-abcd",
        "meeting_id": "M-005",
        "action_item_id": "AI-002",
        "draft_id": "DRF-301514F3",
        "date": "2026-11-25",
        "note": "Brief for M-005 on 2026-11-25 at 15:00 UTC sent to callum.macleod@northwind.example",
        "nested": ["phone (555) 867-5309", "account 4455667788"],
    }
    scrubbed = redact(payload)

    # Correlation keys preserved verbatim
    assert scrubbed["trace_id"] == payload["trace_id"]
    assert scrubbed["span_id"] == payload["span_id"]
    assert scrubbed["invocation_id"] == payload["invocation_id"]

    # Operational IDs and ISO dates preserved
    assert scrubbed["meeting_id"] == "M-005"
    assert scrubbed["action_item_id"] == "AI-002"
    assert scrubbed["draft_id"] == "DRF-301514F3"
    assert scrubbed["date"] == "2026-11-25"
    assert "M-005 on 2026-11-25" in scrubbed["note"]

    # PII inside nested values redacted
    assert REDACTED_EMAIL in scrubbed["note"]
    assert "callum.macleod@northwind.example" not in scrubbed["note"]
    assert scrubbed["nested"] == [f"phone {REDACTED_PHONE}", f"account {REDACTED_NUMBER}"]


def test_redact_session_for_memory_scrubs_copy_without_mutating_live_session() -> None:
    live_session = Session(
        id="sess-1",
        app_name="app",
        user_id="tester",
        events=[
            Event(
                invocation_id="inv-1",
                author="user",
                content=types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            text="Reach me at priya.subramanian@northwind.example or 555-321-7654; UK English preference."
                        )
                    ],
                ),
            ),
            Event(
                invocation_id="inv-1",
                author="prep_coordinator",
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                name="lookup_attendee_profile",
                                args={"email": "priya.subramanian@northwind.example"},
                            )
                        )
                    ],
                ),
            ),
        ],
    )

    scrubbed = redact_session_for_memory(live_session)

    # Scrubbed snapshot has PII masked while non-PII facts survive
    user_text = scrubbed.events[0].content.parts[0].text
    assert REDACTED_EMAIL in user_text
    assert REDACTED_PHONE in user_text
    assert "UK English preference" in user_text
    assert scrubbed.events[1].content.parts[0].function_call.args["email"] == REDACTED_EMAIL

    # Original live session remains completely untouched
    assert "priya.subramanian@northwind.example" in live_session.events[0].content.parts[0].text
