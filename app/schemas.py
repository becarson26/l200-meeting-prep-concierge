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

"""Strict Pydantic v2 input, domain, output, and error schemas for Meeting Prep Concierge tools.

Every model inherits from :class:`StrictSchema` with ``ConfigDict(extra="forbid")``,
bounded string/list/numeric constraints, ``Literal`` enums for constrained choices,
and explicit ``Field(description=...)`` metadata on every attribute.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Constrained Literal Enums & Shared Validation Patterns
# ---------------------------------------------------------------------------

ToolStatus = Literal["ok", "error"]
MeetingStatus = Literal["past", "upcoming"]
ActionItemStatus = Literal["open", "closed"]
PriorityLevel = Literal["high", "medium", "low"]
AudienceFocus = Literal["executive", "technical", "commercial", "balanced"]
ApprovalStatus = Literal["draft", "pending_approval", "approved", "rejected"]
MeetingSentiment = Literal["positive", "neutral", "concerned", "escalated"]
NoteSource = Literal["seed_notes", "recorded_outcome"]

MEETING_ID_PATTERN = r"^M-\d{3,6}$"
ACTION_ITEM_ID_PATTERN = r"^AI-\d{3,6}$"
DRAFT_ID_PATTERN = r"^DRF-[A-Z0-9]{6,12}$"
DELIVERY_ID_PATTERN = r"^MSG-[A-Z0-9]{6,12}$"
OUTCOME_ID_PATTERN = r"^OUT-[A-Z0-9]{6,12}$"
ISO_DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}$"
UTC_TIME_PATTERN = r"^\d{2}:\d{2}$"
EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"

_EMAIL_RE = re.compile(EMAIL_PATTERN)


def _validate_string_list_items(
    values: list[str] | None,
    *,
    field_label: str,
    min_item_len: int = 2,
    max_item_len: int = 300,
) -> list[str] | None:
    """Validate and trim each string item inside a bounded list field."""
    if values is None:
        return None
    cleaned: list[str] = []
    for idx, raw in enumerate(values):
        item = raw.strip()
        if len(item) < min_item_len:
            raise ValueError(
                f"{field_label}[{idx}] must be at least {min_item_len} characters long."
            )
        if len(item) > max_item_len:
            raise ValueError(
                f"{field_label}[{idx}] must be at most {max_item_len} characters long."
            )
        cleaned.append(item)
    return cleaned


def _validate_email_list(
    values: list[str] | None, *, field_label: str
) -> list[str] | None:
    """Normalize and validate each email entry inside a list."""
    if values is None:
        return None
    cleaned: list[str] = []
    for idx, raw in enumerate(values):
        addr = raw.strip().lower()
        if len(addr) < 5 or len(addr) > 120 or not _EMAIL_RE.match(addr):
            raise ValueError(
                f"{field_label}[{idx}] ('{raw}') is not a valid email address."
            )
        cleaned.append(addr)
    return cleaned


class StrictSchema(BaseModel):
    """Base schema enforcing forbidden extra fields and whitespace normalization."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# ---------------------------------------------------------------------------
# Standard Guided Error Schema (Rubric 1.4)
# ---------------------------------------------------------------------------


class ToolErrorResponse(StrictSchema):
    """Uniform structured error returned by every tool instead of raising."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    status: Literal["error"] = Field(
        default="error",
        description="Fixed error status indicator signaling the tool call did not succeed.",
    )
    error_message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Clear, human-readable explanation of what failed without stack traces.",
    )
    recovery_hint: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Concrete next step instructing the LLM how to recover or retry.",
    )


# ---------------------------------------------------------------------------
# Core Domain Models
# ---------------------------------------------------------------------------


class CommunicationPreferences(StrictSchema):
    """How an attendee prefers to receive pre-reads and participate in meetings."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    style: str = Field(
        ...,
        min_length=3,
        max_length=300,
        description="Preferred communication framing and level of technical or financial detail.",
    )
    pre_read_lead_time_hours: int = Field(
        ...,
        ge=0,
        le=168,
        description="Minimum hours before the meeting the attendee prefers to receive a brief.",
    )
    preferred_channel: str = Field(
        ...,
        min_length=3,
        max_length=200,
        description="Preferred format or channel for pre-meeting collaboration.",
    )
    dislikes: str = Field(
        ...,
        min_length=3,
        max_length=300,
        description="Meeting anti-patterns or framing styles to avoid for this attendee.",
    )


class AttendeeProfile(StrictSchema):
    """Directory record for a meeting participant."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Full display name of the attendee.",
    )
    role: str = Field(
        ...,
        min_length=2,
        max_length=120,
        description="Job title or functional role of the attendee.",
    )
    organisation: str = Field(
        ...,
        min_length=2,
        max_length=120,
        description="Company or organisation the attendee belongs to.",
    )
    department: str = Field(
        ...,
        min_length=2,
        max_length=120,
        description="Internal division or department within the organisation.",
    )
    email: str = Field(
        ...,
        min_length=5,
        max_length=120,
        pattern=EMAIL_PATTERN,
        description="Primary email address uniquely identifying the attendee.",
    )
    communication_preferences: CommunicationPreferences = Field(
        ...,
        description="Structured communication preferences used to tailor meeting briefs.",
    )


class MeetingRecord(StrictSchema):
    """Calendar metadata and agenda for a single scheduled or completed meeting."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(
        ...,
        min_length=5,
        max_length=16,
        pattern=MEETING_ID_PATTERN,
        description="Unique meeting identifier (e.g. 'M-005').",
    )
    title: str = Field(
        ...,
        min_length=3,
        max_length=200,
        description="Human-readable meeting title.",
    )
    date: str = Field(
        ...,
        min_length=10,
        max_length=10,
        pattern=ISO_DATE_PATTERN,
        description="Scheduled calendar date in YYYY-MM-DD format.",
    )
    time_utc: str = Field(
        ...,
        min_length=5,
        max_length=5,
        pattern=UTC_TIME_PATTERN,
        description="Start time in 24-hour HH:MM UTC format.",
    )
    duration_minutes: int = Field(
        ...,
        ge=5,
        le=480,
        description="Scheduled duration of the meeting in minutes.",
    )
    status: MeetingStatus = Field(
        ...,
        description="Whether the meeting is in the past or upcoming.",
    )
    attendees: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Email addresses of all invited attendees.",
    )
    agenda: list[str] = Field(
        ...,
        min_length=1,
        max_length=30,
        description="Ordered agenda items planned for the meeting.",
    )


class MeetingNoteRecord(StrictSchema):
    """Recorded notes or outcomes associated with a past meeting."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    meeting_id: str = Field(
        ...,
        min_length=5,
        max_length=16,
        pattern=MEETING_ID_PATTERN,
        description="Meeting identifier these notes belong to.",
    )
    recorded_by: str = Field(
        ...,
        min_length=3,
        max_length=120,
        description="Email or actor label identifying who captured the notes.",
    )
    recorded_at: str = Field(
        ...,
        min_length=10,
        max_length=40,
        description="ISO-8601 UTC timestamp when the notes were recorded.",
    )
    summary: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Narrative summary of discussions and takeaways.",
    )
    key_decisions: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Explicit decisions agreed during the meeting.",
    )
    open_questions: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Unresolved questions carried forward from the meeting.",
    )
    risks_raised: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Risks, blockers, or audit findings flagged during the meeting.",
    )
    source: NoteSource = Field(
        default="seed_notes",
        description="Origin of the note record ('seed_notes' fixture or 'recorded_outcome' runtime entry).",
    )


class ActionItemRecord(StrictSchema):
    """A tracked commitment owned by a participant and linked to a meeting."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: str = Field(
        ...,
        min_length=5,
        max_length=16,
        pattern=ACTION_ITEM_ID_PATTERN,
        description="Unique action item identifier (e.g. 'AI-002').",
    )
    meeting_id: str = Field(
        ...,
        min_length=5,
        max_length=16,
        pattern=MEETING_ID_PATTERN,
        description="Meeting ID where this commitment originated.",
    )
    title: str = Field(
        ...,
        min_length=3,
        max_length=200,
        description="Short actionable title of the commitment.",
    )
    description: str = Field(
        ...,
        min_length=5,
        max_length=600,
        description="Detailed deliverable expectations for completing the commitment.",
    )
    owner: str = Field(
        ...,
        min_length=5,
        max_length=120,
        pattern=EMAIL_PATTERN,
        description="Email address of the person accountable for this action item.",
    )
    due_date: str = Field(
        ...,
        min_length=10,
        max_length=10,
        pattern=ISO_DATE_PATTERN,
        description="Target completion date in YYYY-MM-DD format.",
    )
    priority: PriorityLevel = Field(
        ...,
        description="Urgency tier of the action item.",
    )
    status: ActionItemStatus = Field(
        ...,
        description="Current lifecycle state ('open' or 'closed').",
    )
    completed_date: str | None = Field(
        default=None,
        min_length=10,
        max_length=10,
        pattern=ISO_DATE_PATTERN,
        description="Completion date in YYYY-MM-DD format if closed, otherwise null.",
    )


class AttendeeTailoringNote(StrictSchema):
    """Per-attendee communication guidance embedded inside a meeting brief."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Attendee full name.",
    )
    email: str = Field(
        ...,
        min_length=5,
        max_length=120,
        pattern=EMAIL_PATTERN,
        description="Attendee email address.",
    )
    role: str = Field(
        ...,
        min_length=2,
        max_length=120,
        description="Attendee job title.",
    )
    preferred_channel: str = Field(
        ...,
        min_length=3,
        max_length=200,
        description="How this attendee prefers to review pre-reads.",
    )
    style_guidance: str = Field(
        ...,
        min_length=3,
        max_length=300,
        description="Tailoring tip reflecting this attendee's preferred framing.",
    )
    pre_read_lead_time_hours: int = Field(
        ...,
        ge=0,
        le=168,
        description="Requested pre-read lead time in hours.",
    )


class MeetingBriefDocument(StrictSchema):
    """Structured pre-meeting preparation brief ready for review."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    title: str = Field(
        ...,
        min_length=3,
        max_length=250,
        description="Title of the meeting brief.",
    )
    meeting_id: str = Field(
        ...,
        min_length=5,
        max_length=16,
        pattern=MEETING_ID_PATTERN,
        description="Meeting identifier this brief prepares attendees for.",
    )
    meeting_date: str = Field(
        ...,
        min_length=10,
        max_length=10,
        pattern=ISO_DATE_PATTERN,
        description="Scheduled meeting date in YYYY-MM-DD format.",
    )
    meeting_time_utc: str = Field(
        ...,
        min_length=5,
        max_length=5,
        pattern=UTC_TIME_PATTERN,
        description="Scheduled meeting start time in HH:MM UTC.",
    )
    audience_focus: AudienceFocus = Field(
        ...,
        description="Framing focus chosen for this brief.",
    )
    recipients: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Intended attendee email recipients (not yet sent).",
    )
    executive_summary: str = Field(
        ...,
        min_length=10,
        max_length=1500,
        description="Concise overview of meeting purpose, stakes, and readiness.",
    )
    agenda: list[str] = Field(
        ...,
        min_length=1,
        max_length=30,
        description="Meeting agenda topics covered in the brief.",
    )
    attendee_tailoring_notes: list[AttendeeTailoringNote] = Field(
        default_factory=list,
        max_length=50,
        description="Per-attendee communication preferences and pre-read guidance.",
    )
    open_action_items: list[ActionItemRecord] = Field(
        default_factory=list,
        max_length=50,
        description="Relevant unresolved action items owned by invited attendees.",
    )
    related_past_decisions: list[str] = Field(
        default_factory=list,
        max_length=30,
        description="Key decisions pulled from related prior meetings.",
    )
    talking_points: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="Optional custom talking points requested by the user.",
    )
    rendered_markdown: str = Field(
        ...,
        min_length=20,
        max_length=12000,
        description="Human-readable Markdown rendering of the drafted brief.",
    )


# ---------------------------------------------------------------------------
# Runtime Storage Schemas (persisted under runtime/, gitignored)
# ---------------------------------------------------------------------------


class StoredDraftRecord(StrictSchema):
    """Persisted state of a drafted brief and its human-approval lifecycle."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    draft_id: str = Field(
        ...,
        min_length=10,
        max_length=32,
        pattern=DRAFT_ID_PATTERN,
        description="Unique identifier of the staged draft.",
    )
    meeting_id: str = Field(
        ...,
        min_length=5,
        max_length=16,
        pattern=MEETING_ID_PATTERN,
        description="Meeting identifier associated with this draft.",
    )
    created_at_utc: str = Field(
        ...,
        min_length=10,
        max_length=40,
        description="ISO-8601 UTC timestamp when the draft was created.",
    )
    approval_status: ApprovalStatus = Field(
        ...,
        description="Current human-approval status of the draft.",
    )
    approval_requested_at_utc: str | None = Field(
        default=None,
        min_length=10,
        max_length=40,
        description="ISO-8601 UTC timestamp when approval was requested, if any.",
    )
    reviewer_note: str = Field(
        default="",
        max_length=500,
        description="Optional note supplied when requesting approval.",
    )
    sent: bool = Field(
        default=False,
        description="Whether this draft has already been written to the simulated outbox.",
    )
    delivery_id: str | None = Field(
        default=None,
        min_length=10,
        max_length=32,
        pattern=DELIVERY_ID_PATTERN,
        description="Delivery identifier once sent, otherwise null.",
    )
    brief: MeetingBriefDocument = Field(
        ...,
        description="Full drafted meeting brief payload.",
    )


class OutboxDeliveryRecord(StrictSchema):
    """Simulated outbound email record appended to runtime/outbox.json."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    delivery_id: str = Field(
        ...,
        min_length=10,
        max_length=32,
        pattern=DELIVERY_ID_PATTERN,
        description="Unique simulated delivery identifier.",
    )
    draft_id: str = Field(
        ...,
        min_length=10,
        max_length=32,
        pattern=DRAFT_ID_PATTERN,
        description="Approved draft identifier that was dispatched.",
    )
    meeting_id: str = Field(
        ...,
        min_length=5,
        max_length=16,
        pattern=MEETING_ID_PATTERN,
        description="Meeting identifier the brief belongs to.",
    )
    subject: str = Field(
        ...,
        min_length=3,
        max_length=260,
        description="Simulated email subject line.",
    )
    recipients: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Recipient email addresses.",
    )
    sent_at_utc: str = Field(
        ...,
        min_length=10,
        max_length=40,
        description="ISO-8601 UTC timestamp of simulated delivery.",
    )
    body_markdown: str = Field(
        ...,
        min_length=20,
        max_length=12000,
        description="Markdown body written to the simulated outbox.",
    )
    simulated: Literal[True] = Field(
        default=True,
        description="Indicates delivery was written locally rather than sent over SMTP.",
    )


class MeetingOutcomeRecord(StrictSchema):
    """Post-meeting learnings persisted to runtime/outcomes.json."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    outcome_id: str = Field(
        ...,
        min_length=10,
        max_length=32,
        pattern=OUTCOME_ID_PATTERN,
        description="Unique identifier of the recorded meeting outcome.",
    )
    meeting_id: str = Field(
        ...,
        min_length=5,
        max_length=16,
        pattern=MEETING_ID_PATTERN,
        description="Meeting identifier this outcome belongs to.",
    )
    recorded_at_utc: str = Field(
        ...,
        min_length=10,
        max_length=40,
        description="ISO-8601 UTC timestamp when the outcome was saved.",
    )
    summary: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Post-meeting summary of results and learnings.",
    )
    key_decisions: list[str] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="Decisions finalized during the meeting.",
    )
    follow_up_actions: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="New follow-up commitments captured after the meeting.",
    )
    risks_identified: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="New or continuing risks surfaced during the meeting.",
    )
    sentiment: MeetingSentiment = Field(
        ...,
        description="Overall stakeholder sentiment observed in the meeting.",
    )


# ---------------------------------------------------------------------------
# Tool Input & Output Models (16 strict models for the 8 tools)
# ---------------------------------------------------------------------------


# 1. fetch_meeting_details
class FetchMeetingDetailsInput(StrictSchema):
    """Validated input for ``fetch_meeting_details``."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    meeting_id: str = Field(
        ...,
        min_length=5,
        max_length=16,
        pattern=MEETING_ID_PATTERN,
        description="Exact meeting identifier in 'M-NNN' format (e.g. 'M-005').",
    )


class FetchMeetingDetailsOutput(StrictSchema):
    """Validated output for ``fetch_meeting_details``."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    status: Literal["ok"] = Field(
        default="ok",
        description="Success status indicator.",
    )
    meeting: MeetingRecord = Field(
        ...,
        description="Complete meeting metadata, attendee list, and agenda.",
    )


# 2. lookup_attendee_profile
class LookupAttendeeProfileInput(StrictSchema):
    """Validated input for ``lookup_attendee_profile``."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    email: str = Field(
        ...,
        min_length=5,
        max_length=120,
        pattern=EMAIL_PATTERN,
        description="Attendee email address to look up (case-insensitive, e.g. 'nadia.okafor@northwind.example').",
    )

    @field_validator("email")
    @classmethod
    def _normalize_email(cls, value: str) -> str:
        return value.strip().lower()


class LookupAttendeeProfileOutput(StrictSchema):
    """Validated output for ``lookup_attendee_profile``."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    status: Literal["ok"] = Field(
        default="ok",
        description="Success status indicator.",
    )
    attendee: AttendeeProfile = Field(
        ...,
        description="Matched attendee profile including role, organisation, and communication preferences.",
    )


# 3. search_past_meeting_notes
class SearchPastMeetingNotesInput(StrictSchema):
    """Validated input for ``search_past_meeting_notes``."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    topic: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description="Optional keyword or phrase to match across summaries, decisions, open questions, and risks.",
    )
    meeting_id: str | None = Field(
        default=None,
        min_length=5,
        max_length=16,
        pattern=MEETING_ID_PATTERN,
        description="Optional past meeting identifier in 'M-NNN' format to restrict the search.",
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of matching note records to return (1-20).",
    )

    @model_validator(mode="after")
    def _require_topic_or_meeting_id(self) -> SearchPastMeetingNotesInput:
        if not self.topic and not self.meeting_id:
            raise ValueError(
                "At least one of 'topic' or 'meeting_id' must be provided."
            )
        return self


class SearchPastMeetingNotesOutput(StrictSchema):
    """Validated output for ``search_past_meeting_notes``."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    status: Literal["ok"] = Field(
        default="ok",
        description="Success status indicator.",
    )
    query_topic: str | None = Field(
        default=None,
        max_length=200,
        description="Topic keyword filter applied to the search, if any.",
    )
    query_meeting_id: str | None = Field(
        default=None,
        max_length=16,
        description="Meeting ID filter applied to the search, if any.",
    )
    total_matches: int = Field(
        ...,
        ge=0,
        le=100,
        description="Total number of matching note records returned.",
    )
    matches: list[MeetingNoteRecord] = Field(
        default_factory=list,
        max_length=20,
        description="Matched meeting notes and recorded post-meeting outcomes.",
    )


# 4. get_open_action_items
class GetOpenActionItemsInput(StrictSchema):
    """Validated input for ``get_open_action_items``."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    owner_email: str | None = Field(
        default=None,
        min_length=5,
        max_length=120,
        pattern=EMAIL_PATTERN,
        description="Optional owner email address to filter open commitments.",
    )
    meeting_id: str | None = Field(
        default=None,
        min_length=5,
        max_length=16,
        pattern=MEETING_ID_PATTERN,
        description="Optional originating meeting ID ('M-NNN') to filter open commitments.",
    )
    priority: PriorityLevel | None = Field(
        default=None,
        description="Optional priority filter ('high', 'medium', or 'low').",
    )
    max_results: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Maximum number of open action items to return (1-50).",
    )

    @field_validator("owner_email")
    @classmethod
    def _normalize_owner_email(cls, value: str | None) -> str | None:
        return value.strip().lower() if value is not None else None


class GetOpenActionItemsOutput(StrictSchema):
    """Validated output for ``get_open_action_items``."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    status: Literal["ok"] = Field(
        default="ok",
        description="Success status indicator.",
    )
    total_open: int = Field(
        ...,
        ge=0,
        le=200,
        description="Count of open action items returned matching the supplied filters.",
    )
    filters_applied: dict[str, str] = Field(
        default_factory=dict,
        description="Key-value summary of filters applied to the open action items query.",
    )
    items: list[ActionItemRecord] = Field(
        default_factory=list,
        max_length=50,
        description="Matching unresolved ('open') action items sorted by due date.",
    )


# 5. draft_meeting_brief
class DraftMeetingBriefInput(StrictSchema):
    """Validated input for ``draft_meeting_brief``."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    meeting_id: str = Field(
        ...,
        min_length=5,
        max_length=16,
        pattern=MEETING_ID_PATTERN,
        description="Target meeting identifier in 'M-NNN' format to draft a brief for.",
    )
    audience_focus: AudienceFocus = Field(
        default="balanced",
        description="Framing focus ('executive', 'technical', 'commercial', or 'balanced').",
    )
    key_talking_points: list[str] | None = Field(
        default=None,
        max_length=10,
        description="Optional list of up to 10 specific talking points to highlight in the brief.",
    )
    include_open_action_items: bool = Field(
        default=True,
        description="Whether to include open commitments owned by invited attendees.",
    )

    @field_validator("key_talking_points")
    @classmethod
    def _check_talking_points(cls, value: list[str] | None) -> list[str] | None:
        return _validate_string_list_items(
            value, field_label="key_talking_points", min_item_len=3, max_item_len=300
        )


class DraftMeetingBriefOutput(StrictSchema):
    """Validated output for ``draft_meeting_brief``."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    status: Literal["ok"] = Field(
        default="ok",
        description="Success status indicator.",
    )
    draft_id: str = Field(
        ...,
        min_length=10,
        max_length=32,
        pattern=DRAFT_ID_PATTERN,
        description="Unique identifier assigned to the staged draft for approval and delivery.",
    )
    meeting_id: str = Field(
        ...,
        min_length=5,
        max_length=16,
        pattern=MEETING_ID_PATTERN,
        description="Meeting identifier this draft targets.",
    )
    approval_status: Literal["draft"] = Field(
        default="draft",
        description="Lifecycle status confirming the brief is only a draft and not approved yet.",
    )
    sent: Literal[False] = Field(
        default=False,
        description="Always False; drafting a brief never sends anything to attendees.",
    )
    brief: MeetingBriefDocument = Field(
        ...,
        description="Complete drafted meeting brief content and Markdown rendering.",
    )


# 6. request_brief_approval
class RequestBriefApprovalInput(StrictSchema):
    """Validated input for ``request_brief_approval``."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    draft_id: str = Field(
        ...,
        min_length=10,
        max_length=32,
        pattern=DRAFT_ID_PATTERN,
        description="Identifier of an existing draft produced by draft_meeting_brief (e.g. 'DRF-A1B2C3D4').",
    )
    reviewer_note: str = Field(
        default="",
        max_length=500,
        description="Optional context or highlight note for the human reviewer.",
    )


class RequestBriefApprovalOutput(StrictSchema):
    """Validated output for ``request_brief_approval``."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    status: Literal["ok"] = Field(
        default="ok",
        description="Success status indicator.",
    )
    draft_id: str = Field(
        ...,
        min_length=10,
        max_length=32,
        pattern=DRAFT_ID_PATTERN,
        description="Draft identifier surfaced for human review.",
    )
    meeting_id: str = Field(
        ...,
        min_length=5,
        max_length=16,
        pattern=MEETING_ID_PATTERN,
        description="Meeting identifier associated with the draft.",
    )
    approval_status: ApprovalStatus = Field(
        ...,
        description="Updated approval state ('pending_approval', 'approved', or 'rejected').",
    )
    recipients: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Attendee emails who will receive the brief once approved.",
    )
    brief_preview: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Truncated preview of the drafted brief for human inspection.",
    )
    reviewer_note: str = Field(
        default="",
        max_length=500,
        description="Reviewer context note attached to the approval request.",
    )
    next_step: str = Field(
        ...,
        min_length=10,
        max_length=400,
        description="Explicit instruction describing what must happen before delivery.",
    )


# 7. send_brief_to_attendees
class SendBriefToAttendeesInput(StrictSchema):
    """Validated input for ``send_brief_to_attendees``."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    draft_id: str = Field(
        ...,
        min_length=10,
        max_length=32,
        pattern=DRAFT_ID_PATTERN,
        description="Identifier of an approved brief draft to deliver.",
    )
    approval_confirmed: bool = Field(
        default=False,
        description="Must be True only after a human explicitly approved the draft shown by request_brief_approval.",
    )
    recipient_override: list[str] | None = Field(
        default=None,
        min_length=1,
        max_length=20,
        description="Optional list of 1-20 recipient emails overriding the meeting's default attendees.",
    )

    @field_validator("recipient_override")
    @classmethod
    def _check_recipients(cls, value: list[str] | None) -> list[str] | None:
        return _validate_email_list(value, field_label="recipient_override")


class SendBriefToAttendeesOutput(StrictSchema):
    """Validated output for ``send_brief_to_attendees``."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    status: Literal["ok"] = Field(
        default="ok",
        description="Success status indicator.",
    )
    delivery_id: str = Field(
        ...,
        min_length=10,
        max_length=32,
        pattern=DELIVERY_ID_PATTERN,
        description="Unique delivery ID assigned to this simulated dispatch.",
    )
    draft_id: str = Field(
        ...,
        min_length=10,
        max_length=32,
        pattern=DRAFT_ID_PATTERN,
        description="Draft identifier that was delivered.",
    )
    meeting_id: str = Field(
        ...,
        min_length=5,
        max_length=16,
        pattern=MEETING_ID_PATTERN,
        description="Meeting identifier for the delivered brief.",
    )
    recipients: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Recipient email addresses recorded in the simulated outbox.",
    )
    sent_at_utc: str = Field(
        ...,
        min_length=10,
        max_length=40,
        description="ISO-8601 UTC timestamp of delivery.",
    )
    outbox_path: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Local path of the simulated outbox JSON file where delivery was logged.",
    )
    simulated: Literal[True] = Field(
        default=True,
        description="Always True to confirm no external email server was contacted.",
    )


# 8. save_meeting_outcome
class SaveMeetingOutcomeInput(StrictSchema):
    """Validated input for ``save_meeting_outcome``."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    meeting_id: str = Field(
        ...,
        min_length=5,
        max_length=16,
        pattern=MEETING_ID_PATTERN,
        description="Meeting identifier ('M-NNN') for which post-meeting learnings are being saved.",
    )
    summary: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Concise summary of what happened and what was learned (10-2000 chars).",
    )
    key_decisions: list[str] = Field(
        ...,
        min_length=1,
        max_length=20,
        description="At least 1 and at most 20 concrete decisions made during the meeting.",
    )
    follow_up_actions: list[str] | None = Field(
        default=None,
        max_length=20,
        description="Optional list of up to 20 follow-up commitments arising from the meeting.",
    )
    risks_identified: list[str] | None = Field(
        default=None,
        max_length=20,
        description="Optional list of up to 20 risks or blockers surfaced during the meeting.",
    )
    sentiment: MeetingSentiment = Field(
        default="neutral",
        description="Observed meeting sentiment ('positive', 'neutral', 'concerned', or 'escalated').",
    )

    @field_validator("key_decisions")
    @classmethod
    def _check_key_decisions(cls, value: list[str]) -> list[str]:
        result = _validate_string_list_items(
            value, field_label="key_decisions", min_item_len=3, max_item_len=300
        )
        assert result is not None
        return result

    @field_validator("follow_up_actions")
    @classmethod
    def _check_follow_up_actions(cls, value: list[str] | None) -> list[str] | None:
        return _validate_string_list_items(
            value, field_label="follow_up_actions", min_item_len=3, max_item_len=300
        )

    @field_validator("risks_identified")
    @classmethod
    def _check_risks_identified(cls, value: list[str] | None) -> list[str] | None:
        return _validate_string_list_items(
            value, field_label="risks_identified", min_item_len=3, max_item_len=300
        )


class SaveMeetingOutcomeOutput(StrictSchema):
    """Validated output for ``save_meeting_outcome``."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    status: Literal["ok"] = Field(
        default="ok",
        description="Success status indicator.",
    )
    outcome_id: str = Field(
        ...,
        min_length=10,
        max_length=32,
        pattern=OUTCOME_ID_PATTERN,
        description="Unique identifier assigned to the stored outcome record.",
    )
    meeting_id: str = Field(
        ...,
        min_length=5,
        max_length=16,
        pattern=MEETING_ID_PATTERN,
        description="Meeting identifier the saved outcome belongs to.",
    )
    recorded_at_utc: str = Field(
        ...,
        min_length=10,
        max_length=40,
        description="ISO-8601 UTC timestamp when the outcome was persisted.",
    )
    sentiment: MeetingSentiment = Field(
        ...,
        description="Recorded sentiment classification.",
    )
    decisions_count: int = Field(
        ...,
        ge=1,
        le=20,
        description="Number of key decisions saved.",
    )
    follow_ups_count: int = Field(
        ...,
        ge=0,
        le=20,
        description="Number of follow-up actions saved.",
    )
    storage_path: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Local runtime path where the outcome was persisted.",
    )
