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

"""Meeting Prep Concierge tools backed by local JSON fixtures and runtime files.

Architecture note:
    The data layer below is deliberately file-backed behind thin read/write
    repository helpers so a real Google Calendar, Gmail, Google Docs, or CRM
    backend can be substituted later without changing any tool signature,
    Pydantic schema, or agent prompt.

    Read-only seed fixtures live in ``data/`` (committed to the repository):
        - ``data/meetings.json``
        - ``data/attendees.json``
        - ``data/notes.json``
        - ``data/action_items.json``

    Mutable runtime state lives in ``runtime/`` (gitignored so fixtures stay
    clean and evaluations remain deterministic across runs):
        - ``runtime/drafts.json``
        - ``runtime/outbox.json``
        - ``runtime/outcomes.json``
"""

from __future__ import annotations

import datetime
import json
import os
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import Field, ValidationError

from app.schemas import (
    ActionItemRecord,
    AttendeeProfile,
    AttendeeTailoringNote,
    AudienceFocus,
    DraftMeetingBriefInput,
    DraftMeetingBriefOutput,
    FetchMeetingDetailsInput,
    FetchMeetingDetailsOutput,
    GetOpenActionItemsInput,
    GetOpenActionItemsOutput,
    LookupAttendeeProfileInput,
    LookupAttendeeProfileOutput,
    MeetingBriefDocument,
    MeetingNoteRecord,
    MeetingOutcomeRecord,
    MeetingRecord,
    MeetingSentiment,
    OutboxDeliveryRecord,
    PriorityLevel,
    RequestBriefApprovalInput,
    RequestBriefApprovalOutput,
    SaveMeetingOutcomeInput,
    SaveMeetingOutcomeOutput,
    SearchPastMeetingNotesInput,
    SearchPastMeetingNotesOutput,
    SendBriefToAttendeesInput,
    SendBriefToAttendeesOutput,
    StoredDraftRecord,
    ToolErrorResponse,
)

if TYPE_CHECKING:
    from google.adk.tools import ToolContext

# ---------------------------------------------------------------------------
# File-Backed Repository Layer
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = _REPO_ROOT / "data"
DEFAULT_RUNTIME_DIR = _REPO_ROOT / "runtime"

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
)


def _get_data_dir() -> Path:
    """Resolve the read-only fixtures directory (overridable for tests)."""
    override = os.environ.get("MEETING_PREP_DATA_DIR")
    return Path(override) if override else DEFAULT_DATA_DIR


def _get_runtime_dir() -> Path:
    """Resolve the gitignored mutable runtime state directory."""
    override = os.environ.get("MEETING_PREP_RUNTIME_DIR")
    return Path(override) if override else DEFAULT_RUNTIME_DIR


def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _new_id(prefix: str) -> str:
    """Generate a compact uppercase identifier matching schema regex patterns."""
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"


def _error_response(error_message: str, recovery_hint: str) -> dict[str, Any]:
    """Build a schema-validated uniform error payload (never raises)."""
    return ToolErrorResponse(
        status="error",
        error_message=error_message[:1000],
        recovery_hint=recovery_hint[:1000],
    ).model_dump()


def _format_validation_error(exc: ValidationError) -> str:
    """Summarize Pydantic validation errors concisely without stack traces or URLs."""
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(segment) for segment in err.get("loc", ())) or "input"
        msg = err.get("msg", "invalid value")
        parts.append(f"{loc}: {msg}")
    return "Invalid arguments — " + "; ".join(parts)


def _read_json_file(path: Path) -> Any:
    """Load and parse a required JSON fixture file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_optional_runtime_json(filename: str, default: Any) -> Any:
    """Read an optional mutable JSON file from runtime/, returning default if absent."""
    path = _get_runtime_dir() / filename
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_runtime_json(filename: str, payload: Any) -> Path:
    """Atomically write JSON state inside runtime/, creating runtime/ on demand."""
    runtime_dir = _get_runtime_dir()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    target = runtime_dir / filename
    tmp_path = runtime_dir / f".{filename}.{uuid.uuid4().hex}.tmp"
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    tmp_path.replace(target)
    return target


def _load_meetings() -> list[MeetingRecord]:
    raw = _read_json_file(_get_data_dir() / "meetings.json")
    return [MeetingRecord.model_validate(item) for item in raw]


def _load_attendees() -> list[AttendeeProfile]:
    raw = _read_json_file(_get_data_dir() / "attendees.json")
    return [AttendeeProfile.model_validate(item) for item in raw]


def _load_action_items() -> list[ActionItemRecord]:
    raw = _read_json_file(_get_data_dir() / "action_items.json")
    return [ActionItemRecord.model_validate(item) for item in raw]


def _load_all_notes() -> list[MeetingNoteRecord]:
    """Load seed past-meeting notes and any post-meeting outcomes saved at runtime."""
    records: list[MeetingNoteRecord] = []

    raw_notes = _read_json_file(_get_data_dir() / "notes.json")
    for note_data in raw_notes.values():
        payload = dict(note_data)
        payload.setdefault("source", "seed_notes")
        records.append(MeetingNoteRecord.model_validate(payload))

    raw_outcomes = _read_optional_runtime_json("outcomes.json", default=[])
    for outcome_raw in raw_outcomes:
        outcome = MeetingOutcomeRecord.model_validate(outcome_raw)
        records.append(
            MeetingNoteRecord(
                meeting_id=outcome.meeting_id,
                recorded_by="meeting_prep_concierge",
                recorded_at=outcome.recorded_at_utc,
                summary=outcome.summary,
                key_decisions=outcome.key_decisions,
                open_questions=outcome.follow_up_actions,
                risks_raised=outcome.risks_identified,
                source="recorded_outcome",
            )
        )

    return records


def _load_drafts() -> dict[str, StoredDraftRecord]:
    raw = _read_optional_runtime_json("drafts.json", default={})
    return {
        draft_id: StoredDraftRecord.model_validate(entry)
        for draft_id, entry in raw.items()
    }


def _save_drafts(drafts: dict[str, StoredDraftRecord]) -> Path:
    serialized = {
        draft_id: record.model_dump() for draft_id, record in drafts.items()
    }
    return _write_runtime_json("drafts.json", serialized)


def _format_meeting_catalog(meetings: list[MeetingRecord]) -> str:
    return ", ".join(f"{m.id} ({m.title}, {m.status})" for m in meetings)


def _tokenize_keywords(text: str) -> set[str]:
    tokens = {
        tok
        for tok in re.findall(r"[a-z0-9]+", text.lower())
        if len(tok) >= 3 and tok not in _STOPWORDS
    }
    return tokens


def _note_search_corpus(note: MeetingNoteRecord) -> str:
    return " ".join(
        [
            note.meeting_id,
            note.summary,
            *note.key_decisions,
            *note.open_questions,
            *note.risks_raised,
        ]
    ).lower()


def _render_brief_markdown(
    meeting: MeetingRecord,
    audience_focus: AudienceFocus,
    executive_summary: str,
    tailoring_notes: list[AttendeeTailoringNote],
    open_items: list[ActionItemRecord],
    past_decisions: list[str],
    talking_points: list[str],
) -> str:
    lines: list[str] = [
        f"# Meeting Prep Brief: {meeting.title} ({meeting.id})",
        "",
        f"- **Date & Time:** {meeting.date} at {meeting.time_utc} UTC ({meeting.duration_minutes} min)",
        f"- **Audience Focus:** {audience_focus.title()}",
        f"- **Recipients:** {', '.join(meeting.attendees)}",
        "",
        "## Executive Summary",
        executive_summary,
        "",
        "## Agenda",
    ]
    for idx, item in enumerate(meeting.agenda, start=1):
        lines.append(f"{idx}. {item}")

    if talking_points:
        lines.extend(["", "## Key Talking Points"])
        for point in talking_points:
            lines.append(f"- {point}")

    if open_items:
        lines.extend(["", "## Relevant Open Action Items"])
        for item in open_items:
            lines.append(
                f"- **[{item.priority.upper()}] {item.id}** ({item.owner}, due {item.due_date}): {item.title}"
            )

    if past_decisions:
        lines.extend(["", "## Related Prior Decisions"])
        for decision in past_decisions:
            lines.append(f"- {decision}")

    if tailoring_notes:
        lines.extend(["", "## Attendee Communication Preferences"])
        for note in tailoring_notes:
            lines.append(
                f"- **{note.name}** ({note.role}) — {note.style_guidance} "
                f"Prefers {note.preferred_channel} ({note.pre_read_lead_time_hours}h lead time)."
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool 1: fetch_meeting_details
# ---------------------------------------------------------------------------


def fetch_meeting_details(
    meeting_id: Annotated[
        str,
        Field(
            min_length=5,
            max_length=16,
            pattern=r"^M-\d{3,6}$",
            description="Exact meeting identifier in 'M-NNN' format (e.g. 'M-005').",
        ),
    ],
) -> dict[str, Any]:
    """Load calendar details, invited attendee emails, and agenda for a single meeting.

    Args:
        meeting_id: Exact meeting identifier in ``M-NNN`` format (5-16 chars,
            matching ``^M-\\d{3,6}$``), such as ``"M-005"``.

    Returns:
        On success:
            ``{"status": "ok", "meeting": {"id": str, "title": str, "date": str, "time_utc": str, "duration_minutes": int, "status": "past" | "upcoming", "attendees": list[str], "agenda": list[str]}}``.
        On failure:
            ``{"status": "error", "error_message": str, "recovery_hint": str}``.

    When NOT to use:
        - Do NOT use this tool to read summaries, decisions, or risks from past
          meetings; use ``search_past_meeting_notes`` instead.
        - Do NOT use this tool to inspect attendee roles or communication
          preferences; use ``lookup_attendee_profile`` with each email returned
          here.
        - Do NOT use this tool to list unresolved tasks; use
          ``get_open_action_items``.
    """
    try:
        validated = FetchMeetingDetailsInput.model_validate(
            {"meeting_id": meeting_id}
        )
    except ValidationError as exc:
        try:
            catalog = _format_meeting_catalog(_load_meetings())
            hint = (
                "Provide meeting_id in 'M-NNN' format (for example 'M-005'). "
                f"Available meetings: {catalog}."
            )
        except Exception:
            hint = "Provide meeting_id in 'M-NNN' format (for example 'M-001' through 'M-007')."
        return _error_response(_format_validation_error(exc), hint)

    try:
        meetings = _load_meetings()
    except Exception as exc:
        return _error_response(
            f"Unable to read meetings store: {type(exc).__name__}: {exc}",
            "Verify that data/meetings.json exists and contains valid JSON, then retry fetch_meeting_details.",
        )

    for meeting in meetings:
        if meeting.id == validated.meeting_id:
            return FetchMeetingDetailsOutput(status="ok", meeting=meeting).model_dump()

    return _error_response(
        f"No meeting found with id '{validated.meeting_id}'.",
        f"Retry fetch_meeting_details with one of the available meeting IDs: {_format_meeting_catalog(meetings)}.",
    )


# ---------------------------------------------------------------------------
# Tool 2: lookup_attendee_profile
# ---------------------------------------------------------------------------


def lookup_attendee_profile(
    email: Annotated[
        str,
        Field(
            min_length=5,
            max_length=120,
            pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
            description="Attendee email address to look up (case-insensitive, e.g. 'nadia.okafor@northwind.example').",
        ),
    ],
) -> dict[str, Any]:
    """Look up a single attendee's role, organisation, and communication preferences by email.

    Args:
        email: Primary email address of the attendee (5-120 chars, valid email
            format, case-insensitive), such as ``"nadia.okafor@northwind.example"``.

    Returns:
        On success:
            ``{"status": "ok", "attendee": {"name": str, "role": str, "organisation": str, "department": str, "email": str, "communication_preferences": {"style": str, "pre_read_lead_time_hours": int, "preferred_channel": str, "dislikes": str}}}``.
        On failure:
            ``{"status": "error", "error_message": str, "recovery_hint": str}``.

    When NOT to use:
        - Do NOT use this tool to discover who is invited to a meeting; call
          ``fetch_meeting_details`` first to get attendee emails.
        - Do NOT use this tool to check what tasks a person owns; call
          ``get_open_action_items`` with ``owner_email`` instead.
    """
    try:
        validated = LookupAttendeeProfileInput.model_validate({"email": email})
    except ValidationError as exc:
        return _error_response(
            _format_validation_error(exc),
            "Pass a full email address (e.g. 'nadia.okafor@northwind.example'). "
            "Call fetch_meeting_details first if you need to look up attendee emails for a meeting.",
        )

    try:
        attendees = _load_attendees()
    except Exception as exc:
        return _error_response(
            f"Unable to read attendees directory: {type(exc).__name__}: {exc}",
            "Verify that data/attendees.json exists and contains valid JSON, then retry lookup_attendee_profile.",
        )

    for profile in attendees:
        if profile.email.lower() == validated.email:
            return LookupAttendeeProfileOutput(
                status="ok", attendee=profile
            ).model_dump()

    known = ", ".join(f"{a.name} <{a.email}>" for a in attendees)
    return _error_response(
        f"No attendee profile found for email '{validated.email}'.",
        "Call fetch_meeting_details to confirm exact attendee addresses, or retry "
        f"lookup_attendee_profile with a known directory email: {known}.",
    )


# ---------------------------------------------------------------------------
# Tool 3: search_past_meeting_notes
# ---------------------------------------------------------------------------


def search_past_meeting_notes(
    topic: Annotated[
        str | None,
        Field(
            min_length=1,
            max_length=200,
            description="Optional keyword or phrase to search across summaries, decisions, open questions, and risks.",
        ),
    ] = None,
    meeting_id: Annotated[
        str | None,
        Field(
            min_length=5,
            max_length=16,
            pattern=r"^M-\d{3,6}$",
            description="Optional past meeting identifier in 'M-NNN' format to restrict search results.",
        ),
    ] = None,
    max_results: Annotated[
        int,
        Field(
            ge=1,
            le=20,
            description="Maximum number of matching note records to return (1-20).",
        ),
    ] = 5,
) -> dict[str, Any]:
    """Search recorded notes from past meetings and saved post-meeting outcomes.

    Args:
        topic: Optional keyword or phrase (1-200 chars) matched case-insensitively
            against note summaries, key decisions, open questions, and risks.
        meeting_id: Optional meeting identifier in ``M-NNN`` format to narrow
            results to a specific meeting. At least one of ``topic`` or
            ``meeting_id`` must be provided.
        max_results: Maximum number of note matches to return (integer 1-20,
            default 5).

    Returns:
        On success:
            ``{"status": "ok", "query_topic": str | None, "query_meeting_id": str | None, "total_matches": int, "matches": list[{"meeting_id": str, "recorded_by": str, "recorded_at": str, "summary": str, "key_decisions": list[str], "open_questions": list[str], "risks_raised": list[str], "source": "seed_notes" | "recorded_outcome"}]}``.
        On failure:
            ``{"status": "error", "error_message": str, "recovery_hint": str}``.

    When NOT to use:
        - Do NOT use this tool to inspect an upcoming meeting's schedule,
          attendees, or agenda; call ``fetch_meeting_details`` instead.
        - Do NOT use this tool to check open commitments or due dates; call
          ``get_open_action_items`` instead.
        - Do NOT use this tool to record new learnings after a meeting; call
          ``save_meeting_outcome`` instead.
    """
    try:
        validated = SearchPastMeetingNotesInput.model_validate(
            {
                "topic": topic,
                "meeting_id": meeting_id,
                "max_results": max_results,
            }
        )
    except ValidationError as exc:
        return _error_response(
            _format_validation_error(exc),
            "Supply at least one of 'topic' (1-200 chars, e.g. 'fiber' or 'billing') "
            "or 'meeting_id' (e.g. 'M-001'), with max_results between 1 and 20.",
        )

    try:
        meetings = _load_meetings()
        notes = _load_all_notes()
    except Exception as exc:
        return _error_response(
            f"Unable to read past meeting notes: {type(exc).__name__}: {exc}",
            "Verify that data/meetings.json and data/notes.json are readable JSON files, then retry.",
        )

    meeting_by_id = {m.id: m for m in meetings}
    past_ids = [m.id for m in meetings if m.status == "past"]

    if validated.meeting_id:
        target_meeting = meeting_by_id.get(validated.meeting_id)
        if target_meeting is None:
            return _error_response(
                f"Meeting '{validated.meeting_id}' does not exist.",
                f"Retry search_past_meeting_notes with a past meeting ID ({', '.join(past_ids)}) "
                "or omit meeting_id and search by topic.",
            )
        scoped_notes = [n for n in notes if n.meeting_id == validated.meeting_id]
        if not scoped_notes and target_meeting.status == "upcoming":
            return _error_response(
                f"Meeting '{validated.meeting_id}' ({target_meeting.title}) is upcoming and has no recorded notes yet.",
                f"Call fetch_meeting_details(meeting_id='{validated.meeting_id}') for its agenda, "
                f"or call search_past_meeting_notes with a topic keyword or past meeting ID ({', '.join(past_ids)}).",
            )
    else:
        scoped_notes = notes

    if validated.topic:
        query_lower = validated.topic.lower()
        query_tokens = _tokenize_keywords(validated.topic)
        scored: list[tuple[int, MeetingNoteRecord]] = []
        for note in scoped_notes:
            corpus = _note_search_corpus(note)
            if query_lower in corpus:
                scored.append((100, note))
            elif query_tokens:
                overlap = sum(1 for tok in query_tokens if tok in corpus)
                if overlap > 0:
                    scored.append((overlap, note))
        scored.sort(key=lambda pair: (-pair[0], pair[1].meeting_id))
        matched = [note for _, note in scored[: validated.max_results]]
    else:
        matched = scoped_notes[: validated.max_results]

    if not matched:
        return _error_response(
            f"No meeting notes matched topic={validated.topic!r}, meeting_id={validated.meeting_id!r}.",
            "Try a broader single-word keyword (such as 'fiber', 'billing', 'SLA', or 'router') "
            f"or query a specific past meeting ID directly: {', '.join(past_ids)}.",
        )

    return SearchPastMeetingNotesOutput(
        status="ok",
        query_topic=validated.topic,
        query_meeting_id=validated.meeting_id,
        total_matches=len(matched),
        matches=matched,
    ).model_dump()


# ---------------------------------------------------------------------------
# Tool 4: get_open_action_items
# ---------------------------------------------------------------------------


def get_open_action_items(
    owner_email: Annotated[
        str | None,
        Field(
            min_length=5,
            max_length=120,
            pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
            description="Optional owner email address to filter open commitments.",
        ),
    ] = None,
    meeting_id: Annotated[
        str | None,
        Field(
            min_length=5,
            max_length=16,
            pattern=r"^M-\d{3,6}$",
            description="Optional originating meeting ID ('M-NNN') to filter open commitments.",
        ),
    ] = None,
    priority: Annotated[
        PriorityLevel | None,
        Field(
            description="Optional priority filter ('high', 'medium', or 'low').",
        ),
    ] = None,
    max_results: Annotated[
        int,
        Field(
            ge=1,
            le=50,
            description="Maximum number of open action items to return (1-50).",
        ),
    ] = 20,
) -> dict[str, Any]:
    """Retrieve unresolved ('open') action items, optionally filtered by owner, meeting, or priority.

    Args:
        owner_email: Optional email address of the accountable owner
            (case-insensitive, 5-120 chars).
        meeting_id: Optional originating meeting identifier in ``M-NNN`` format.
        priority: Optional urgency filter (``"high"``, ``"medium"``, or ``"low"``).
        max_results: Maximum number of open items to return (integer 1-50,
            default 20).

    Returns:
        On success:
            ``{"status": "ok", "total_open": int, "filters_applied": dict[str, str], "items": list[{"id": str, "meeting_id": str, "title": str, "description": str, "owner": str, "due_date": str, "priority": "high" | "medium" | "low", "status": "open", "completed_date": None}]}``.
        On failure:
            ``{"status": "error", "error_message": str, "recovery_hint": str}``.

    When NOT to use:
        - Do NOT use this tool to read narrative meeting notes or past
          decisions; call ``search_past_meeting_notes`` instead.
        - Do NOT use this tool to inspect an attendee's role or communication
          style; call ``lookup_attendee_profile`` instead.
        - Do NOT use this tool to record new follow-up actions after a meeting;
          call ``save_meeting_outcome`` instead.
    """
    try:
        validated = GetOpenActionItemsInput.model_validate(
            {
                "owner_email": owner_email,
                "meeting_id": meeting_id,
                "priority": priority,
                "max_results": max_results,
            }
        )
    except ValidationError as exc:
        return _error_response(
            _format_validation_error(exc),
            "Provide valid optional filters: owner_email (full email), meeting_id ('M-NNN'), "
            "priority ('high', 'medium', or 'low'), and max_results (1-50).",
        )

    try:
        attendees = _load_attendees()
        meetings = _load_meetings()
        items = _load_action_items()
    except Exception as exc:
        return _error_response(
            f"Unable to read action items store: {type(exc).__name__}: {exc}",
            "Verify data/action_items.json, data/attendees.json, and data/meetings.json are valid JSON, then retry.",
        )

    known_emails = {a.email.lower() for a in attendees}
    if validated.owner_email and validated.owner_email not in known_emails:
        return _error_response(
            f"Unknown owner_email '{validated.owner_email}'.",
            "Call fetch_meeting_details to verify attendee addresses, or retry with a known owner: "
            f"{', '.join(sorted(known_emails))}.",
        )

    known_meetings = {m.id for m in meetings}
    if validated.meeting_id and validated.meeting_id not in known_meetings:
        return _error_response(
            f"Unknown meeting_id '{validated.meeting_id}'.",
            f"Retry get_open_action_items with a valid meeting ID ({_format_meeting_catalog(meetings)}) "
            "or omit meeting_id to list all open items.",
        )

    open_items = [item for item in items if item.status == "open"]
    filters: dict[str, str] = {}

    if validated.owner_email:
        open_items = [
            item for item in open_items if item.owner.lower() == validated.owner_email
        ]
        filters["owner_email"] = validated.owner_email
    if validated.meeting_id:
        open_items = [
            item for item in open_items if item.meeting_id == validated.meeting_id
        ]
        filters["meeting_id"] = validated.meeting_id
    if validated.priority:
        open_items = [
            item for item in open_items if item.priority == validated.priority
        ]
        filters["priority"] = validated.priority

    open_items.sort(key=lambda item: (item.due_date, item.id))
    limited = open_items[: validated.max_results]

    return GetOpenActionItemsOutput(
        status="ok",
        total_open=len(limited),
        filters_applied=filters,
        items=limited,
    ).model_dump()


# ---------------------------------------------------------------------------
# Tool 5: draft_meeting_brief
# ---------------------------------------------------------------------------


def draft_meeting_brief(
    meeting_id: Annotated[
        str,
        Field(
            min_length=5,
            max_length=16,
            pattern=r"^M-\d{3,6}$",
            description="Target meeting identifier in 'M-NNN' format to draft a brief for.",
        ),
    ],
    audience_focus: Annotated[
        AudienceFocus,
        Field(
            description="Framing focus ('executive', 'technical', 'commercial', or 'balanced').",
        ),
    ] = "balanced",
    key_talking_points: Annotated[
        list[str] | None,
        Field(
            max_length=10,
            description="Optional list of up to 10 specific talking points to highlight in the brief.",
        ),
    ] = None,
    include_open_action_items: Annotated[
        bool,
        Field(
            description="Whether to include open commitments owned by invited attendees.",
        ),
    ] = True,
) -> dict[str, Any]:
    """Assemble and stage a tailored pre-meeting brief draft WITHOUT sending it.

    Combines meeting agenda, invited attendee communication preferences, related
    past meeting decisions, and optional open action items into a structured
    brief saved in ``runtime/drafts.json`` with ``approval_status="draft"`` and
    ``sent=False``.

    Args:
        meeting_id: Target meeting identifier in ``M-NNN`` format (e.g. ``"M-005"``).
        audience_focus: Framing emphasis — one of ``"executive"``, ``"technical"``,
            ``"commercial"``, or ``"balanced"`` (default ``"balanced"``).
        key_talking_points: Optional list of 1-10 short talking points (each
            3-300 chars) to weave into the brief.
        include_open_action_items: Whether to embed unresolved action items owned
            by invited attendees (default ``True``).

    Returns:
        On success:
            ``{"status": "ok", "draft_id": str, "meeting_id": str, "approval_status": "draft", "sent": False, "brief": {"title": str, "meeting_id": str, "meeting_date": str, "meeting_time_utc": str, "audience_focus": str, "recipients": list[str], "executive_summary": str, "agenda": list[str], "attendee_tailoring_notes": list[dict], "open_action_items": list[dict], "related_past_decisions": list[str], "talking_points": list[str], "rendered_markdown": str}}``.
        On failure:
            ``{"status": "error", "error_message": str, "recovery_hint": str}``.

    When NOT to use:
        - Do NOT use this tool to send a brief to attendees; after drafting,
          you MUST call ``request_brief_approval`` and wait for human approval
          before calling ``send_brief_to_attendees``.
        - Do NOT use this tool to record learnings after a meeting has occurred;
          call ``save_meeting_outcome`` instead.
    """
    try:
        validated = DraftMeetingBriefInput.model_validate(
            {
                "meeting_id": meeting_id,
                "audience_focus": audience_focus,
                "key_talking_points": key_talking_points,
                "include_open_action_items": include_open_action_items,
            }
        )
    except ValidationError as exc:
        return _error_response(
            _format_validation_error(exc),
            "Provide meeting_id in 'M-NNN' format, audience_focus in "
            "('executive', 'technical', 'commercial', 'balanced'), and up to 10 "
            "talking points (3-300 chars each).",
        )

    try:
        meetings = _load_meetings()
        attendees = _load_attendees()
        notes = _load_all_notes()
        action_items = _load_action_items()
    except Exception as exc:
        return _error_response(
            f"Unable to load source data for drafting brief: {type(exc).__name__}: {exc}",
            "Verify that data/ JSON fixtures exist and are valid, then retry draft_meeting_brief.",
        )

    meeting = next((m for m in meetings if m.id == validated.meeting_id), None)
    if meeting is None:
        return _error_response(
            f"Cannot draft brief: meeting '{validated.meeting_id}' was not found.",
            f"Call fetch_meeting_details or retry draft_meeting_brief with a valid ID: {_format_meeting_catalog(meetings)}.",
        )

    attendee_by_email = {a.email.lower(): a for a in attendees}
    tailoring_notes: list[AttendeeTailoringNote] = []
    for email in meeting.attendees:
        profile = attendee_by_email.get(email.lower())
        if profile:
            tailoring_notes.append(
                AttendeeTailoringNote(
                    name=profile.name,
                    email=profile.email,
                    role=profile.role,
                    preferred_channel=profile.communication_preferences.preferred_channel,
                    style_guidance=profile.communication_preferences.style,
                    pre_read_lead_time_hours=profile.communication_preferences.pre_read_lead_time_hours,
                )
            )

    invited_set = {e.lower() for e in meeting.attendees}
    relevant_open_items: list[ActionItemRecord] = []
    if validated.include_open_action_items:
        relevant_open_items = sorted(
            [
                item
                for item in action_items
                if item.status == "open" and item.owner.lower() in invited_set
            ],
            key=lambda item: (item.due_date, item.id),
        )

    # Pull prior decisions from past meetings with keyword overlap against this meeting's title/agenda.
    meeting_keywords = _tokenize_keywords(
        meeting.title + " " + " ".join(meeting.agenda)
    )
    related_decisions: list[str] = []
    for note in notes:
        if note.meeting_id == meeting.id:
            continue
        corpus_tokens = _tokenize_keywords(_note_search_corpus(note))
        if len(meeting_keywords & corpus_tokens) >= 2:
            for decision in note.key_decisions:
                entry = f"[{note.meeting_id}] {decision}"
                if entry not in related_decisions:
                    related_decisions.append(entry)
    related_decisions = related_decisions[:15]

    focus_framing = {
        "executive": "Prioritises decision asks, budget/schedule risks, and go/no-go clarity upfront.",
        "technical": "Emphasises architecture dependencies, telemetry thresholds, and failure-mode readiness.",
        "commercial": "Highlights enterprise renewal impact, SLA credit exposure, and customer comms timing.",
        "balanced": "Balances strategic decisions, operational readiness, commercial impact, and security controls.",
    }[validated.audience_focus]

    executive_summary = (
        f"Preparation brief for '{meeting.title}' ({meeting.id}) on {meeting.date} at "
        f"{meeting.time_utc} UTC with {len(meeting.attendees)} Northwind Telecom stakeholders. "
        f"{focus_framing} Covers {len(meeting.agenda)} agenda items, "
        f"{len(relevant_open_items)} relevant open action items, and "
        f"{len(related_decisions)} related prior decisions."
    )

    talking_points = validated.key_talking_points or []
    rendered_md = _render_brief_markdown(
        meeting=meeting,
        audience_focus=validated.audience_focus,
        executive_summary=executive_summary,
        tailoring_notes=tailoring_notes,
        open_items=relevant_open_items,
        past_decisions=related_decisions,
        talking_points=talking_points,
    )

    brief = MeetingBriefDocument(
        title=f"Meeting Prep Brief: {meeting.title}",
        meeting_id=meeting.id,
        meeting_date=meeting.date,
        meeting_time_utc=meeting.time_utc,
        audience_focus=validated.audience_focus,
        recipients=meeting.attendees,
        executive_summary=executive_summary,
        agenda=meeting.agenda,
        attendee_tailoring_notes=tailoring_notes,
        open_action_items=relevant_open_items,
        related_past_decisions=related_decisions,
        talking_points=talking_points,
        rendered_markdown=rendered_md,
    )

    draft_id = _new_id("DRF")
    stored = StoredDraftRecord(
        draft_id=draft_id,
        meeting_id=meeting.id,
        created_at_utc=_utc_now_iso(),
        approval_status="draft",
        sent=False,
        brief=brief,
    )

    try:
        drafts = _load_drafts()
        drafts[draft_id] = stored
        _save_drafts(drafts)
    except Exception as exc:
        return _error_response(
            f"Unable to stage draft in runtime storage: {type(exc).__name__}: {exc}",
            "Check write permissions on the runtime/ directory and retry draft_meeting_brief.",
        )

    return DraftMeetingBriefOutput(
        status="ok",
        draft_id=draft_id,
        meeting_id=meeting.id,
        approval_status="draft",
        sent=False,
        brief=brief,
    ).model_dump()


# ---------------------------------------------------------------------------
# Tool 6: request_brief_approval
# ---------------------------------------------------------------------------


def request_brief_approval(
    draft_id: Annotated[
        str,
        Field(
            min_length=10,
            max_length=32,
            pattern=r"^DRF-[A-Z0-9]{6,12}$",
            description="Identifier of an existing draft produced by draft_meeting_brief (e.g. 'DRF-A1B2C3D4').",
        ),
    ],
    reviewer_note: Annotated[
        str,
        Field(
            max_length=500,
            description="Optional context or highlight note for the human reviewer.",
        ),
    ] = "",
    tool_context: ToolContext | None = None,
) -> dict[str, Any]:
    """Surface a staged meeting brief draft for explicit human approval before sending.

    Transitions the draft in ``runtime/drafts.json`` to ``pending_approval`` (or
    records the human decision if an ADK ``ToolContext`` confirmation response is
    present) and returns a preview for human sign-off. Never sends the brief.

    Args:
        draft_id: Draft identifier returned by ``draft_meeting_brief`` (matching
            ``^DRF-[A-Z0-9]{6,12}$``).
        reviewer_note: Optional reviewer message (0-500 chars) explaining what
            to verify before approving delivery.
        tool_context: Optional ADK runtime ``ToolContext`` injected automatically
            by ADK (hidden from the LLM schema) to trigger native confirmation.

    Returns:
        On success:
            ``{"status": "ok", "draft_id": str, "meeting_id": str, "approval_status": "pending_approval" | "approved" | "rejected", "recipients": list[str], "brief_preview": str, "reviewer_note": str, "next_step": str}``.
        On failure:
            ``{"status": "error", "error_message": str, "recovery_hint": str}``.

    When NOT to use:
        - Do NOT use this tool before a draft exists; call
          ``draft_meeting_brief`` first to obtain a ``draft_id``.
        - Do NOT use this tool to deliver the brief; after the human approves,
          call ``send_brief_to_attendees``.
    """
    try:
        validated = RequestBriefApprovalInput.model_validate(
            {"draft_id": draft_id, "reviewer_note": reviewer_note}
        )
    except ValidationError as exc:
        return _error_response(
            _format_validation_error(exc),
            "Pass the exact draft_id returned by draft_meeting_brief (format 'DRF-XXXXXXXX') "
            "and keep reviewer_note under 500 characters.",
        )

    try:
        drafts = _load_drafts()
    except Exception as exc:
        return _error_response(
            f"Unable to load staged drafts: {type(exc).__name__}: {exc}",
            "Call draft_meeting_brief to create a fresh draft, then retry request_brief_approval.",
        )

    record = drafts.get(validated.draft_id)
    if record is None:
        existing = ", ".join(sorted(drafts.keys())) if drafts else "none"
        return _error_response(
            f"Draft '{validated.draft_id}' was not found.",
            f"Call draft_meeting_brief first to create a draft (existing drafts: {existing}), "
            "then pass its draft_id to request_brief_approval.",
        )

    if record.sent:
        return _error_response(
            f"Draft '{validated.draft_id}' was already delivered (delivery_id={record.delivery_id}).",
            "No further approval is needed for this draft. Call draft_meeting_brief if you want to prepare a revised brief.",
        )

    preview = record.brief.rendered_markdown[:1950]
    if len(record.brief.rendered_markdown) > 1950:
        preview = preview.rstrip() + "\n... [truncated for preview]"

    new_status: Any = "pending_approval"
    next_step = (
        "Present the brief preview and recipients to the user and wait for explicit approval. "
        f"Only after they confirm, call send_brief_to_attendees(draft_id='{record.draft_id}', approval_confirmed=True)."
    )

    if tool_context is not None:
        confirmation = getattr(tool_context, "tool_confirmation", None)
        if confirmation is None:
            request_fn = getattr(tool_context, "request_confirmation", None)
            if callable(request_fn):
                request_fn(
                    hint=(
                        f"Approve sending brief '{record.brief.title}' ({record.draft_id}) "
                        f"to {len(record.brief.recipients)} attendees?"
                    ),
                    payload={
                        "draft_id": record.draft_id,
                        "meeting_id": record.meeting_id,
                        "recipients": record.brief.recipients,
                        "reviewer_note": validated.reviewer_note,
                    },
                )
        elif getattr(confirmation, "confirmed", False):
            new_status = "approved"
            next_step = (
                f"Human approval confirmed. You may now call "
                f"send_brief_to_attendees(draft_id='{record.draft_id}', approval_confirmed=True)."
            )
        else:
            new_status = "rejected"
            next_step = (
                "The human rejected this draft. Ask what changes are needed and call "
                f"draft_meeting_brief(meeting_id='{record.meeting_id}') to produce a revised version."
            )

    updated = record.model_copy(
        update={
            "approval_status": new_status,
            "approval_requested_at_utc": _utc_now_iso(),
            "reviewer_note": validated.reviewer_note,
        }
    )

    try:
        drafts[record.draft_id] = updated
        _save_drafts(drafts)
    except Exception as exc:
        return _error_response(
            f"Unable to persist approval state: {type(exc).__name__}: {exc}",
            "Verify write access to runtime/drafts.json and retry request_brief_approval.",
        )

    return RequestBriefApprovalOutput(
        status="ok",
        draft_id=updated.draft_id,
        meeting_id=updated.meeting_id,
        approval_status=updated.approval_status,
        recipients=updated.brief.recipients,
        brief_preview=preview,
        reviewer_note=updated.reviewer_note,
        next_step=next_step,
    ).model_dump()


# ---------------------------------------------------------------------------
# Tool 7: send_brief_to_attendees
# ---------------------------------------------------------------------------


def send_brief_to_attendees(
    draft_id: Annotated[
        str,
        Field(
            min_length=10,
            max_length=32,
            pattern=r"^DRF-[A-Z0-9]{6,12}$",
            description="Identifier of an approved brief draft to deliver.",
        ),
    ],
    approval_confirmed: Annotated[
        bool,
        Field(
            description="Must be True only after explicit human approval following request_brief_approval.",
        ),
    ] = False,
    recipient_override: Annotated[
        list[str] | None,
        Field(
            min_length=1,
            max_length=20,
            description="Optional list of 1-20 '@northwind.example' emails overriding default attendees.",
        ),
    ] = None,
) -> dict[str, Any]:
    """Perform a simulated delivery of an approved brief by writing to ``runtime/outbox.json``.

    Enforces the human-approval gate: refuses to deliver any draft that has not
    passed through ``request_brief_approval`` and received explicit confirmation.

    Args:
        draft_id: Draft identifier created by ``draft_meeting_brief`` (matching
            ``^DRF-[A-Z0-9]{6,12}$``).
        approval_confirmed: Boolean confirming the human explicitly approved the
            draft presented by ``request_brief_approval`` (default ``False``).
        recipient_override: Optional list of 1-20 recipient emails replacing the
            meeting's default attendee list. Must use ``@northwind.example``.

    Returns:
        On success:
            ``{"status": "ok", "delivery_id": str, "draft_id": str, "meeting_id": str, "recipients": list[str], "sent_at_utc": str, "outbox_path": str, "simulated": True}``.
        On failure:
            ``{"status": "error", "error_message": str, "recovery_hint": str}``.

    When NOT to use:
        - Do NOT use this tool to compose or revise a brief; call
          ``draft_meeting_brief`` first.
        - Do NOT use this tool before requesting human sign-off; call
          ``request_brief_approval`` first and wait for confirmation.
        - Do NOT use this tool to archive post-meeting notes; call
          ``save_meeting_outcome`` instead.
    """
    try:
        validated = SendBriefToAttendeesInput.model_validate(
            {
                "draft_id": draft_id,
                "approval_confirmed": approval_confirmed,
                "recipient_override": recipient_override,
            }
        )
    except ValidationError as exc:
        return _error_response(
            _format_validation_error(exc),
            "Provide a valid draft_id ('DRF-XXXXXXXX'), set approval_confirmed=True only after "
            "human approval, and ensure any recipient_override contains 1-20 valid email addresses.",
        )

    try:
        drafts = _load_drafts()
    except Exception as exc:
        return _error_response(
            f"Unable to load staged drafts: {type(exc).__name__}: {exc}",
            "Call draft_meeting_brief to stage a draft, request approval via request_brief_approval, then retry.",
        )

    record = drafts.get(validated.draft_id)
    if record is None:
        return _error_response(
            f"Cannot send: draft '{validated.draft_id}' was not found.",
            "Call draft_meeting_brief first to generate a valid draft_id, then call request_brief_approval.",
        )

    if record.sent:
        return _error_response(
            f"Draft '{validated.draft_id}' was already sent as '{record.delivery_id}'.",
            "Do not resend the same draft. If updates are needed, call draft_meeting_brief to create a new draft.",
        )

    if record.approval_status == "rejected":
        return _error_response(
            f"Draft '{validated.draft_id}' was rejected during human review and cannot be sent.",
            f"Ask the user for requested revisions and call draft_meeting_brief(meeting_id='{record.meeting_id}') to create a new draft.",
        )

    if record.approval_status == "draft":
        return _error_response(
            f"Delivery blocked: human approval has not been requested for draft '{validated.draft_id}'.",
            f"Call request_brief_approval(draft_id='{validated.draft_id}') first, show the preview to the user, "
            "and wait for explicit confirmation before calling send_brief_to_attendees.",
        )

    if record.approval_status == "pending_approval" and not validated.approval_confirmed:
        return _error_response(
            f"Delivery blocked: draft '{validated.draft_id}' is awaiting explicit human confirmation.",
            "Confirm with the user that they approve sending this brief, then re-call "
            f"send_brief_to_attendees(draft_id='{validated.draft_id}', approval_confirmed=True).",
        )

    recipients = validated.recipient_override or record.brief.recipients
    disallowed = [addr for addr in recipients if not addr.endswith("@northwind.example")]
    if disallowed:
        return _error_response(
            f"Delivery blocked: recipient(s) outside approved Northwind Telecom domain: {', '.join(disallowed)}.",
            "Use only '@northwind.example' addresses returned by fetch_meeting_details or lookup_attendee_profile.",
        )

    delivery_id = _new_id("MSG")
    sent_at = _utc_now_iso()
    outbox_entry = OutboxDeliveryRecord(
        delivery_id=delivery_id,
        draft_id=record.draft_id,
        meeting_id=record.meeting_id,
        subject=f"[Meeting Brief] {record.brief.title} ({record.brief.meeting_date})",
        recipients=recipients,
        sent_at_utc=sent_at,
        body_markdown=record.brief.rendered_markdown,
        simulated=True,
    )

    try:
        outbox_raw = _read_optional_runtime_json("outbox.json", default=[])
        existing_outbox = [
            OutboxDeliveryRecord.model_validate(item).model_dump()
            for item in outbox_raw
        ]
        existing_outbox.append(outbox_entry.model_dump())
        outbox_path = _write_runtime_json("outbox.json", existing_outbox)

        drafts[record.draft_id] = record.model_copy(
            update={
                "approval_status": "approved",
                "sent": True,
                "delivery_id": delivery_id,
            }
        )
        _save_drafts(drafts)
    except Exception as exc:
        return _error_response(
            f"Unable to write simulated delivery to outbox: {type(exc).__name__}: {exc}",
            "Verify write permissions on runtime/outbox.json and retry send_brief_to_attendees.",
        )

    return SendBriefToAttendeesOutput(
        status="ok",
        delivery_id=delivery_id,
        draft_id=record.draft_id,
        meeting_id=record.meeting_id,
        recipients=recipients,
        sent_at_utc=sent_at,
        outbox_path=str(outbox_path),
        simulated=True,
    ).model_dump()


# ---------------------------------------------------------------------------
# Tool 8: save_meeting_outcome
# ---------------------------------------------------------------------------


def save_meeting_outcome(
    meeting_id: Annotated[
        str,
        Field(
            min_length=5,
            max_length=16,
            pattern=r"^M-\d{3,6}$",
            description="Meeting identifier ('M-NNN') for which post-meeting learnings are being saved.",
        ),
    ],
    summary: Annotated[
        str,
        Field(
            min_length=10,
            max_length=2000,
            description="Concise summary of what happened and what was learned (10-2000 chars).",
        ),
    ],
    key_decisions: Annotated[
        list[str],
        Field(
            min_length=1,
            max_length=20,
            description="At least 1 and at most 20 concrete decisions made during the meeting.",
        ),
    ],
    follow_up_actions: Annotated[
        list[str] | None,
        Field(
            max_length=20,
            description="Optional list of up to 20 follow-up commitments arising from the meeting.",
        ),
    ] = None,
    risks_identified: Annotated[
        list[str] | None,
        Field(
            max_length=20,
            description="Optional list of up to 20 risks or blockers surfaced during the meeting.",
        ),
    ] = None,
    sentiment: Annotated[
        MeetingSentiment,
        Field(
            description="Observed meeting sentiment ('positive', 'neutral', 'concerned', or 'escalated').",
        ),
    ] = "neutral",
) -> dict[str, Any]:
    """Record structured post-meeting learnings, decisions, risks, and follow-ups.

    Persists the outcome to ``runtime/outcomes.json`` so future calls to
    ``search_past_meeting_notes`` and ``draft_meeting_brief`` can retrieve these
    learnings without mutating the seed fixtures in ``data/``.

    Args:
        meeting_id: Meeting identifier in ``M-NNN`` format (e.g. ``"M-005"``).
        summary: Post-meeting narrative summary (10-2000 chars).
        key_decisions: Non-empty list of 1-20 decisions made (each 3-300 chars).
        follow_up_actions: Optional list of 1-20 follow-up commitments (each
            3-300 chars).
        risks_identified: Optional list of 1-20 risks raised (each 3-300 chars).
        sentiment: Overall stakeholder tone — one of ``"positive"``,
            ``"neutral"``, ``"concerned"``, or ``"escalated"`` (default
            ``"neutral"``).

    Returns:
        On success:
            ``{"status": "ok", "outcome_id": str, "meeting_id": str, "recorded_at_utc": str, "sentiment": str, "decisions_count": int, "follow_ups_count": int, "storage_path": str}``.
        On failure:
            ``{"status": "error", "error_message": str, "recovery_hint": str}``.

    When NOT to use:
        - Do NOT use this tool before a meeting to prepare a pre-read brief;
          call ``draft_meeting_brief`` instead.
        - Do NOT use this tool to search existing meeting notes; call
          ``search_past_meeting_notes`` instead.
        - Do NOT use this tool to distribute a brief to attendees; use
          ``request_brief_approval`` followed by ``send_brief_to_attendees``.
    """
    try:
        validated = SaveMeetingOutcomeInput.model_validate(
            {
                "meeting_id": meeting_id,
                "summary": summary,
                "key_decisions": key_decisions,
                "follow_up_actions": follow_up_actions,
                "risks_identified": risks_identified,
                "sentiment": sentiment,
            }
        )
    except ValidationError as exc:
        return _error_response(
            _format_validation_error(exc),
            "Provide a valid meeting_id ('M-NNN'), a summary of 10-2000 characters, "
            "at least 1 non-empty key_decisions entry (3-300 chars each), and sentiment in "
            "('positive', 'neutral', 'concerned', 'escalated').",
        )

    try:
        meetings = _load_meetings()
    except Exception as exc:
        return _error_response(
            f"Unable to read meetings fixture to validate meeting_id: {type(exc).__name__}: {exc}",
            "Verify that data/meetings.json exists and contains valid JSON, then retry save_meeting_outcome.",
        )

    if not any(m.id == validated.meeting_id for m in meetings):
        return _error_response(
            f"Cannot save outcome: meeting '{validated.meeting_id}' was not found.",
            f"Retry save_meeting_outcome with a valid meeting ID: {_format_meeting_catalog(meetings)}.",
        )

    outcome_id = _new_id("OUT")
    recorded_at = _utc_now_iso()
    follow_ups = validated.follow_up_actions or []
    risks = validated.risks_identified or []

    outcome_record = MeetingOutcomeRecord(
        outcome_id=outcome_id,
        meeting_id=validated.meeting_id,
        recorded_at_utc=recorded_at,
        summary=validated.summary,
        key_decisions=validated.key_decisions,
        follow_up_actions=follow_ups,
        risks_identified=risks,
        sentiment=validated.sentiment,
    )

    try:
        outcomes_raw = _read_optional_runtime_json("outcomes.json", default=[])
        existing_outcomes = [
            MeetingOutcomeRecord.model_validate(item).model_dump()
            for item in outcomes_raw
        ]
        existing_outcomes.append(outcome_record.model_dump())
        storage_path = _write_runtime_json("outcomes.json", existing_outcomes)
    except Exception as exc:
        return _error_response(
            f"Unable to persist meeting outcome: {type(exc).__name__}: {exc}",
            "Check write permissions on runtime/outcomes.json and retry save_meeting_outcome.",
        )

    return SaveMeetingOutcomeOutput(
        status="ok",
        outcome_id=outcome_id,
        meeting_id=validated.meeting_id,
        recorded_at_utc=recorded_at,
        sentiment=validated.sentiment,
        decisions_count=len(validated.key_decisions),
        follow_ups_count=len(follow_ups),
        storage_path=str(storage_path),
    ).model_dump()


ALL_MEETING_PREP_TOOLS = (
    fetch_meeting_details,
    lookup_attendee_profile,
    search_past_meeting_notes,
    get_open_action_items,
    draft_meeting_brief,
    request_brief_approval,
    send_brief_to_attendees,
    save_meeting_outcome,
)
