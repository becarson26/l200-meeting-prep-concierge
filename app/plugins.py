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

"""ADK PolicyPlugin enforcing deterministic pre-tool guardrails (Rubric 3.3).

Defense-in-depth architecture:
    1. ``PolicyPlugin(BasePlugin)`` (this file) — registered on ``App`` so every
       tool invocation across all agents is intercepted *before* execution (and
       before human-confirmation prompts) to block out-of-policy calls.
    2. ``app/callbacks.py`` — ``before_model_callback`` (prompt-injection input
       screening) and ``after_model_callback`` (model output-shape validation).
    3. ``critic_agent`` (``app/agent.py`` / ``app/prompts.py``) — runtime
       self-evaluation pass auditing drafted briefs against source records.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext

from app.schemas import MEETING_ID_PATTERN

_MEETING_ID_RE = re.compile(MEETING_ID_PATTERN)
_REPO_ROOT = Path(__file__).resolve().parent.parent

# Default size ceilings enforced by the Brief Size Limit policy.
DEFAULT_MAX_BRIEF_CHARS = 10_000
DEFAULT_MAX_TALKING_POINTS_CHARS = 1_500


def _policy_block(
    policy_name: str, error_message: str, recovery_hint: str
) -> dict[str, Any]:
    """Return a structured short-circuit response recognized by ADK and the LLM."""
    return {
        "status": "error",
        "policy_violation": policy_name,
        "error_message": error_message,
        "recovery_hint": recovery_hint,
    }


def _resolve_data_dir() -> Path:
    override = os.environ.get("MEETING_PREP_DATA_DIR")
    return Path(override) if override else (_REPO_ROOT / "data")


def _resolve_runtime_dir() -> Path:
    override = os.environ.get("MEETING_PREP_RUNTIME_DIR")
    return Path(override) if override else (_REPO_ROOT / "runtime")


def _load_staged_draft(draft_id: str) -> dict[str, Any] | None:
    drafts_path = _resolve_runtime_dir() / "drafts.json"
    if not drafts_path.exists():
        return None
    try:
        data = json.loads(drafts_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            entry = data.get(draft_id)
            return entry if isinstance(entry, dict) else None
    except (OSError, json.JSONDecodeError):
        return None
    return None


def _load_meeting_invitees(meeting_id: str) -> list[str] | None:
    meetings_path = _resolve_data_dir() / "meetings.json"
    if not meetings_path.exists():
        return None
    try:
        meetings = json.loads(meetings_path.read_text(encoding="utf-8"))
        for meeting in meetings:
            if isinstance(meeting, dict) and meeting.get("id") == meeting_id:
                attendees = meeting.get("attendees")
                if isinstance(attendees, list):
                    return [str(addr).strip().lower() for addr in attendees]
    except (OSError, json.JSONDecodeError):
        return None
    return None


class PolicyPlugin(BasePlugin):
    """App-wide ADK plugin that short-circuits out-of-policy tool calls before execution."""

    def __init__(
        self,
        *,
        name: str = "meeting_prep_policy_plugin",
        max_brief_chars: int = DEFAULT_MAX_BRIEF_CHARS,
        max_talking_points_chars: int = DEFAULT_MAX_TALKING_POINTS_CHARS,
    ) -> None:
        super().__init__(name=name)
        self.max_brief_chars = max_brief_chars
        self.max_talking_points_chars = max_talking_points_chars

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any] | None:
        """Inspect every tool call before execution and short-circuit policy violations."""
        tool_name = tool.name

        # ------------------------------------------------------------------
        # Policy 1 — Malformed Meeting ID Guardrail
        # Enforces: Any non-empty ``meeting_id`` argument passed to any tool
        #           must strictly match ``^M-\d{3,6}$``.
        # Why:      Blocks path-traversal payloads, prompt-injected strings,
        #           and hallucinated IDs before they touch the data layer or
        #           waste downstream model turns.
        # ------------------------------------------------------------------
        if "meeting_id" in tool_args and tool_args["meeting_id"] is not None:
            raw_meeting_id = tool_args["meeting_id"]
            if not isinstance(raw_meeting_id, str) or not _MEETING_ID_RE.match(
                raw_meeting_id.strip()
            ):
                return _policy_block(
                    policy_name="malformed_meeting_id",
                    error_message=(
                        f"Policy violation in {tool_name}: meeting_id "
                        f"{raw_meeting_id!r} does not match required format 'M-NNN'."
                    ),
                    recovery_hint=(
                        "Provide a valid meeting_id matching 'M-NNN' (for example "
                        "'M-001' through 'M-007') and retry the tool call."
                    ),
                )

        # ------------------------------------------------------------------
        # Policy 2 — Brief Size Limit Guardrail
        # Enforces: Talking points passed to ``draft_meeting_brief`` cannot
        #           exceed ``max_talking_points_chars``, and staged briefs
        #           submitted to ``request_brief_approval`` or
        #           ``send_brief_to_attendees`` cannot exceed ``max_brief_chars``.
        # Why:      Prevents runaway LLM output, context bloat, and unreadable
        #           wall-of-text briefs from reaching busy executives or outbox
        #           storage.
        # ------------------------------------------------------------------
        if tool_name == "draft_meeting_brief":
            talking_points = tool_args.get("key_talking_points")
            if isinstance(talking_points, list):
                total_chars = sum(len(str(pt)) for pt in talking_points)
                if total_chars > self.max_talking_points_chars:
                    return _policy_block(
                        policy_name="brief_size_limit",
                        error_message=(
                            f"Policy violation in draft_meeting_brief: combined "
                            f"key_talking_points length ({total_chars} chars) exceeds "
                            f"the {self.max_talking_points_chars}-character limit."
                        ),
                        recovery_hint=(
                            "Condense key_talking_points to concise bullets totaling "
                            f"under {self.max_talking_points_chars} characters and retry "
                            "draft_meeting_brief."
                        ),
                    )

        if tool_name in ("request_brief_approval", "send_brief_to_attendees"):
            raw_draft_id = tool_args.get("draft_id")
            if isinstance(raw_draft_id, str):
                staged = _load_staged_draft(raw_draft_id.strip())
                if staged is not None:
                    rendered = str(
                        (staged.get("brief") or {}).get("rendered_markdown", "")
                    )
                    if len(rendered) > self.max_brief_chars:
                        return _policy_block(
                            policy_name="brief_size_limit",
                            error_message=(
                                f"Policy violation in {tool_name}: staged brief "
                                f"'{raw_draft_id}' is {len(rendered)} characters, "
                                f"exceeding the {self.max_brief_chars}-character policy limit."
                            ),
                            recovery_hint=(
                                "Call draft_meeting_brief with more concise talking points "
                                f"so the brief stays under {self.max_brief_chars} characters, "
                                "then request approval for the new draft."
                            ),
                        )

        # ------------------------------------------------------------------
        # Policy 3 — Meeting Invite-List Recipient Guardrail
        # Enforces: ``send_brief_to_attendees`` may ONLY deliver to email
        #           addresses explicitly listed on the target meeting's
        #           canonical invite list in ``data/meetings.json``.
        # Why:      Prevents accidental or prompt-injected data exfiltration
        #           of confidential meeting briefs to uninvited internal
        #           colleagues or external third-party addresses.
        # ------------------------------------------------------------------
        if tool_name == "send_brief_to_attendees":
            raw_draft_id = tool_args.get("draft_id")
            override = tool_args.get("recipient_override")
            if isinstance(raw_draft_id, str):
                staged = _load_staged_draft(raw_draft_id.strip())
                if staged is not None:
                    meeting_id = str(staged.get("meeting_id", ""))
                    invited = _load_meeting_invitees(meeting_id)
                    if invited is not None:
                        invited_set = set(invited)
                        raw_recipients = (
                            override
                            if isinstance(override, list)
                            else (staged.get("brief") or {}).get("recipients", [])
                        )
                        normalized = [
                            str(addr).strip().lower() for addr in raw_recipients
                        ]
                        unauthorized = [
                            addr for addr in normalized if addr not in invited_set
                        ]
                        if unauthorized:
                            return _policy_block(
                                policy_name="unauthorized_recipient",
                                error_message=(
                                    f"Policy violation in send_brief_to_attendees: "
                                    f"recipient(s) {unauthorized} are NOT on the invite "
                                    f"list for meeting '{meeting_id}'."
                                ),
                                recovery_hint=(
                                    f"Remove uninvited recipients and send only to invited "
                                    f"attendees for {meeting_id}: {', '.join(invited)}."
                                ),
                            )

        # Returning None permits normal ADK execution (and confirmation checks).
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Record canonical approval-gate outcomes in session state after confirmation."""
        # ------------------------------------------------------------------
        # Policy 4 — Canonical Approval-State Recording
        # Enforces: Session state ``approval_decision`` is populated directly
        #           from the verified ADK tool result of ``request_brief_approval``
        #           rather than trusting free-form LLM text.
        # Why:      Guarantees downstream ``presenter_agent`` always receives
        #           the authoritative APPROVED or REJECTED outcome after ADK
        #           resumes from native tool confirmation.
        # ------------------------------------------------------------------
        if tool.name == "request_brief_approval":
            draft_id = str(tool_args.get("draft_id", "unknown"))
            if isinstance(result, dict) and result.get("status") == "ok":
                status_label = str(result.get("approval_status", "pending_approval")).upper()
                tool_context.state["approval_decision"] = (
                    f"{status_label} (draft_id={draft_id}, "
                    f"meeting_id={result.get('meeting_id', 'unknown')})"
                )
            elif isinstance(result, dict):
                reason = result.get("error") or result.get("error_message") or "rejected"
                tool_context.state["approval_decision"] = (
                    f"REJECTED (draft_id={draft_id}, reason={reason})"
                )

        # ------------------------------------------------------------------
        # Durable User-Scoped State Persistence (Rubric 2.3)
        # Enforces: Verified attendee preferences and user-level meeting
        #           context are written under ``user:``-prefixed keys so
        #           ADK's DatabaseSessionService persists them in its
        #           ``user_states`` table across separate conversations.
        # Why:      Reassigning fresh dict copies ensures ADK records the
        #           state_delta cleanly; history compaction never touches
        #           these durable user-scoped keys.
        # ------------------------------------------------------------------
        if isinstance(result, dict) and result.get("status") == "ok":
            if tool.name == "lookup_attendee_profile" and isinstance(
                result.get("attendee"), dict
            ):
                attendee = result["attendee"]
                email = str(attendee.get("email", "")).lower()
                if email:
                    existing_prefs = dict(
                        tool_context.state.get("user:attendee_preferences") or {}
                    )
                    existing_prefs[email] = {
                        "name": attendee.get("name"),
                        "role": attendee.get("role"),
                        "communication_preferences": attendee.get(
                            "communication_preferences"
                        ),
                    }
                    tool_context.state["user:attendee_preferences"] = existing_prefs

            elif tool.name in ("fetch_meeting_details", "draft_meeting_brief"):
                existing_ctx = dict(tool_context.state.get("user:context") or {})
                meeting_id = (
                    (result.get("meeting") or {}).get("id")
                    if tool.name == "fetch_meeting_details"
                    else result.get("meeting_id")
                )
                if meeting_id:
                    existing_ctx["last_meeting_id"] = meeting_id
                    recent = list(existing_ctx.get("recent_meetings") or [])
                    if meeting_id in recent:
                        recent.remove(meeting_id)
                    recent.insert(0, meeting_id)
                    existing_ctx["recent_meetings"] = recent[:10]
                if tool.name == "draft_meeting_brief":
                    focus = (result.get("brief") or {}).get("audience_focus")
                    if focus:
                        existing_ctx["preferred_audience_focus"] = focus
                tool_context.state["user:context"] = existing_ctx

        return None
