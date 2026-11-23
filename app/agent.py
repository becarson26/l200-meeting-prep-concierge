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

"""Meeting Prep Concierge — Hybrid Multi-Agent Orchestration (Rubric 3.1 & 3.2).

Orchestration architecture rationale:
    This system deliberately combines **two orchestration styles** in one ADK
    application so each problem class gets the right level of control:

    1. **LLM-driven delegation (`root_agent` / `prep_coordinator`, `LlmAgent`)**:
       User turns arrive with unpredictable intent — a quick attendee profile
       lookup, an open action-item check, a post-meeting outcome note, an
       approval/send follow-up, or a full meeting-brief request. An LLM router
       on ``SMART_MODEL`` is used here because natural-language intent
       classification, clarifying questions, and deciding *when* to invoke a
       single tool vs. hand off to the full workflow require dynamic judgment.

    2. **Deterministic workflow (`prep_workflow`, `SequentialAgent` + `ParallelAgent`)**:
       Once the coordinator decides a full meeting brief is needed, execution
       switches to a fixed, auditable pipeline:

           gather_step (ParallelAgent)
             ├─ attendee_agent  [FAST_MODEL]  -> state["attendee_dossiers"]
             └─ history_agent   [FAST_MODEL]  -> state["account_history"]
           brief_agent          [SMART_MODEL] -> state["brief_draft"]
           critic_agent         [FAST_MODEL]  -> state["critique"]
           # [Part 4 insertion point: human approval gate]
           presenter_agent      [FAST_MODEL]  -> final reply

       Deterministic orchestration is used here because brief preparation must
       never skip evidence gathering, benefits from concurrent I/O fan-out in
       ``gather_step``, requires mandatory self-critique before presentation,
       passes structured state deterministically via ``{key}`` templates, and
       provides a guaranteed insertion point for the Part 4 approval gate.
"""

from __future__ import annotations

from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.apps import App

from app.config import FAST_MODEL, SMART_MODEL, build_gemini_model
from app.tools import (
    draft_meeting_brief,
    fetch_meeting_details,
    get_open_action_items,
    lookup_attendee_profile,
    request_brief_approval,
    save_meeting_outcome,
    search_past_meeting_notes,
    send_brief_to_attendees,
)

# ---------------------------------------------------------------------------
# Stage 1 (Parallel Gather): Attendee Dossiers + Account/Meeting History
# ---------------------------------------------------------------------------

attendee_agent = LlmAgent(
    name="attendee_agent",
    model=build_gemini_model(FAST_MODEL),
    description=(
        "Fetches the target meeting's invitees and compiles structured attendee "
        "dossiers with roles, departments, and communication preferences."
    ),
    instruction=(
        "You are the Attendee Research Specialist for Northwind Telecom's Meeting Prep Concierge.\n"
        "Your ONLY job is to gather complete attendee dossiers for the meeting requested in the conversation.\n\n"
        "Steps:\n"
        "1. Identify the target meeting_id (e.g., M-005) from the conversation and call "
        "fetch_meeting_details(meeting_id=...).\n"
        "2. For every attendee email listed on that meeting, call lookup_attendee_profile(email=...).\n"
        "3. If any tool returns status='error', follow its recovery_hint cleanly without guessing.\n"
        "4. Output a concise, structured dossier listing the meeting ID, title, date/time, agenda, "
        "and each attendee's name, email, role, department, communication style, preferred channel, "
        "pre-read lead time, and dislikes.\n"
        "Do NOT draft the final meeting brief and do NOT address the user directly."
    ),
    tools=[fetch_meeting_details, lookup_attendee_profile],
    output_key="attendee_dossiers",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

history_agent = LlmAgent(
    name="history_agent",
    model=build_gemini_model(FAST_MODEL),
    description=(
        "Gathers relevant past meeting notes, prior decisions, risks, and open "
        "action items linked to the target meeting and its attendees."
    ),
    instruction=(
        "You are the Account & Meeting History Specialist for Northwind Telecom's Meeting Prep Concierge.\n"
        "Your ONLY job is to collect historical context and unresolved commitments for the requested meeting.\n\n"
        "Steps:\n"
        "1. Identify the target meeting_id from the conversation and call fetch_meeting_details(meeting_id=...).\n"
        "2. Search past meeting notes using search_past_meeting_notes with topic keywords drawn from "
        "the meeting's title and agenda (run 1-3 focused searches if the meeting spans multiple workstreams).\n"
        "3. Call get_open_action_items() (or filter by invited attendee emails) to identify every open "
        "commitment relevant to the meeting's topics or invited attendees, flagging any past-due dates.\n"
        "4. If any tool returns status='error', follow its recovery_hint rather than inventing facts.\n"
        "5. Output a concise structured summary of: (a) related past meetings and key decisions, "
        "(b) open questions and risks carried forward, and (c) relevant open action items with ID, owner, "
        "priority, and due date.\n"
        "Do NOT draft the final meeting brief and do NOT address the user directly."
    ),
    tools=[fetch_meeting_details, search_past_meeting_notes, get_open_action_items],
    output_key="account_history",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

gather_step = ParallelAgent(
    name="gather_step",
    description=(
        "Concurrently gathers attendee dossiers and historical notes/action items "
        "into session state before brief synthesis."
    ),
    sub_agents=[attendee_agent, history_agent],
)

# ---------------------------------------------------------------------------
# Stage 2: Synthesis & Staged Draft Creation (SMART_MODEL)
# ---------------------------------------------------------------------------

brief_agent = LlmAgent(
    name="brief_agent",
    model=build_gemini_model(SMART_MODEL),
    description=(
        "Synthesizes attendee dossiers and account history into a tailored meeting "
        "prep brief and stages a non-sent draft via draft_meeting_brief."
    ),
    instruction=(
        "You are the Senior Brief Strategist for Northwind Telecom's Meeting Prep Concierge.\n"
        "Synthesize the gathered research below into an executive-ready pre-meeting brief.\n\n"
        "=== ATTENDEE DOSSIERS ===\n"
        "{attendee_dossiers}\n\n"
        "=== ACCOUNT HISTORY & OPEN COMMITMENTS ===\n"
        "{account_history}\n\n"
        "Steps:\n"
        "1. Choose the best audience_focus ('executive', 'technical', 'commercial', or 'balanced') "
        "based on the user's request and the invited stakeholders' communication preferences.\n"
        "2. Formulate 3-6 sharp key_talking_points that connect the agenda, overdue/high-priority "
        "action items, and carried-forward risks.\n"
        "3. Call draft_meeting_brief(meeting_id=..., audience_focus=..., key_talking_points=..., "
        "include_open_action_items=True) to stage the draft. This NEVER sends anything.\n"
        "4. Output the complete drafted brief clearly including:\n"
        "   - The staged draft_id returned by draft_meeting_brief (mandatory)\n"
        "   - Meeting metadata and intended recipients\n"
        "   - Executive summary & decision asks\n"
        "   - Tailored framing notes aligned to attendee communication preferences\n"
        "   - Relevant open/overdue action items with owners and due dates\n"
        "   - Related prior decisions and open risks\n"
        "Ground every fact strictly in the gathered tool data above. Never claim the brief was sent."
    ),
    tools=[draft_meeting_brief],
    output_key="brief_draft",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

# ---------------------------------------------------------------------------
# Stage 3: Automated Quality & Policy Self-Critique (FAST_MODEL)
# ---------------------------------------------------------------------------

critic_agent = LlmAgent(
    name="critic_agent",
    model=build_gemini_model(FAST_MODEL),
    description=(
        "Audits the drafted brief against attendee dossiers and account history "
        "for completeness, accuracy, tailoring, and safety constraints."
    ),
    instruction=(
        "You are the Quality & Policy Critic for Northwind Telecom's Meeting Prep Concierge.\n"
        "Audit the drafted brief below against the source research before it reaches the user.\n\n"
        "=== DRAFTED BRIEF ===\n"
        "{brief_draft}\n\n"
        "=== SOURCE ATTENDEE DOSSIERS ===\n"
        "{attendee_dossiers}\n\n"
        "=== SOURCE ACCOUNT HISTORY ===\n"
        "{account_history}\n\n"
        "Evaluate against this checklist:\n"
        "1. Grounding: Every meeting ID, date, attendee email, action item ID, and prior decision "
        "matches the source data with zero hallucinated facts.\n"
        "2. Completeness: High-priority or overdue open action items and key carried-forward risks "
        "for invited attendees are highlighted.\n"
        "3. Attendee Tailoring: Key communication preferences (e.g., lead-time expectations, "
        "dislikes, decision framing) are reflected.\n"
        "4. Draft Safety: A valid staged draft_id (DRF-...) is present and the brief clearly states "
        "it is a DRAFT that has NOT been sent.\n\n"
        "Output a concise structured critique with:\n"
        "- Verdict: PASS or REVISE_IN_PRESENTATION\n"
        "- Strengths (1-3 bullets)\n"
        "- Required Corrections or Additions (specific bullet list, or 'None')\n"
        "Do NOT address the user directly."
    ),
    output_key="critique",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

# ---------------------------------------------------------------------------
# Stage 4: Final Presentation (FAST_MODEL)
# ---------------------------------------------------------------------------

presenter_agent = LlmAgent(
    name="presenter_agent",
    model=build_gemini_model(FAST_MODEL),
    description=(
        "Applies critic feedback to the drafted brief and delivers the polished "
        "final response with clear next steps to the user."
    ),
    instruction=(
        "You are the Presentation Editor for Northwind Telecom's Meeting Prep Concierge.\n"
        "Produce the final user-facing reply by combining the drafted brief with the critic's audit.\n\n"
        "=== DRAFTED BRIEF ===\n"
        "{brief_draft}\n\n"
        "=== CRITIC REVIEW ===\n"
        "{critique}\n\n"
        "Rules:\n"
        "1. Incorporate every required correction or omission flagged in the critic review so the "
        "final brief is accurate and complete.\n"
        "2. Present the brief in crisp, scannable Markdown with clear sections (Executive Summary, "
        "Agenda & Talking Points, Open Action Items, Prior Decisions & Risks, Attendee Tailoring Notes).\n"
        "3. Prominently display the staged draft_id (DRF-...) and explicitly state that this brief "
        "is a **draft** and has **not** been sent to attendees.\n"
        "4. End with concrete next steps asking whether the user would like to adjust the brief "
        "or request approval to send it to attendees.\n"
        "Do NOT expose raw internal critique boilerplate unless noting a genuine data caveat."
    ),
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

# ---------------------------------------------------------------------------
# Deterministic Workflow: SequentialAgent
# ---------------------------------------------------------------------------

prep_workflow = SequentialAgent(
    name="prep_workflow",
    description=(
        "Deterministic multi-stage pipeline for preparing a comprehensive meeting brief: "
        "parallel attendee/history gathering -> smart brief synthesis -> quality critique "
        "-> final presentation."
    ),
    sub_agents=[
        gather_step,
        brief_agent,
        critic_agent,
        # -------------------------------------------------------------------
        # INSERTION POINT (Part 4 — Human-in-the-Loop Approval Gate):
        # Insert the approval_gate_agent / confirmation step here between
        # critic_agent and presenter_agent so every brief can optionally pass
        # through an explicit code-enforced approval checkpoint before delivery.
        # -------------------------------------------------------------------
        presenter_agent,
    ],
)

# ---------------------------------------------------------------------------
# LLM-Driven Delegation: Root Coordinator (SMART_MODEL)
# ---------------------------------------------------------------------------

root_agent = LlmAgent(
    # Keep in sync with create_params.root_agent_name in agents-cli-manifest.yaml:
    # telemetry reports this name as gen_ai.agent.name, and agents-cli reports the
    # manifest's value. Renaming the agent only here makes the two disagree, and
    # anything selecting traces by name stops finding this agent's.
    name="prep_coordinator",
    model=build_gemini_model(SMART_MODEL),
    description=(
        "Northwind Telecom's Meeting Prep Concierge coordinator. Handles direct "
        "meeting/attendee/action-item lookups, approval and simulated brief delivery, "
        "post-meeting outcome recording, and delegates full meeting-brief preparation "
        "to the deterministic prep_workflow."
    ),
    instruction=(
        "You are the Meeting Prep Concierge Coordinator for Northwind Telecom.\n"
        "You route requests cleanly between direct tool execution and the deterministic "
        "prep_workflow pipeline.\n\n"
        "Routing policy:\n"
        "1. Full meeting prep / brief creation (e.g., 'prepare a brief for M-005', "
        "'help me prep for the fiber rollout checkpoint'):\n"
        "   - If the user did not specify a meeting_id, clarify or check meeting details first.\n"
        "   - Once the target meeting is clear, delegate immediately to 'prep_workflow'. "
        "Do NOT try to manually write a full multi-section brief yourself.\n"
        "2. Direct lookups & conversational queries:\n"
        "   - Attendee bio or communication preferences -> call lookup_attendee_profile directly.\n"
        "   - Single meeting schedule, attendees, or agenda -> call fetch_meeting_details directly.\n"
        "   - Past meeting notes or decisions -> call search_past_meeting_notes directly.\n"
        "   - Unresolved tasks by owner/meeting/priority -> call get_open_action_items directly.\n"
        "3. Brief approval & simulated delivery (strict two-step safeguard):\n"
        "   - Never call send_brief_to_attendees without first calling request_brief_approval(draft_id=...) "
        "and receiving explicit human confirmation in the conversation.\n"
        "   - Only after the user explicitly confirms sending may you call "
        "send_brief_to_attendees(draft_id=..., approval_confirmed=True).\n"
        "4. Post-meeting learnings:\n"
        "   - When the user shares decisions/outcomes after a meeting, call save_meeting_outcome.\n"
        "5. Error recovery:\n"
        "   - Every tool returns status='ok' or status='error' with a recovery_hint. "
        "Always follow the recovery_hint rather than guessing or inventing Northwind Telecom facts."
    ),
    tools=[
        fetch_meeting_details,
        lookup_attendee_profile,
        search_past_meeting_notes,
        get_open_action_items,
        request_brief_approval,
        send_brief_to_attendees,
        save_meeting_outcome,
    ],
    sub_agents=[prep_workflow],
)

# Existing App(...) wiring preserved intact — plugins will be attached here in Part 4.
app = App(
    root_agent=root_agent,
    name="app",
)
