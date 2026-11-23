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

"""Meeting Prep Concierge — Hybrid Multi-Agent Orchestration (Rubric 2.1, 3.1-3.4).

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
             ├─ attendee_agent       [FAST_MODEL]  -> state["attendee_dossiers"]
             └─ history_agent        [FAST_MODEL]  -> state["account_history"]
           brief_agent               [SMART_MODEL] -> state["brief_draft"]
           critic_agent              [FAST_MODEL]  -> state["critique"]        (self-eval pass)
           approval_gate_agent       [FAST_MODEL]  -> state["approval_decision"] (native HITL stop)
           presenter_agent           [FAST_MODEL]  -> final reply

       Deterministic orchestration guarantees parallel evidence gathering,
       mandatory self-evaluation via ``critic_agent``, an unbypassable native
       ADK human-confirmation checkpoint via ``approval_gate_agent``, and
       predictable ``{key}`` session-state handoffs.

Guardrails & Human-in-the-Loop wiring:
    - All system instructions live in ``app/prompts.py`` (Rubric 2.1).
    - ``PolicyPlugin(BasePlugin)`` in ``app/plugins.py`` is registered on
      ``App`` to short-circuit malformed meeting IDs, oversized briefs, and
      off-invite recipients before tool execution (Rubric 3.3).
    - ``app/callbacks.py`` screens user inputs for prompt injection
      (``before_model_callback``) and validates output shape
      (``after_model_callback``) (Rubric 3.3).
    - ``critic_agent`` performs the runtime self-evaluation guardrail pass
      before approval and presentation (Rubric 3.3).
    - Both ``request_brief_approval`` (inside ``approval_gate_agent``) and
      ``send_brief_to_attendees`` (on ``prep_coordinator``) use ADK's native
      ``FunctionTool(..., require_confirmation=True)`` with
      ``ResumabilityConfig(is_resumable=True)`` on ``App`` (Rubric 3.4).
"""

from __future__ import annotations

from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps import App, ResumabilityConfig
from google.adk.apps.app import EventsCompactionConfig
from google.adk.apps.llm_event_summarizer import LlmEventSummarizer
from google.adk.tools import FunctionTool, load_memory
from google.genai import types

from app.callbacks import (
    enforce_sliding_history_window_before_model,
    schedule_background_memory_ingestion_after_agent,
    screen_prompt_injection_before_model,
    validate_output_shape_after_model,
)
from app.config import (
    COMPACTION_EVENT_RETENTION_SIZE,
    COMPACTION_INTERVAL,
    COMPACTION_OVERLAP_SIZE,
    COMPACTION_TOKEN_THRESHOLD,
    CONTEXT_CACHE_INTERVALS,
    CONTEXT_CACHE_MIN_TOKENS,
    CONTEXT_CACHE_TTL_SECONDS,
    FAST_MODEL,
    SMART_MODEL,
    build_gemini_model,
)
from app.plugins import PolicyPlugin
from app.prompts import (
    APPROVAL_GATE_AGENT_INSTRUCTION,
    ATTENDEE_AGENT_INSTRUCTION,
    BRIEF_AGENT_INSTRUCTION,
    CRITIC_AGENT_INSTRUCTION,
    HISTORY_AGENT_INSTRUCTION,
    PREP_COORDINATOR_INSTRUCTION,
    PRESENTER_AGENT_INSTRUCTION,
)
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
# Human-in-the-Loop Native ADK Tool Confirmation Wrappers (Rubric 3.4)
# ---------------------------------------------------------------------------

# Native HITL Stop #1: pauses execution via ADK's built-in ToolConfirmation
# inside prep_workflow's approval_gate_agent before marking a draft approved.
confirm_brief_approval_tool = FunctionTool(
    func=request_brief_approval,
    require_confirmation=True,
)

# Native HITL Stop #2: pauses execution via ADK's built-in ToolConfirmation
# before any simulated brief delivery can write to runtime/outbox.json.
send_brief_tool = FunctionTool(
    func=send_brief_to_attendees,
    require_confirmation=True,
)

# Shared before_model_callback chain: screen user input for prompt injection
# first, then enforce the complementary HISTORY_WINDOW sliding window on
# ephemeral request history (never touching durable user:-scoped state).
_BEFORE_MODEL_CALLBACKS = [
    screen_prompt_injection_before_model,
    enforce_sliding_history_window_before_model,
]

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
    instruction=ATTENDEE_AGENT_INSTRUCTION,
    tools=[fetch_meeting_details, lookup_attendee_profile],
    output_key="attendee_dossiers",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    before_model_callback=_BEFORE_MODEL_CALLBACKS,
    after_model_callback=validate_output_shape_after_model,
)

history_agent = LlmAgent(
    name="history_agent",
    model=build_gemini_model(FAST_MODEL),
    description=(
        "Gathers relevant past meeting notes, prior decisions, risks, and open "
        "action items linked to the target meeting and its attendees."
    ),
    instruction=HISTORY_AGENT_INSTRUCTION,
    tools=[fetch_meeting_details, search_past_meeting_notes, get_open_action_items],
    output_key="account_history",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    before_model_callback=_BEFORE_MODEL_CALLBACKS,
    after_model_callback=validate_output_shape_after_model,
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
    instruction=BRIEF_AGENT_INSTRUCTION,
    tools=[draft_meeting_brief],
    output_key="brief_draft",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    before_model_callback=_BEFORE_MODEL_CALLBACKS,
    after_model_callback=validate_output_shape_after_model,
)

# ---------------------------------------------------------------------------
# Stage 3: Runtime Self-Evaluation Guardrail Pass (Rubric 3.3, FAST_MODEL)
# ---------------------------------------------------------------------------
# Policy: Runtime Self-Evaluation — critic_agent audits brief_draft against
# attendee_dossiers and account_history for grounding, completeness, attendee
# tailoring, and draft safety before the human approval gate runs.

critic_agent = LlmAgent(
    name="critic_agent",
    model=build_gemini_model(FAST_MODEL),
    description=(
        "Self-evaluation critic auditing the drafted brief against attendee dossiers "
        "and account history for completeness, accuracy, tailoring, and safety."
    ),
    instruction=CRITIC_AGENT_INSTRUCTION,
    output_key="critique",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    before_model_callback=_BEFORE_MODEL_CALLBACKS,
    after_model_callback=validate_output_shape_after_model,
)

# ---------------------------------------------------------------------------
# Stage 5: Final Presentation (FAST_MODEL)
# ---------------------------------------------------------------------------

presenter_agent = LlmAgent(
    name="presenter_agent",
    model=build_gemini_model(FAST_MODEL),
    description=(
        "Applies critic feedback and approval gate outcome to deliver the polished "
        "final response with clear next steps to the user."
    ),
    instruction=PRESENTER_AGENT_INSTRUCTION,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    before_model_callback=_BEFORE_MODEL_CALLBACKS,
    after_model_callback=validate_output_shape_after_model,
    after_agent_callback=schedule_background_memory_ingestion_after_agent,
)

# ---------------------------------------------------------------------------
# Stage 4: Human-in-the-Loop Approval Gate (Rubric 3.4, FAST_MODEL)
# ---------------------------------------------------------------------------
# Inserted between critic_agent and presenter_agent: invokes
# confirm_brief_approval_tool (require_confirmation=True) so ADK pauses for
# explicit human sign-off before handing off to presenter_agent.
#
# ADK 2.15 resumption note: When a confirmation-paused agent inside a
# SequentialAgent sub-agent resumes under a root LlmAgent, ADK resumes the
# paused leaf directly. Attaching presenter_agent as approval_gate_agent's
# guarded downstream sub_agent ensures presenter_agent ALWAYS runs after the
# human approves or rejects, while PolicyPlugin.after_tool_callback records
# state["approval_decision"] authoritatively and Turn 2 returns cleanly to
# prep_coordinator (because presenter_agent disallows all transfers).

approval_gate_agent = LlmAgent(
    name="approval_gate_agent",
    model=build_gemini_model(FAST_MODEL),
    description=(
        "Human-in-the-loop approval gate invoking ADK's native confirmation "
        "mechanism on request_brief_approval before handing off to presenter_agent."
    ),
    instruction=APPROVAL_GATE_AGENT_INSTRUCTION,
    tools=[confirm_brief_approval_tool],
    sub_agents=[presenter_agent],
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
    before_model_callback=_BEFORE_MODEL_CALLBACKS,
    after_model_callback=validate_output_shape_after_model,
)

# ---------------------------------------------------------------------------
# Deterministic Workflow: SequentialAgent
# ---------------------------------------------------------------------------

prep_workflow = SequentialAgent(
    name="prep_workflow",
    description=(
        "Deterministic multi-stage pipeline for preparing a comprehensive meeting brief: "
        "parallel attendee/history gathering -> smart brief synthesis -> quality self-eval "
        "critic -> native HITL approval gate -> final presentation."
    ),
    sub_agents=[
        gather_step,
        brief_agent,
        critic_agent,
        approval_gate_agent,
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
        "meeting/attendee/action-item lookups, cross-session memory recall, "
        "confirmation-gated approval and simulated brief delivery, post-meeting "
        "outcome recording, and delegates full meeting-brief preparation to the "
        "deterministic prep_workflow."
    ),
    instruction=PREP_COORDINATOR_INSTRUCTION,
    tools=[
        fetch_meeting_details,
        lookup_attendee_profile,
        search_past_meeting_notes,
        get_open_action_items,
        confirm_brief_approval_tool,
        send_brief_tool,
        save_meeting_outcome,
        load_memory,
    ],
    sub_agents=[prep_workflow],
    before_model_callback=_BEFORE_MODEL_CALLBACKS,
    after_model_callback=validate_output_shape_after_model,
    after_agent_callback=schedule_background_memory_ingestion_after_agent,
)

# ---------------------------------------------------------------------------
# ADK App Wiring: Guardrails, Resumability, Compaction & Context Caching
# ---------------------------------------------------------------------------
# History Compaction & Caching (Rubric 2.2):
#   - EventsCompactionConfig runs post-invocation using FAST_MODEL (explicit
#     LlmEventSummarizer so ADK does not default to SMART_MODEL). Compaction
#     triggers when observed prompt tokens >= COMPACTION_TOKEN_THRESHOLD (12,000,
#     keeping the newest 6 raw events) OR every COMPACTION_INTERVAL (6) user
#     invocations with a 2-invocation overlap. Compaction summarizes only
#     conversational events — durable ``user:``-scoped state is never touched.
#   - ContextCacheConfig gives each transferred agent its own explicit prompt
#     cache once the Gemini 3 floor (4,096 tokens) is reached, eliminating
#     ADK 2.15's uncached multi-agent transfer warning.
app = App(
    root_agent=root_agent,
    name="app",
    plugins=[PolicyPlugin()],
    resumability_config=ResumabilityConfig(is_resumable=True),
    events_compaction_config=EventsCompactionConfig(
        summarizer=LlmEventSummarizer(llm=build_gemini_model(FAST_MODEL)),
        token_threshold=COMPACTION_TOKEN_THRESHOLD,
        event_retention_size=COMPACTION_EVENT_RETENTION_SIZE,
        compaction_interval=COMPACTION_INTERVAL,
        overlap_size=COMPACTION_OVERLAP_SIZE,
    ),
    context_cache_config=ContextCacheConfig(
        min_tokens=CONTEXT_CACHE_MIN_TOKENS,
        ttl_seconds=CONTEXT_CACHE_TTL_SECONDS,
        cache_intervals=CONTEXT_CACHE_INTERVALS,
        create_http_options=types.HttpOptions(timeout=10_000),
    ),
)
