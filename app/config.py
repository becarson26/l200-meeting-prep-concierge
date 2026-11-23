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

"""Strategic model routing configuration for the Meeting Prep Concierge (Rubric 3.2).

All model-tier selections live in this single module so the routing decision,
cost/latency trade-offs, and task-complexity boundaries are auditable in one place.
Both model constants can be overridden via environment variables without code changes.
"""

from __future__ import annotations

import os
from typing import Final

from google.adk.models import Gemini
from google.genai import types

# ---------------------------------------------------------------------------
# Fast / Flash Tier — Low Cost, Low Latency, Bounded Tool & Formatting Work
# ---------------------------------------------------------------------------
# Cost:        Lowest per-token input/output pricing in the Gemini lineup,
#              making parallel fan-out calls and iterative checks economical.
# Latency:     Sub-second time-to-first-token, critical for keeping the
#              ParallelAgent gather stage, checklist critic, and final
#              presentation responsive within an interactive session.
# Complexity:  Ideal for narrow, well-scoped tasks with deterministic schemas
#              or clear rubrics: fetching attendee dossiers, searching meeting
#              notes/action items, running a structured quality checklist, and
#              formatting an already-synthesized brief for display.
FAST_MODEL: Final[str] = os.getenv("FAST_MODEL", "gemini-3.8-flash")

# ---------------------------------------------------------------------------
# Smart / Pro Tier — Higher Reasoning Depth, Justified Selectively
# ---------------------------------------------------------------------------
# Cost:        Higher per-token cost than Flash; restricted strictly to the
#              two agents where deeper reasoning materially improves quality.
# Latency:     Longer thinking/generation latency than Flash, acceptable only
#              when coordinating multi-turn intent or synthesizing cross-source
#              executive narratives once per meeting-prep run.
# Complexity:  Required for open-ended judgment: (1) prep_coordinator deciding
#              whether to answer directly, call tools, enforce human approval,
#              or delegate to prep_workflow; and (2) brief_agent synthesizing
#              conflicting attendee preferences, overdue commitments, and prior
#              decisions into a tailored executive brief.
SMART_MODEL: Final[str] = os.getenv("SMART_MODEL", "gemini-pro-latest")

# Explicit, self-documenting routing matrix readable at a glance.
MODEL_ROUTING_TABLE: Final[dict[str, dict[str, str]]] = {
    "prep_coordinator": {
        "tier": "SMART_MODEL",
        "model": SMART_MODEL,
        "task_complexity": "high",
        "rationale": (
            "Top-level conversational router deciding between direct tool calls, "
            "approval/sending safeguards, outcome capture, and delegating to prep_workflow."
        ),
    },
    "attendee_agent": {
        "tier": "FAST_MODEL",
        "model": FAST_MODEL,
        "task_complexity": "low",
        "rationale": (
            "Parallel gather branch executing bounded lookup tools "
            "(fetch_meeting_details, lookup_attendee_profile) into state['attendee_dossiers']."
        ),
    },
    "history_agent": {
        "tier": "FAST_MODEL",
        "model": FAST_MODEL,
        "task_complexity": "low",
        "rationale": (
            "Parallel gather branch executing bounded search/filter tools "
            "(search_past_meeting_notes, get_open_action_items) into state['account_history']."
        ),
    },
    "brief_agent": {
        "tier": "SMART_MODEL",
        "model": SMART_MODEL,
        "task_complexity": "high",
        "rationale": (
            "Cross-source synthesis combining attendee communication styles, "
            "past decisions, risks, and open commitments into a tailored brief draft."
        ),
    },
    "critic_agent": {
        "tier": "FAST_MODEL",
        "model": FAST_MODEL,
        "task_complexity": "medium",
        "rationale": (
            "Bounded rubric/checklist audit verifying brief_draft against "
            "attendee_dossiers and account_history with low latency."
        ),
    },
    "approval_gate_agent": {
        "tier": "FAST_MODEL",
        "model": FAST_MODEL,
        "task_complexity": "low",
        "rationale": (
            "Deterministic HITL checkpoint extracting draft_id and invoking "
            "ADK's native confirmation-gated request_brief_approval tool."
        ),
    },
    "presenter_agent": {
        "tier": "FAST_MODEL",
        "model": FAST_MODEL,
        "task_complexity": "low",
        "rationale": (
            "Deterministic formatting pass applying critic adjustments to "
            "produce a clean, concise user-facing response."
        ),
    },
}


def build_gemini_model(model_name: str, *, retry_attempts: int = 3) -> Gemini:
    """Wrap a routed Gemini model ID with consistent HTTP retry options."""
    return Gemini(
        model=model_name,
        retry_options=types.HttpRetryOptions(attempts=retry_attempts),
    )
