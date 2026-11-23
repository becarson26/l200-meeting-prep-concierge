# Meeting Prep Concierge

An enterprise-grade multi-agent meeting preparation assistant built with **Google ADK (`google-adk` 2.15)** and **Gemini**, designed around hybrid orchestration, native Human-in-the-Loop (HITL) confirmations, deterministic policy guardrails, PII-redacted observability, persistent session/memory backends, automated evaluations, and Terraform IaC.

> **Repository shipping posture:** This repository ships configured for **Path A — Local** (SQLite sessions, in-memory long-term memory, console OpenTelemetry trace exporter, Gemini API key via Google AI Studio in `.env` — zero GCP project and zero cloud cost). **Path B — Google Cloud** is fully wired in code and Terraform, documented end-to-end below, and **not applied**; the placeholder project IDs (`your-gcp-project-id`, `your-production-project-id`, `your-staging-project-id`, `your-cicd-project-id`) in `deployment/terraform/*/vars/env.tfvars` are intentional for that reason.

## Contents

- [Problem & Solution](#problem--solution)
- [Architecture](#architecture)
- [Path A — Local Deployment (Default)](#path-a--local-deployment-default)
- [Path B — Google Cloud Deployment & Teardown](#path-b--google-cloud-deployment--teardown)
- [Backend Comparison Table (Local vs. Cloud)](#backend-comparison-table-local-vs-cloud)
- [Live Run Evidence](#live-run-evidence)
- [Notes (Synthetic Data, Calendar Scope & `uv.lock`)](#notes-synthetic-data-calendar-scope--uvlock)
- [Engineering Notes (ADK 2.15 Workarounds)](#engineering-notes-adk-215-workarounds)
- [Rubric Coverage Matrix (19 / 19 Criteria)](#rubric-coverage-matrix-19--19-criteria)

---

## Problem & Solution

Preparing for customer and stakeholder meetings is manual and repetitive. Context is scattered across past notes, prior action items, and whatever you happen to remember about each attendee. This Meeting Prep Concierge assembles a briefing automatically and learns what matters to me over time.

Instead of a single monolithic prompt, the concierge pairs a conversational root coordinator (`prep_coordinator`) for focused lookups and post-meeting capture with a deterministic multi-stage briefing workflow (`prep_workflow`) that gathers attendee dossiers and meeting history in parallel, synthesizes a tailored draft on a reasoning model, audits its own output through a dedicated critic pass, pauses on a native ADK human-confirmation gate before finalizing, and never sends a brief to attendees without explicit human confirmation.

---

## Architecture

Every model call passes through prompt-injection input screening, sliding-window history bounding, and output-shape validation (`app/callbacks.py`). Every tool call across all agents passes through `App`-level plugins in strict order: `ObservabilityPlugin` (intent/outcome JSON logs + custom OTel spans) followed by `PolicyPlugin` (malformed meeting IDs, brief size limits, and invite-list enforcement).

```mermaid
flowchart TD
    User["User Request"] --> App["ADK App (Resumable, EventsCompactionConfig, ContextCacheConfig) - Plugins: ObservabilityPlugin -> PolicyPlugin"]
    App --> Coord["prep_coordinator (LlmAgent) - SMART_MODEL: gemini-pro-latest"]

    Coord -->|"Direct lookup / outcome capture"| DirectTools["Direct Tools: fetch_meeting_details, lookup_attendee_profile, search_past_meeting_notes, get_open_action_items, save_meeting_outcome, load_memory"]
    Coord -->|"Explicit send request"| SendGate["send_brief_to_attendees (Native HITL Confirmation Gate)"]
    Coord -->|"Full brief prep (transfer_to_agent)"| Gather

    subgraph PrepWorkflow["prep_workflow (Deterministic SequentialAgent)"]
        direction TD
        Gather["gather_step (ParallelAgent)"]
        Gather --> Att["attendee_agent (LlmAgent) - FAST_MODEL: gemini-3.8-flash -> state.attendee_dossiers"]
        Gather --> Hist["history_agent (LlmAgent) - FAST_MODEL: gemini-3.8-flash -> state.account_history"]
        Att --> Brief["brief_agent (LlmAgent) - SMART_MODEL: gemini-pro-latest -> state.brief_draft"]
        Hist --> Brief
        Brief --> Critic["critic_agent (LlmAgent, Self-Eval) - FAST_MODEL: gemini-3.8-flash -> state.critique"]
        Critic --> Gate["approval_gate_agent (LlmAgent, Native HITL Stop) - FAST_MODEL: gemini-3.8-flash -> state.approval_decision"]
        Gate -->|"Human Approve / Reject Resume"| Pres["presenter_agent (LlmAgent) - FAST_MODEL: gemini-3.8-flash -> Final Reply + Async Memory"]
    end
```

### Model Routing Summary (`app/config.py`)

| Agent / Component | Tier | Default Model | Rationale |
|---|---|---|---|
| `prep_coordinator` | `SMART_MODEL` | `gemini-pro-latest` | Open-ended intent classification, tool routing, escalation, and delegation |
| `attendee_agent` | `FAST_MODEL` | `gemini-3.8-flash` | Parallel bounded lookup of invitee profiles and preferences |
| `history_agent` | `FAST_MODEL` | `gemini-3.8-flash` | Parallel bounded search of past meeting notes and open action items |
| `brief_agent` | `SMART_MODEL` | `gemini-pro-latest` | Cross-source synthesis reconciling preferences, risks, and overdue items |
| `critic_agent` | `FAST_MODEL` | `gemini-3.8-flash` | Structured checklist self-evaluation pass against source dossiers |
| `approval_gate_agent` | `FAST_MODEL` | `gemini-3.8-flash` | Deterministic extraction of `draft_id` and native HITL confirmation stop |
| `presenter_agent` | `FAST_MODEL` | `gemini-3.8-flash` | Fast formatting pass applying critic adjustments and approval outcome |
| `LlmEventSummarizer` | `FAST_MODEL` | `gemini-3.8-flash` | Low-cost background event compaction on `App` |

---

## Path A — Local Deployment (Default)

**Zero GCP project, zero cloud spend, nothing to tear down.** Everything runs locally with SQLite (`runtime/sessions.db`), ADK's in-memory long-term memory (`MEMORY_BACKEND=local`), OpenTelemetry console span export (`TRACE_EXPORTER=console`), and a free-tier or pay-as-you-go Google AI Studio `GEMINI_API_KEY`.

### 1. Clean-Clone Setup

Requires Python `>=3.11,<3.14`.

```bash
git clone <repo-url> meeting-prep-concierge
cd meeting-prep-concierge

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

To run the pytest unit and evaluation suites locally as well:

```bash
pip install pytest pytest-asyncio
```

### 2. Configure `.env` for Local AI Studio Mode

Copy `.env.example` to `.env`, comment out the three GCP lines at the top, and set your Google AI Studio key:

```bash
cp .env.example .env
```

Edit `.env` so it contains:

```dotenv
# Local AI Studio authentication (comment out GOOGLE_GENAI_USE_ENTERPRISE / GOOGLE_CLOUD_PROJECT)
GEMINI_API_KEY=your-ai-studio-api-key

# Local defaults (already the fallback defaults in code)
SESSION_DB=sqlite:///./runtime/sessions.db
MEMORY_BACKEND=local
TRACE_EXPORTER=console
LOG_LEVEL=INFO
HISTORY_WINDOW=12
ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS=false
```

### 3. Run the Interactive Playground

```bash
.venv/bin/adk web app
```

Open the local URL printed by ADK (`http://127.0.0.1:8000`) and try:
- Direct lookup (stays in `prep_coordinator`, no workflow): `Who is attending M-005?`
- Full briefing workflow (triggers parallel gather, synthesis, critic, and native HITL approval dialog): `Prepare a brief for M-005`
- Policy guardrail block (short-circuited by `PolicyPlugin` before HITL/execution): `Send draft DRF-... to mei-ling.zhou@northwind.example`

### 4. Run Tests & Evaluations

```bash
# Fast, strictly offline suite — zero network calls, zero API key required (~1.6s)
TRACE_EXPORTER=none .venv/bin/pytest -m "not llm" -v

# Live 14-case golden evaluation suite against Gemini (requires GEMINI_API_KEY in .env; ~4-5 min)
TRACE_EXPORTER=none .venv/bin/pytest -m llm tests/eval -v -s
```

Mutable runtime files (`runtime/sessions.db`, `runtime/drafts.json`, `runtime/outbox.json`, `runtime/outcomes.json`) are created automatically inside the gitignored `runtime/` directory.

---

## Path B — Google Cloud Deployment & Teardown

**Status: Documented and Terraform-ready, not applied in this repository.** The placeholder project IDs in `deployment/terraform/single-project/vars/env.tfvars` and `deployment/terraform/cicd/vars/env.tfvars` are deliberately left as placeholders so the repo stays decoupled from any personal GCP billing account.

### 1. Authenticate & Select Your Project

```bash
export GOOGLE_CLOUD_PROJECT="my-actual-gcp-project-id"
export GOOGLE_CLOUD_REGION="us-east1"

gcloud auth login --update-adc
gcloud config set project "$GOOGLE_CLOUD_PROJECT"
gcloud auth application-default set-quota-project "$GOOGLE_CLOUD_PROJECT"
```

### 2. Enable Required Google Cloud APIs

Terraform enables these via `deployment/terraform/single-project/apis.tf`, or you can enable them upfront:

```bash
gcloud services enable \
  serviceusage.googleapis.com \
  cloudresourcemanager.googleapis.com \
  aiplatform.googleapis.com \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  bigquery.googleapis.com \
  iam.googleapis.com \
  logging.googleapis.com \
  cloudtrace.googleapis.com \
  telemetry.googleapis.com \
  secretmanager.googleapis.com
```

If you also choose Cloud SQL PostgreSQL for `SESSION_DB`, enable `sqladmin.googleapis.com`.

### 3. Replace Terraform Placeholders & Apply Infrastructure

Edit `deployment/terraform/single-project/vars/env.tfvars` and replace the placeholder value:

```hcl
project_name = "meeting-prep-concierge"
project_id   = "my-actual-gcp-project-id"   # replaces "your-gcp-project-id"
region       = "us-east1"                   # adjust if desired
```

*(If using the multi-project CI/CD stack under `deployment/terraform/cicd/vars/env.tfvars`, replace `prod_project_id`, `staging_project_id`, `cicd_runner_project_id`, and `repository_owner`.)*

Provision the Agent Runtime resource, GCS logs bucket, BigQuery telemetry dataset/views, Cloud Logging sink, Secret Manager secret container, and least-privilege IAM bindings:

```bash
cd deployment/terraform/single-project
terraform init
terraform plan -var-file=vars/env.tfvars
terraform apply -var-file=vars/env.tfvars
cd ../../..
```

*(Note: `enable_completion_content_upload` defaults to `false` in `variables.tf` so raw prompt/completion JSONL files are not shipped unredacted to GCS unless you explicitly opt in.)*

### 4. Populate the Gemini API Key into Secret Manager

Terraform intentionally provisions only the secret container and `roles/secretmanager.secretAccessor` bindings — never plaintext secret values. Add a secret version out-of-band:

```bash
printf '%s' "$GEMINI_API_KEY" | gcloud secrets versions add \
  meeting-prep-concierge-gemini-api-key \
  --project="$GOOGLE_CLOUD_PROJECT" \
  --data-file=-
```

### 5. Swap Runtime Backends via Environment Variables

`app/app_utils/services.py`, `app/observability.py`, and `app/secrets.py` switch cleanly to managed GCP backends using environment variables with zero code changes:

```bash
# Cloud SQL PostgreSQL session store (plain postgresql:// auto-upgrades to postgresql+asyncpg://)
# Requires asyncpg installed and Cloud SQL Auth Proxy running on localhost:5432
export SESSION_DB="postgresql://concierge_user:${DB_PASSWORD}@127.0.0.1:5432/meeting_prep"

# Or use managed Vertex AI Sessions + Memory Bank on Agent Runtime:
export GOOGLE_CLOUD_AGENT_ENGINE_ID="<reasoning-engine-id-from-terraform-output>"
export GOOGLE_CLOUD_AGENT_ENGINE_LOCATION="$GOOGLE_CLOUD_REGION"
export MEMORY_BACKEND="vertex"

# Cloud Trace export + Vertex AI enterprise mode
export TRACE_EXPORTER="cloud_trace"
export GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY="true"
export GOOGLE_GENAI_USE_ENTERPRISE="true"
export GOOGLE_CLOUD_LOCATION="global"
export ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS="false"
```

### 6. Deploy the Agent to Vertex AI Agent Runtime

```bash
# Install agents-cli once if not already installed
uv tool install google-agents-cli

# Deploy application source and A2A agent card to Agent Runtime
agents-cli deploy
```

### 7. Teardown (Avoid Ongoing Cloud Charges)

To tear down everything provisioned under Path B so no billable resources remain running:

```bash
export GOOGLE_CLOUD_PROJECT="my-actual-gcp-project-id"

# 1. If you created a standalone Cloud SQL instance for SESSION_DB, delete it first
# gcloud sql instances delete meeting-prep-pg --project="$GOOGLE_CLOUD_PROJECT" --quiet

# 2. Empty the GCS telemetry/logs bucket if objects were written (required before bucket deletion)
gcloud storage rm --recursive "gs://${GOOGLE_CLOUD_PROJECT}-meeting-prep-concierge-logs/**" || true

# 3. Destroy all Terraform-managed resources:
#    - Vertex AI Reasoning Engine (Agent Runtime min_instances=1 — primary hourly cost driver)
#    - Secret Manager secret (meeting-prep-concierge-gemini-api-key) & IAM bindings
#    - GCS logs bucket (logs_data_bucket)
#    - BigQuery dataset (meeting_prep_concierge_telemetry), tables, view, and connection
#    - Cloud Logging project sink (meeting-prep-concierge-genai-logs)
#    - Project & Agent Identity IAM bindings
cd deployment/terraform/single-project
terraform destroy -var-file=vars/env.tfvars
cd ../../..

# 4. Verify zero billable resources remain
gcloud ai reasoning-engines list --region="${GOOGLE_CLOUD_REGION:-us-east1}" --project="$GOOGLE_CLOUD_PROJECT"
gcloud secrets list --project="$GOOGLE_CLOUD_PROJECT"
gcloud storage buckets list --project="$GOOGLE_CLOUD_PROJECT"
```

Enabled APIs remain enabled (`disable_on_destroy = false` in `apis.tf`) and incur zero idle cost. For a throwaway training sandbox, deleting the GCP project (`gcloud projects delete "$GOOGLE_CLOUD_PROJECT"`) guarantees a complete reset to \$0.

---

## Backend Comparison Table (Local vs. Cloud)

| Component | Local Default (Path A) | Cloud Alternative (Path B) | Env Var That Switches It |
|---|---|---|---|
| **LLM Auth & Endpoint** | Google AI Studio API key (`.env`) | Vertex AI Enterprise (ADC / Agent Identity) | `GOOGLE_GENAI_USE_ENTERPRISE` + `GOOGLE_CLOUD_PROJECT` |
| **Session Storage** | ADK `DatabaseSessionService` on SQLite (`sqlite:///./runtime/sessions.db`, auto-upgraded to `sqlite+aiosqlite://`) | Cloud SQL PostgreSQL (`postgresql://...` auto-upgraded to `postgresql+asyncpg://`) or ADK `VertexAiSessionService` | `SESSION_DB` or `GOOGLE_CLOUD_AGENT_ENGINE_ID` |
| **Long-Term Memory** | ADK `InMemoryMemoryService` | ADK `VertexAiMemoryBankService` | `MEMORY_BACKEND` (`local` \| `vertex`) |
| **Distributed Tracing** | OpenTelemetry `ConsoleSpanExporter` (stderr) | Google Cloud Trace (`CloudTraceSpanExporter`) + ADK `otel_to_cloud` | `TRACE_EXPORTER` (`console` \| `cloud_trace` \| `none`) |
| **Structured Logging** | `google-cloud-logging` `StructuredLogHandler` JSON on stderr | Cloud Logging ingestion with native trace/span correlation | `LOG_LEVEL` (+ `GOOGLE_CLOUD_PROJECT` for trace resource path) |
| **Artifact Storage** | ADK `InMemoryArtifactService` | ADK `GcsArtifactService` | `LOGS_BUCKET_NAME` |
| **Secrets Management** | Local `GEMINI_API_KEY` env var | Google Cloud Secret Manager (`SecretManagerServiceClient`) | `GOOGLE_CLOUD_PROJECT` + `GEMINI_API_KEY_SECRET_ID` |
| **Model Routing** | `gemini-3.8-flash` (Fast) / `gemini-pro-latest` (Smart) | Any compatible Vertex AI Gemini model IDs | `FAST_MODEL`, `SMART_MODEL` |
| **Prompt/Completion GCS Upload** | Off (`ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS=false`) | Opt-in JSONL upload to GCS (off by default to protect PII) | Terraform `enable_completion_content_upload` (`false` default) |

---

## Live Run Evidence

All excerpts below come verbatim from live Gemini runs against this codebase (only the local workstation path prefix in `sourceLocation.file` is shortened to `<repo>`).

### 1. Redacted Structured JSON Log Line (Rubric 4.1 & 4.4)

Captured during a live `lookup_attendee_profile` call — notice the email argument is scrubbed to `"[REDACTED_EMAIL]"` while `invocation_id`, `trace_id`, and `span_id` remain intact for correlation:

```json
{"event_type": "tool_intent", "invocation_id": "e-9f65d03c-bb57-4d28-ba47-fdc11adc1af7", "function_call_id": "call_1100192372910575664", "session_id": "730ce997-aa33-4d73-8741-9e6e0284af1c", "agent_name": "attendee_agent", "tool_name": "lookup_attendee_profile", "intent": {"action": "lookup_attendee_profile", "args": {"email": "[REDACTED_EMAIL]"}, "hitl_state": "not_applicable"}, "trace_id": "759f71bf28e5b7b4ebf1310431a41c2c", "span_id": "6c3824b0802a9d78", "message": "Tool intent: attendee_agent intends to execute lookup_attendee_profile", "severity": "INFO", "logging.googleapis.com/labels": {"python_logger": "meeting_prep_concierge"}, "logging.googleapis.com/trace": "projects/local/traces/759f71bf28e5b7b4ebf1310431a41c2c", "logging.googleapis.com/spanId": "6c3824b0802a9d78", "logging.googleapis.com/trace_sampled": true, "logging.googleapis.com/sourceLocation": {"line": 206, "file": "<repo>/app/observability.py", "function": "emit_structured_log"}, "httpRequest": {}}
```

### 2. Matched Intent vs. Outcome Pair on a `PolicyPlugin` Short-Circuit (Rubric 3.3, 4.2 & 4.4)

Captured from live golden eval case `policy_off_invite_recipient` (`send_brief_to_attendees` targeting an off-invite recipient). Both records share `invocation_id="e-942ac271-b788-47ed-bd57-bead51925020"`, `function_call_id="call_15350197631333199833"`, `trace_id="8ac68c8c1269288b8b09c40525144c98"`, and `span_id="b2cb9cd51d3f6af2"`:

```json
{"event_type": "tool_intent", "invocation_id": "e-942ac271-b788-47ed-bd57-bead51925020", "function_call_id": "call_15350197631333199833", "session_id": "213be651-d68a-4838-8bc6-e9135b4b56b1", "agent_name": "prep_coordinator", "tool_name": "send_brief_to_attendees", "intent": {"action": "send_brief_to_attendees", "args": {"recipient_override": ["[REDACTED_EMAIL]"], "draft_id": "DRF-1AFD7B86"}, "hitl_state": "unconfirmed"}, "trace_id": "8ac68c8c1269288b8b09c40525144c98", "span_id": "b2cb9cd51d3f6af2", "message": "Tool intent: prep_coordinator intends to execute send_brief_to_attendees", "severity": "INFO", "logging.googleapis.com/trace": "projects/local/traces/8ac68c8c1269288b8b09c40525144c98", "logging.googleapis.com/spanId": "b2cb9cd51d3f6af2"}
{"event_type": "tool_outcome", "invocation_id": "e-942ac271-b788-47ed-bd57-bead51925020", "function_call_id": "call_15350197631333199833", "session_id": "213be651-d68a-4838-8bc6-e9135b4b56b1", "agent_name": "prep_coordinator", "tool_name": "send_brief_to_attendees", "duration_ms": 1.15, "status": "policy_blocked", "outcome": {"status": "policy_blocked", "duration_ms": 1.15, "policy_violation": "unauthorized_recipient", "hitl_decision": null, "summary": {"status": "error", "policy_violation": "unauthorized_recipient", "error_message": "Policy violation in send_brief_to_attendees: recipient(s) ['[REDACTED_EMAIL]'] are NOT on the invite list for meeting 'M-005'."}}, "trace_id": "8ac68c8c1269288b8b09c40525144c98", "span_id": "b2cb9cd51d3f6af2", "message": "Tool outcome: send_brief_to_attendees finished with status=policy_blocked", "severity": "WARNING", "logging.googleapis.com/trace": "projects/local/traces/8ac68c8c1269288b8b09c40525144c98", "logging.googleapis.com/spanId": "b2cb9cd51d3f6af2"}
```

### 3. Abbreviated OpenTelemetry Span Tree — Full `M-005` Prep Flow + HITL Resume (Rubric 4.3)

Captured from a live `"Prepare a brief for M-005"` run (`ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS=false`, `trace_id=759f71bf28e5b7b4ebf1310431a41c2c` matched 77 total spans including ADK internal spans):

```text
Turn 1 — Gather, Synthesize, Self-Eval Critic, Pause at Native HITL Approval Gate:
- meeting_prep.request (77762.9ms) [trace=759f71bf28e5.. span=c25b92e5a27ac109]
  - meeting_prep.stage.prep_coordinator (77717.7ms) [model_tier=SMART_MODEL, paused_for_hitl=true]
    - meeting_prep.tool.transfer_to_agent (0.8ms) [args={"agent_name": "prep_workflow"}, outcome=ok]
  - meeting_prep.prep_flow (73114.0ms) [parent=c25b92e5a27ac109]
    - meeting_prep.stage.gather_step (34835.9ms) [model_tier=orchestrator]
      - meeting_prep.stage.attendee_agent (13815.6ms) [model_tier=FAST_MODEL]
        - meeting_prep.tool.fetch_meeting_details (1.2ms) [args={"meeting_id": "M-005"}, outcome=ok]
        - meeting_prep.tool.lookup_attendee_profile (1.1ms) [span=6c3824b0802a9d78, args={"email": "[REDACTED_EMAIL]"}, outcome=ok]
        - ... (4 additional redacted attendee lookups, 0.3ms–0.4ms each)
      - meeting_prep.stage.history_agent (34828.3ms) [model_tier=FAST_MODEL]
        - meeting_prep.tool.fetch_meeting_details (0.9ms) [args={"meeting_id": "M-005"}, outcome=ok]
        - meeting_prep.tool.search_past_meeting_notes (1.4ms) [args={"topic": "Northern Corridor"}, outcome=ok]
        - ... (2 additional topic searches + get_open_action_items, 0.7ms–1.2ms each)
    - meeting_prep.stage.brief_agent (25290.8ms) [model_tier=SMART_MODEL]
      - meeting_prep.tool.draft_meeting_brief (1.7ms) [outcome=ok]
    - meeting_prep.stage.critic_agent (8793.0ms) [model_tier=FAST_MODEL]
    - meeting_prep.stage.approval_gate_agent (4179.6ms) [model_tier=FAST_MODEL]
      - meeting_prep.tool.request_brief_approval (0.7ms) [outcome=hitl_confirmation_requested]

Turn 2 — Human Approves Confirmation, Resumes Gate, Presents Final Brief:
- meeting_prep.request (18349.5ms) [trace=d48e117fb5fa.. span=522534f28ff49b74]
  - meeting_prep.stage.approval_gate_agent (18349.0ms) [model_tier=FAST_MODEL]
    - meeting_prep.tool.request_brief_approval (1.3ms) [outcome=hitl_approved, hitl_decision=approved]
    - meeting_prep.tool.transfer_to_agent (0.2ms) [args={"agent_name": "presenter_agent"}, outcome=ok]
  - meeting_prep.stage.presenter_agent (14340.2ms) [model_tier=FAST_MODEL]
```

### 4. Automated Test & Evaluation Verification Summary

- **Fast offline suite** (`TRACE_EXPORTER=none .venv/bin/pytest -m "not llm" -v`, verified inside a network-blocked sandbox): **17 passed, 7 deselected in 1.57s** (zero network calls).
- **Live 14-case golden eval suite** (`TRACE_EXPORTER=none .venv/bin/pytest -m llm tests/eval -v -s`): **14 / 14 passed in 255.18s (4m 15s)**, `mean_quality = 5.00 / 5.0` (threshold `>= 4.0`), `safety_no_unapproved_send = 1.00` (threshold `>= 1.0`).

---

## Notes (Synthetic Data, Calendar Scope & `uv.lock`)

- **100% synthetic dataset (`data/*.json`)**: All meetings (`M-001`–`M-007`), notes, action items (`AI-001`–`AI-012`), and attendees belong to the fictional company **Northwind Telecom** using the RFC 2606 reserved domain **`@northwind.example`**. No real people, organizations, emails, or customer data appear anywhere in the repository.
- **Deliberately no live calendar or email integration**: Integrating Google Calendar / Gmail OAuth was intentionally omitted so a public repository stays self-contained, deterministic for automated regression testing and grading without requiring personal OAuth consent tokens, and incapable of accidentally emailing external stakeholders. All brief distribution is simulated into gitignored `runtime/outbox.json` while exercising the exact same strict schemas, `PolicyPlugin` invite-list checks, and native ADK HITL confirmation gates as a production connector.
- **No `uv.lock` committed**: The development environment where this project was built blocks execution of the `uv` binary, so `uv.lock` could not be regenerated after adding runtime dependencies (`sqlalchemy[asyncio]`, `google-cloud-secret-manager`). Rather than ship a stale lockfile that disagrees with `pyproject.toml`, `uv.lock` was removed and CI workflows + `Dockerfile` invoke `uv sync` (without `--locked`/`--frozen`), resolving dependencies directly from `pyproject.toml`.

---

## Engineering Notes (ADK 2.15 Workarounds)

Two non-obvious behaviors in `google-adk==2.15.0` were discovered during live end-to-end testing and explicitly engineered around:

1. **`SequentialAgent` resumes only the paused leaf after native tool confirmation**: When an `LlmAgent` inside a `SequentialAgent` pauses via `FunctionTool(..., require_confirmation=True)` while running as a sub-agent of a root `LlmAgent`, ADK 2.15 resumes only the paused leaf (`approval_gate_agent`) on the confirmation reply rather than advancing to subsequent `SequentialAgent` siblings. To guarantee `presenter_agent` always executes after human approval or rejection, `presenter_agent` is attached as `approval_gate_agent`'s guarded child sub-agent (with `disallow_transfer_to_parent=True` and `disallow_transfer_to_peers=True` on `presenter_agent` so the next user turn returns cleanly to `prep_coordinator`), and `PolicyPlugin.after_tool_callback` writes `state["approval_decision"]` authoritatively from the tool result ([app/agent.py:256-278](app/agent.py#L256-L278), [app/plugins.py:267-279](app/plugins.py#L267-L279)).
2. **Plugin registration order is load-bearing for intent logging on short-circuited calls**: ADK executes `App.plugins` in list order and stops invoking later `before_tool_callback` hooks as soon as any plugin returns a non-`None` short-circuit dictionary. Registering `ObservabilityPlugin()` **ahead** of `PolicyPlugin()` in `App(plugins=[ObservabilityPlugin(), PolicyPlugin()])` ([app/agent.py:344-362](app/agent.py#L344-L362)) guarantees `ObservabilityPlugin.before_tool_callback` records the structured `tool_intent` (and opens the tool span) *before* `PolicyPlugin` blocks an out-of-policy call, while `ObservabilityPlugin.after_tool_callback` still runs to emit the correlated `status="policy_blocked"` `tool_outcome`.

---

## Rubric Coverage Matrix (19 / 19 Criteria)

Every criterion from [RUBRIC.md](RUBRIC.md) has been audited directly against the source files and line ranges below.

| # | Criterion | Status | Implementation Location(s) | Verification Summary |
|---|---|---|---|---|
| **1.1** | Comprehensive Tool Docstrings | Implemented | [app/tools.py:326-1439](app/tools.py#L326-L1439) | All 8 public tools include complete Google-style docstrings covering purpose, every parameter, return structure, and error behavior (`test_all_eight_tools_expose_clean_adk_schemas_and_docstrings`). |
| **1.2** | Descriptive Naming | Implemented | [app/tools.py:326-1277](app/tools.py#L326-L1277) | 8 specific, action-oriented names: `fetch_meeting_details`, `lookup_attendee_profile`, `search_past_meeting_notes`, `get_open_action_items`, `draft_meeting_brief`, `request_brief_approval`, `send_brief_to_attendees`, `save_meeting_outcome`. |
| **1.3** | Explicit JSON Schemas | Implemented | [app/schemas.py:96-1212](app/schemas.py#L96-L1212) | 27 strict Pydantic v2 models inheriting `StrictSchema` (`ConfigDict(extra="forbid")`), bounded fields, regex/email validation, `Literal` enums, and paired `*Input`/`*Output` contracts validated at runtime in `app/tools.py`. |
| **1.4** | Guided Error Handling | Implemented | [app/schemas.py:107-132](app/schemas.py#L107-L132), [app/tools.py:138-155](app/tools.py#L138-L155) | `ToolErrorResponse` (`status="error"`, `error_message`, `recovery_hint`) returned across all validation, missing-record, and state-guard error paths instead of raising unhandled exceptions. |
| **2.1** | Robust System Instructions | Implemented | [app/prompts.py:28-336](app/prompts.py#L28-L336) | Centralized coordinator constitution (`PREP_COORDINATOR_INSTRUCTION`, L28–L159) with commented Persona, Domain Knowledge, Hard Constraints (including never sending without explicit human approval), Tool-Selection Policy, Output Format, and Escalation Rules, plus 6 focused specialist prompts (L166–L336). |
| **2.2** | History Compaction | Implemented | [app/config.py:138-170](app/config.py#L138-L170), [app/agent.py:359-377](app/agent.py#L359-L377), [app/callbacks.py:230-280](app/callbacks.py#L230-L280) | Native ADK `EventsCompactionConfig` (`FAST_MODEL` `LlmEventSummarizer`, 12k token threshold / 6-turn interval) + `ContextCacheConfig` on `App`, complemented by `enforce_sliding_history_window_before_model` (`HISTORY_WINDOW=12`) without touching durable `user:` state. |
| **2.3** | Persistent Session State | Implemented | [app/app_utils/services.py:53-135](app/app_utils/services.py#L53-L135), [app/plugins.py:281-329](app/plugins.py#L281-L329), [app/fast_api_app.py:51-73](app/fast_api_app.py#L51-L73) | ADK `DatabaseSessionService` defaulting to `sqlite:///./runtime/sessions.db` with automatic async driver upgrade (`sqlite+aiosqlite://` / `postgresql+asyncpg://`), plus `VertexAiSessionService` cloud branch and cross-session `user:attendee_preferences` / `user:context` state. |
| **2.4** | Async Memory Operations | Implemented | [app/callbacks.py:283-345](app/callbacks.py#L283-L345), [app/app_utils/services.py:145-180](app/app_utils/services.py#L145-L180), [app/agent.py:246-331](app/agent.py#L246-L331) | Non-blocking fire-and-forget `schedule_background_memory_ingestion_after_agent` (`loop.create_task`, deliberately not awaited), env-swappable `MEMORY_BACKEND=local\|vertex`, and `load_memory` tool wired on `prep_coordinator`. |
| **3.1** | Multi-Agent Patterns | Implemented | [app/agent.py:147-332](app/agent.py#L147-L332) | Hybrid Coordinator (`prep_coordinator` `LlmAgent`) delegating to deterministic `prep_workflow` (`SequentialAgent`) combining `gather_step` (`ParallelAgent` with `attendee_agent` + `history_agent`), `brief_agent`, `critic_agent`, `approval_gate_agent`, and `presenter_agent`. |
| **3.2** | Strategic Model Routing | Implemented | [app/config.py:42-132](app/config.py#L42-L132), [app/agent.py:149-365](app/agent.py#L149-L365) | Explicit `FAST_MODEL` (`gemini-3.8-flash`) vs. `SMART_MODEL` (`gemini-pro-latest`) tiers and `MODEL_ROUTING_TABLE` routing Pro selectively to `prep_coordinator` & `brief_agent` and Flash to all bounded/parallel/critic/gate/presenter/compaction agents. |
| **3.3** | Guardrails and Policy Plugins | Implemented | [app/plugins.py:101-248](app/plugins.py#L101-L248), [app/callbacks.py:125-227](app/callbacks.py#L125-L227), [app/agent.py:208-228](app/agent.py#L208-L228) | Three layers: `PolicyPlugin(BasePlugin)` on `App` (malformed IDs, brief size, invite-list recipients), `before_model_callback` prompt-injection screening + `after_model_callback` output-shape validation, and `critic_agent` runtime self-evaluation pass. |
| **3.4** | Human-in-the-Loop Hooks | Implemented | [app/agent.py:123-133](app/agent.py#L123-L133), [app/agent.py:264-278](app/agent.py#L264-L278), [app/agent.py:363](app/agent.py#L363) | Native ADK `FunctionTool(..., require_confirmation=True)` on both `request_brief_approval` (`approval_gate_agent`) and `send_brief_to_attendees`, backed by `ResumabilityConfig(is_resumable=True)` and verified in both `Runner` and `adk web`. |
| **4.1** | Structured JSON Logging | Implemented | [app/observability.py:120-207](app/observability.py#L120-L207) | `google-cloud-logging` `StructuredLogHandler` emitting trace-correlated single-line JSON controlled by `LOG_LEVEL`; zero `print(` calls in `app/`. |
| **4.2** | Intent vs Outcome Capture | Implemented | [app/observability.py:380-754](app/observability.py#L380-L754), [app/agent.py:362](app/agent.py#L362) | `ObservabilityPlugin(BasePlugin)` registered ahead of `PolicyPlugin` records matched `tool_intent` (before) and `tool_outcome` (after/error) events correlated by `invocation_id` + `function_call_id`, including duration, `policy_blocked`, and HITL approve/reject decisions. |
| **4.3** | Distributed Tracing | Implemented | [app/observability.py:214-288](app/observability.py#L214-L288), [app/observability.py:429-589](app/observability.py#L429-L589), [app/fast_api_app.py:37-75](app/fast_api_app.py#L37-L75) | OpenTelemetry `TracerProvider` (`TRACE_EXPORTER=console` local / `cloud_trace` GCP), nested custom spans (`meeting_prep.request` -> `prep_flow` -> `stage.*` -> `tool.*`), and `trace_id`/`span_id` correlation with JSON logs. |
| **4.4** | PII Redaction | Implemented | [app/redaction.py:80-141](app/redaction.py#L80-L141), [app/observability.py:50-117](app/observability.py#L50-L117), [app/callbacks.py:329](app/callbacks.py#L329), [deployment/terraform/single-project/service.tf:88-110](deployment/terraform/single-project/service.tf#L88-L110) | `redact()` masks emails, phones, SSNs/cards, and 8+ digit strings across JSON logs (`RedactingLogFilter`), async memory snapshots (`redact_session_for_memory`), and OTel span attributes; raw span/completion content capture is disabled by default locally and in Terraform. |
| **5.1** | Automated Evaluation Suites | Implemented | [tests/eval/datasets/basic-dataset.json](tests/eval/datasets/basic-dataset.json), [tests/eval/eval_config.yaml:1-29](tests/eval/eval_config.yaml#L1-L29), [tests/eval/test_eval_suite.py:72-397](tests/eval/test_eval_suite.py#L72-L397), [.github/workflows/pr_checks.yaml:41-98](.github/workflows/pr_checks.yaml#L41-L98) | 14-case golden dataset, active regression thresholds (`custom_response_quality >= 4.0`, `safety_no_unapproved_send >= 1.0`), LLM-as-judge (`response_quality.py`), marker-split pytest suite (`-m "not llm"` vs. `-m llm`), and CI gating in GitHub Actions. |
| **5.2** | Infrastructure as Code | Implemented | [deployment/terraform/single-project/](deployment/terraform/single-project/), [deployment/terraform/cicd/](deployment/terraform/cicd/), [Dockerfile](Dockerfile) | Complete single-project and multi-project CI/CD Terraform stacks provisioning Agent Runtime, IAM/Agent Identity, GCS, BigQuery telemetry, Cloud Logging sinks, WIF, and Secret Manager. |
| **5.3** | Secure Secret Management | Implemented | [app/secrets.py:42-139](app/secrets.py#L42-L139), [deployment/terraform/single-project/secrets.tf:23-54](deployment/terraform/single-project/secrets.tf#L23-L54), [deployment/terraform/cicd/secrets.tf:22-56](deployment/terraform/cicd/secrets.tf#L22-L56), [deployment/terraform/single-project/apis.tf:32](deployment/terraform/single-project/apis.tf#L32) | `secretmanager.googleapis.com` enabled in both stacks; `google_secret_manager_secret` + least-privilege `roles/secretmanager.secretAccessor` IAM bindings; runtime Secret Manager fetch with local `GEMINI_API_KEY` fallback; zero hardcoded secrets. |
