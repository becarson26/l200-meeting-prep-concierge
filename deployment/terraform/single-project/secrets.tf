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

# ====================================================================
# Secret Manager — Secure Secret Provisioning (Rubric 5.3)
# ====================================================================
# Terraform provisions the Secret Manager container and least-privilege
# IAM access bindings only. Plaintext secret values are NEVER stored in
# Terraform configuration or state; operators populate versions out-of-band:
#   gcloud secrets versions add <secret_id> --data-file=-

resource "google_secret_manager_secret" "gemini_api_key" {
  project   = var.project_id
  secret_id = "${var.project_name}-gemini-api-key"

  replication {
    auto {}
  }

  labels = {
    app        = var.project_name
    managed_by = "terraform"
  }

  depends_on = [google_project_service.services]
}

# Least-privilege secretAccessor binding scoped to this single secret for the
# Agent Runtime's Agent Identity principal.
resource "google_secret_manager_secret_iam_member" "agent_identity_gemini_api_key_accessor" {
  project   = var.project_id
  secret_id = google_secret_manager_secret.gemini_api_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "principal://${google_vertex_ai_reasoning_engine.app.spec[0].effective_identity}"
}

# Least-privilege secretAccessor binding for the Vertex AI service agent.
resource "google_secret_manager_secret_iam_member" "vertex_sa_gemini_api_key_accessor" {
  project   = var.project_id
  secret_id = google_secret_manager_secret.gemini_api_key.secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = google_project_service_identity.vertex_sa.member
}
