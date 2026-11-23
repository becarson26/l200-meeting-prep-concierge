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
# Secret Manager — Staging & Production Secrets (Rubric 5.3)
# ====================================================================
# Terraform creates each environment's Secret Manager container and
# least-privilege IAM bindings only. Secret versions are populated
# out-of-band so no plaintext API key is ever committed or written to state.

resource "google_secret_manager_secret" "gemini_api_key" {
  for_each  = local.deploy_project_ids
  project   = each.value
  secret_id = "${var.project_name}-gemini-api-key"

  replication {
    auto {}
  }

  labels = {
    app        = var.project_name
    env        = each.key
    managed_by = "terraform"
  }

  depends_on = [google_project_service.deploy_project_services]
}

# Least-privilege secretAccessor binding for each environment's Agent Identity.
resource "google_secret_manager_secret_iam_member" "agent_identity_gemini_api_key_accessor" {
  for_each  = local.deploy_project_ids
  project   = each.value
  secret_id = google_secret_manager_secret.gemini_api_key[each.key].secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "principal://${google_vertex_ai_reasoning_engine.app[each.key].spec[0].effective_identity}"
}

# Least-privilege secretAccessor binding for the CI/CD runner service account.
resource "google_secret_manager_secret_iam_member" "cicd_sa_gemini_api_key_accessor" {
  for_each  = local.deploy_project_ids
  project   = each.value
  secret_id = google_secret_manager_secret.gemini_api_key[each.key].secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.cicd_runner_sa.email}"
}
