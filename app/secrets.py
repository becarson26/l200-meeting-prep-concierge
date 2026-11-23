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

"""Secure runtime secret retrieval with env-var backend swapping (Rubric 5.3).

Follows the same local-default / cloud-swap architecture as
``app/app_utils/services.py``:
    - **Cloud path** (when ``GOOGLE_CLOUD_PROJECT`` is set): fetches secrets
      from Google Cloud Secret Manager via ``SecretManagerServiceClient`` using
      ADC / Agent Identity (least-privilege ``roles/secretmanager.secretAccessor``
      provisioned by Terraform in ``deployment/terraform/*/secrets.tf``).
    - **Local development fallback** (when ``GOOGLE_CLOUD_PROJECT`` is unset, or
      if Secret Manager is unreachable in a hybrid local setup): reads directly
      from the configured environment variable (e.g. ``GEMINI_API_KEY`` loaded
      from gitignored ``.env``).

No secret values are ever hardcoded or logged.
"""

from __future__ import annotations

import functools
import logging
import os

from app.observability import emit_structured_log, get_structured_logger

DEFAULT_GEMINI_API_KEY_SECRET_ID = "meeting-prep-concierge-gemini-api-key"


def _fetch_from_secret_manager(
    project_id: str,
    secret_id: str,
    *,
    version: str = "latest",
) -> str:
    """Read a secret payload from Google Cloud Secret Manager."""
    from google.cloud import secretmanager

    client = secretmanager.SecretManagerServiceClient()
    resource_name = (
        f"projects/{project_id}/secrets/{secret_id}/versions/{version}"
    )
    response = client.access_secret_version(request={"name": resource_name})
    return response.payload.data.decode("utf-8").strip()


def get_secret(
    secret_id: str,
    *,
    env_var: str,
    project_id: str | None = None,
    version: str = "latest",
) -> str | None:
    """Resolve a secret from GCP Secret Manager when on GCP, else from ``env_var``.

    Args:
        secret_id: Secret Manager secret name (e.g. ``meeting-prep-concierge-gemini-api-key``).
        env_var: Local environment variable fallback name (e.g. ``GEMINI_API_KEY``).
        project_id: Optional explicit GCP project ID; defaults to ``GOOGLE_CLOUD_PROJECT``.
        version: Secret Manager version alias or number (default ``"latest"``).

    Returns:
        The secret string if available, or ``None`` if unset in both backends.
    """
    resolved_project = (
        project_id if project_id is not None else os.getenv("GOOGLE_CLOUD_PROJECT", "")
    ).strip()

    if resolved_project:
        try:
            value = _fetch_from_secret_manager(
                resolved_project, secret_id, version=version
            )
            if value:
                return value
        except Exception as exc:
            # Never log secret payloads; emit a structured warning noting fallback.
            emit_structured_log(
                get_structured_logger(),
                logging.WARNING,
                "Secret Manager lookup failed; falling back to environment variable",
                json_fields={
                    "event_type": "secret_fallback",
                    "secret_id": secret_id,
                    "env_var": env_var,
                    "error_type": type(exc).__name__,
                },
            )

    local_val = os.getenv(env_var, "").strip()
    return local_val or None


@functools.cache
def get_gemini_api_key() -> str | None:
    """Return the Gemini API key via Secret Manager (on GCP) or ``GEMINI_API_KEY`` (local)."""
    secret_id = (
        os.getenv("GEMINI_API_KEY_SECRET_ID", DEFAULT_GEMINI_API_KEY_SECRET_ID).strip()
        or DEFAULT_GEMINI_API_KEY_SECRET_ID
    )
    return get_secret(secret_id, env_var="GEMINI_API_KEY")


def bootstrap_runtime_secrets() -> None:
    """Populate runtime API key from Secret Manager when running without Vertex ADC.

    On Vertex AI Agent Runtime with ``GOOGLE_GENAI_USE_ENTERPRISE=true``,
    ``google-genai`` authenticates via ADC and must not have ``GEMINI_API_KEY``
    injected (which would override enterprise mode). When ``GOOGLE_CLOUD_PROJECT``
    is set for API-key auth (``GOOGLE_GENAI_USE_ENTERPRISE`` unset/false) and
    ``GEMINI_API_KEY`` is not already present, fetches the key from Secret Manager.
    """
    use_enterprise = os.getenv("GOOGLE_GENAI_USE_ENTERPRISE", "").strip().lower() in (
        "true",
        "1",
    )
    if use_enterprise:
        return
    if os.getenv("GEMINI_API_KEY"):
        return
    if not os.getenv("GOOGLE_CLOUD_PROJECT"):
        return

    api_key = get_gemini_api_key()
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key


def clear_secret_cache() -> None:
    """Clear cached secrets (for test isolation across env swaps)."""
    get_gemini_api_key.cache_clear()
