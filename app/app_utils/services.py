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

"""Process-wide ADK session, artifact, and memory services shared across surfaces.

Registered under ``shared://`` so the ADK web routes, the A2A path, and the
reasoning_engine adapter share one backend instance: a session or memory created
on any surface is visible to the others.

Env-var backend swapping (Rubric 2.3 & 2.4):
    - Sessions (``get_session_service``):
        * Local default: ``SESSION_DB`` defaults to ``sqlite:///./runtime/sessions.db``
          (auto-upgraded to ``sqlite+aiosqlite://``) backed by ADK's
          ``DatabaseSessionService``.
        * Cloud SQL drop-in: set ``SESSION_DB=postgresql://user:pass@host/db``
          (auto-upgraded to ``postgresql+asyncpg://``).
        * Vertex AI cloud path: set ``GOOGLE_CLOUD_AGENT_ENGINE_ID`` (plus
          ``GOOGLE_CLOUD_PROJECT`` / ``GOOGLE_CLOUD_AGENT_ENGINE_LOCATION``) to
          use ``VertexAiSessionService``.
    - Memory (``get_memory_service``):
        * Local default: ``MEMORY_BACKEND=local`` uses ADK's
          ``InMemoryMemoryService``.
        * Cloud path: ``MEMORY_BACKEND=vertex`` switches to ADK's
          ``VertexAiMemoryBankService``.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

from google.adk.artifacts import GcsArtifactService, InMemoryArtifactService
from google.adk.cli.service_registry import get_service_registry
from google.adk.cli.utils.service_factory import create_session_service_from_options
from google.adk.memory import InMemoryMemoryService

SESSION_SERVICE_URI = "shared://session"
ARTIFACT_SERVICE_URI = "shared://artifact"
MEMORY_SERVICE_URI = "shared://memory"

DEFAULT_SESSION_DB = "sqlite:///./runtime/sessions.db"

_AGENT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def normalize_async_db_url(raw_url: str) -> str:
    """Upgrade synchronous SQLite/PostgreSQL URLs to async drivers for ADK DatabaseSessionService.

    ADK's ``DatabaseSessionService`` requires an async SQLAlchemy driver
    (``aiosqlite`` for SQLite, ``asyncpg`` for PostgreSQL). Upgrading plain
    ``sqlite://`` and ``postgresql://`` (or ``postgres://``) URLs here allows a
    standard local SQLite or Cloud SQL PostgreSQL connection string to drop in
    unchanged. For file-backed SQLite URLs, ensures the parent directory (such
    as gitignored ``runtime/``) exists before engine initialization.
    """
    url = raw_url.strip()
    if not url:
        url = DEFAULT_SESSION_DB

    # Auto-upgrade plain sync drivers to their async counterparts.
    if url.startswith("sqlite://") and not url.startswith("sqlite+"):
        url = "sqlite+aiosqlite://" + url[len("sqlite://") :]
    elif url.startswith("postgresql://"):
        url = "postgresql+asyncpg://" + url[len("postgresql://") :]
    elif url.startswith("postgres://"):
        url = "postgresql+asyncpg://" + url[len("postgres://") :]

    # Ensure the parent directory exists for local file-backed SQLite URLs.
    if url.startswith("sqlite+aiosqlite:///"):
        db_part = url[len("sqlite+aiosqlite:///") :]
        db_path_str = db_part.split("?", 1)[0]
        if db_path_str and db_path_str != ":memory:":
            if db_path_str.startswith("/"):
                db_path = Path(db_path_str)
            else:
                # Resolve relative SQLite paths (e.g. ./runtime/sessions.db)
                # anchored to the repository root so the DB always lands inside
                # the gitignored runtime/ directory regardless of CWD.
                db_path = (Path(_AGENT_DIR) / db_path_str).resolve()
                query_suffix = (
                    "?" + db_part.split("?", 1)[1] if "?" in db_part else ""
                )
                url = f"sqlite+aiosqlite:///{db_path}{query_suffix}"
            db_path.parent.mkdir(parents=True, exist_ok=True)

    return url


@functools.cache
def get_session_service():
    """Process-wide persistent session service shared across every serving surface.

    Selection order:
      1. Explicit ``SESSION_SERVICE_URI`` env var (ADK URI factory).
      2. ``GOOGLE_CLOUD_AGENT_ENGINE_ID`` env var -> ``VertexAiSessionService``.
      3. ``SESSION_DB`` env var (default ``sqlite:///./runtime/sessions.db``) ->
         ADK ``DatabaseSessionService`` with auto-upgraded async driver.
    """
    if uri := os.environ.get("SESSION_SERVICE_URI"):
        return create_session_service_from_options(
            base_dir=_AGENT_DIR, session_service_uri=uri
        )
    if agent_engine_id := os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_ID"):
        from google.adk.sessions.vertex_ai_session_service import VertexAiSessionService

        return VertexAiSessionService(
            project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
            # Runtime-injected agent-engine region, not GOOGLE_CLOUD_LOCATION
            # (which agent config pins to "global").
            location=os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_LOCATION")
            or os.environ.get("GOOGLE_CLOUD_LOCATION"),
            agent_engine_id=agent_engine_id,
        )

    from google.adk.sessions.database_session_service import DatabaseSessionService

    db_url = normalize_async_db_url(
        os.environ.get("SESSION_DB", DEFAULT_SESSION_DB)
    )
    return DatabaseSessionService(db_url=db_url)


@functools.cache
def get_artifact_service():
    """Process-wide artifact service: GCS when a bucket is set, else in-memory."""
    if bucket := os.environ.get("LOGS_BUCKET_NAME"):
        return GcsArtifactService(bucket_name=bucket)
    return InMemoryArtifactService()


@functools.cache
def get_memory_service():
    """Process-wide memory service selectable via ``MEMORY_BACKEND`` env var.

    - ``MEMORY_BACKEND=local`` (default): ADK ``InMemoryMemoryService`` (no GCP required).
    - ``MEMORY_BACKEND=vertex``: ADK ``VertexAiMemoryBankService`` for durable
      cloud memory extraction and retrieval.
    """
    backend = os.environ.get("MEMORY_BACKEND", "local").strip().lower() or "local"

    if backend == "local":
        return InMemoryMemoryService()

    if backend == "vertex":
        from google.adk.memory import VertexAiMemoryBankService

        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_LOCATION") or os.environ.get(
            "GOOGLE_CLOUD_LOCATION"
        )
        agent_engine_id = os.environ.get("GOOGLE_CLOUD_AGENT_ENGINE_ID")
        if not project or not agent_engine_id:
            raise ValueError(
                "MEMORY_BACKEND=vertex requires GOOGLE_CLOUD_PROJECT and "
                "GOOGLE_CLOUD_AGENT_ENGINE_ID to be set."
            )
        return VertexAiMemoryBankService(
            project=project,
            location=location,
            agent_engine_id=agent_engine_id,
        )

    raise ValueError(
        f"Unsupported MEMORY_BACKEND={backend!r}. Expected 'local' or 'vertex'."
    )


def clear_service_caches() -> None:
    """Clear cached service singletons (intended for test isolation across env swaps)."""
    get_session_service.cache_clear()
    get_artifact_service.cache_clear()
    get_memory_service.cache_clear()


_registry = get_service_registry()
_registry.register_session_service("shared", lambda uri, **kw: get_session_service())
_registry.register_artifact_service("shared", lambda uri, **kw: get_artifact_service())
_registry.register_memory_service("shared", lambda uri, **kw: get_memory_service())
