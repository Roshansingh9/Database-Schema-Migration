"""
FastAPI application for Schema Migration OpenEnv.

Uses create_app() from openenv-core, which auto-generates all standard endpoints:
    GET  /health     — {"status": "healthy"}
    GET  /metadata   — environment name + description
    GET  /schema     — action / observation / state JSON schemas
    GET  /state      — current MigrationState
    POST /reset      — reset episode, returns MigrationObservation
    POST /step       — execute action, returns observation + reward + done
    POST /mcp        — JSON-RPC 2.0 endpoint
    WS   /ws         — WebSocket for persistent sessions
    GET  /docs       — Swagger UI

A singleton environment instance is used for HTTP calls so that state
persists across the multi-step reset → step → ... → submit sequence that
inference.py relies on.
"""

from __future__ import annotations

import os
import sys

# Ensure project root is importable when this module is loaded via uvicorn
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from openenv.core.env_server import create_app
from server.environment import MigrationEnvironment
from models import MigrationAction, MigrationObservation

# ---------------------------------------------------------------------------
# Singleton: same instance returned for every HTTP call so state persists
# across the multi-step episode that inference.py executes.
# close() is a no-op on MigrationEnvironment, so the env is never destroyed.
# ---------------------------------------------------------------------------
_SINGLETON = MigrationEnvironment()


def _env_factory() -> MigrationEnvironment:
    return _SINGLETON


app = create_app(
    _env_factory,
    MigrationAction,
    MigrationObservation,
    env_name="schema-migration-openenv",
    max_concurrent_envs=1,
)


# ---------------------------------------------------------------------------
# Custom endpoint: /grade
# Not part of the standard SDK API but used by inference.py for the fatal
# fallback path (reports seed-state score when the LLM is unavailable).
# ---------------------------------------------------------------------------
@app.post("/grade")
def grade_current():
    """Run the grader on the current DB without ending the episode."""
    score, notes = _SINGLETON.grade()
    return {"score": score, "notes": notes}


# ---------------------------------------------------------------------------
# Entry point for uv run server / pyproject.toml [project.scripts]
# ---------------------------------------------------------------------------
def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    """Start the server. Called by the 'server' console script."""
    import uvicorn

    port = int(os.getenv("PORT", str(port)))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
