"""
Official OpenEnv FastAPI app entrypoint.
"""

from __future__ import annotations

import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from models import MigrationAction, MigrationObservation
from server.environment import MigrationEnvironment

try:
    from openenv.core.env_server import create_fastapi_app

    _SINGLETON = MigrationEnvironment()

    def _env_factory() -> MigrationEnvironment:
        return _SINGLETON

    app = create_fastapi_app(
        _env_factory,
        MigrationAction,
        MigrationObservation,
        max_concurrent_envs=1,
    )
except ImportError:
    from fastapi import FastAPI

    app = FastAPI(title="schema-migration-openenv")
    _ENV = MigrationEnvironment()

    @app.get("/health")
    def health():
        return {"status": "healthy"}

    @app.post("/reset")
    def reset(payload: dict | None = None):
        payload = payload or {}
        obs = _ENV.reset(task=payload.get("task", "add_columns"))
        return obs.model_dump()

    @app.post("/step")
    def step(payload: dict):
        action_payload = payload.get("action", payload)
        obs = _ENV.step(MigrationAction(**action_payload))
        return {
            "observation": obs.model_dump(),
            "reward": obs.reward,
            "done": obs.done,
        }

    @app.get("/state")
    def state():
        return _ENV.state.model_dump() if hasattr(_ENV.state, "model_dump") else _ENV.state.dict()


def main() -> None:
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
