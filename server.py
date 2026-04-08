"""
FastAPI server exposing the SchemaMigrationEnv via HTTP.

Endpoints (OpenEnv spec):
  POST /reset          → MigrationObservation
  POST /step           → {observation, reward, done, info}
  GET  /state          → current state dict
  GET  /tasks          → list of available tasks
  POST /grade          → {score, notes}  (run grader without ending episode)
  GET  /health         → {"status": "ok"}

Query param:  ?task=<task_name>  (used with /reset)
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env.environment import SchemaMigrationEnv
from env.models import MigrationAction

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Schema Migration OpenEnv",
    description=(
        "An OpenEnv-compatible RL environment where AI agents perform real "
        "database schema migrations against live SQLite databases. "
        "Graded by actually executing SQL and verifying database state."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance (stateful per-process — fine for single-worker HF Space)
_env: Optional[SchemaMigrationEnv] = None


def _get_env() -> SchemaMigrationEnv:
    global _env
    if _env is None:
        _env = SchemaMigrationEnv(task_name="add_columns")
    return _env


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task: Optional[str] = "add_columns"


class StepRequest(BaseModel):
    action_type: str
    sql: Optional[str] = None


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "env": "schema-migration-openenv", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks() -> Dict[str, Any]:
    from tasks.task_definitions import TASKS
    return {
        "tasks": [
            {
                "name": t.name,
                "difficulty": t.difficulty,
                "description": t.description,
                "max_steps": t.max_steps,
                "requirements": t.requirements,
            }
            for t in TASKS.values()
        ]
    }


@app.post("/reset")
def reset(request: Optional[ResetRequest] = None) -> Dict[str, Any]:
    """
    Reset the environment with the given task (default: add_columns).
    Returns the initial observation.
    """
    global _env
    task_name = (request.task if request else None) or "add_columns"
    from tasks.task_definitions import TASKS
    if task_name not in TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task '{task_name}'. Available: {list(TASKS.keys())}",
        )
    _env = SchemaMigrationEnv(task_name=task_name)
    obs = _env.reset()
    return obs.model_dump()


@app.post("/step")
def step(request: StepRequest) -> StepResponse:
    """Execute one action and return the resulting observation, reward, done, info."""
    env = _get_env()
    action = MigrationAction(action_type=request.action_type, sql=request.sql)
    try:
        obs, reward, done, info = env.step(action)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return StepResponse(
        observation=obs.dict(),
        reward=reward.value,
        done=done,
        info={**info, "reward_message": reward.message},
    )


@app.get("/state")
def state() -> Dict[str, Any]:
    """Return the current internal environment state."""
    return _get_env().state()


@app.post("/grade")
def grade() -> Dict[str, Any]:
    """Run the grader against the current DB state without ending the episode."""
    env = _get_env()
    score, notes = env.grade()
    return {"score": score, "notes": notes}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
