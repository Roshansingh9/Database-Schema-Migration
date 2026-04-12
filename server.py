"""
Compatibility FastAPI server that delegates to the legacy runtime.

The official validator should use server/app.py, but keeping this file aligned
avoids broken imports for older local workflows.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from env.legacy_environment import SchemaMigrationEnv
from env.legacy_models import MigrationAction
from tasks.task_definitions import TASKS

app = FastAPI(title="schema-migration-openenv", version="1.0.0")
_env: Optional[SchemaMigrationEnv] = None


def _get_env() -> SchemaMigrationEnv:
    global _env
    if _env is None:
        _env = SchemaMigrationEnv(task_name="add_columns")
    return _env


class ResetRequest(BaseModel):
    task: Optional[str] = "add_columns"


class StepRequest(BaseModel):
    action_type: str
    sql: Optional[str] = None


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "healthy"}


@app.post("/reset")
def reset(request: Optional[ResetRequest] = None) -> Dict[str, Any]:
    global _env
    task_name = (request.task if request else None) or "add_columns"
    if task_name not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task '{task_name}'")
    _env = SchemaMigrationEnv(task_name=task_name)
    return _env.reset().model_dump()


@app.post("/step")
def step(request: StepRequest) -> Dict[str, Any]:
    env = _get_env()
    obs, reward, done, info = env.step(MigrationAction(action_type=request.action_type, sql=request.sql))
    return {
        "observation": obs.model_dump(),
        "reward": reward.value,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state() -> Dict[str, Any]:
    return _get_env().state()


def main() -> None:
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
