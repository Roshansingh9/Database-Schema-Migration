"""
OpenEnv client for schema-migration-openenv.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Generic, Optional, TypeVar

try:
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient
except ImportError:
    TObs = TypeVar("TObs")

    @dataclass
    class StepResult(Generic[TObs]):
        observation: TObs
        reward: Optional[float]
        done: bool

    class EnvClient(Generic[TObs]):
        pass

from models import MigrationAction, MigrationObservation, MigrationState


class SchemaMigrationEnvClient(EnvClient):
    def _step_payload(self, action: MigrationAction) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"action_type": str(action.action_type)}
        if action.sql is not None:
            payload["sql"] = action.sql
        return payload

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult:
        obs_data = payload.get("observation", payload)
        observation = MigrationObservation(
            done=payload.get("done", obs_data.get("done", False)),
            reward=payload.get("reward", obs_data.get("reward")),
            current_schema=obs_data.get("current_schema", []),
            migration_spec=obs_data.get("migration_spec", ""),
            requirements=obs_data.get("requirements", []),
            migration_hints=obs_data.get("migration_hints", {}),
            migration_buffer=obs_data.get("migration_buffer", ""),
            execution_history=obs_data.get("execution_history", []),
            step_history=obs_data.get("step_history", []),
            execution_logs=obs_data.get("execution_logs", []),
            last_result=obs_data.get("last_result"),
            step=obs_data.get("step", 0),
            max_steps=obs_data.get("max_steps", 20),
            partial_score=obs_data.get("partial_score", 0.0),
            hint=obs_data.get("hint"),
        )
        return StepResult(observation=observation, reward=observation.reward, done=observation.done)

    def _parse_state(self, payload: Dict[str, Any]) -> MigrationState:
        return MigrationState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", payload.get("step", 0)),
            task=payload.get("task", "add_columns"),
            difficulty=payload.get("difficulty", "easy"),
            env_done=payload.get("env_done", payload.get("done", False)),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
            rollback_count=payload.get("rollback_count", 0),
            successful_executions=payload.get("successful_executions", 0),
            syntax_failures=payload.get("syntax_failures", 0),
            execution_failures=payload.get("execution_failures", 0),
        )
