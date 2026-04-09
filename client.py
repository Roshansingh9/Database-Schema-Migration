"""
SchemaMigrationEnvClient — OpenEnv client for the Schema Migration environment.

Usage (sync):
    from client import SchemaMigrationEnvClient
    from models import MigrationAction

    with SchemaMigrationEnvClient(base_url="http://localhost:7860").sync() as env:
        result = env.reset()
        print(result.observation.migration_spec)

        action = MigrationAction(action_type="inspect_schema")
        result = env.step(action)
        print(result.observation.current_schema)
        print(result.done, result.reward)
"""

from __future__ import annotations

from typing import Dict

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult

from models import MigrationAction, MigrationObservation, MigrationState


class SchemaMigrationEnvClient(
    EnvClient[MigrationAction, MigrationObservation, MigrationState]
):
    """WebSocket client for the Schema Migration OpenEnv environment."""

    def _step_payload(self, action: MigrationAction) -> Dict:
        """Serialize MigrationAction to the wire dict sent over WebSocket."""
        payload: Dict = {"action_type": str(action.action_type)}
        if action.sql is not None:
            payload["sql"] = action.sql
        return payload

    def _parse_result(
        self, payload: Dict
    ) -> StepResult[MigrationObservation]:
        """Deserialize server response into a typed StepResult."""
        obs_data = payload.get("observation", {})
        observation = MigrationObservation(
            done=payload.get("done", False),
            reward=payload.get("reward"),
            current_schema=obs_data.get("current_schema", []),
            migration_spec=obs_data.get("migration_spec", ""),
            requirements=obs_data.get("requirements", []),
            migration_buffer=obs_data.get("migration_buffer", ""),
            execution_history=obs_data.get("execution_history", []),
            last_result=obs_data.get("last_result"),
            step=obs_data.get("step", 0),
            max_steps=obs_data.get("max_steps", 20),
            partial_score=obs_data.get("partial_score", 0.0),
            hint=obs_data.get("hint"),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> MigrationState:
        """Deserialize server response into a typed MigrationState."""
        return MigrationState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task=payload.get("task", "add_columns"),
            difficulty=payload.get("difficulty", "easy"),
            env_done=payload.get("env_done", False),
            cumulative_reward=payload.get("cumulative_reward", 0.0),
            rollback_count=payload.get("rollback_count", 0),
        )
