"""
Official OpenEnv Environment wrapper for the legacy schema migration runtime.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

try:
    from openenv.core.env_server import Environment
except ImportError:
    class Environment:
        SUPPORTS_CONCURRENT_SESSIONS = False

        def __init__(self) -> None:
            pass

from env.legacy_environment import SchemaMigrationEnv
from env.legacy_models import MigrationAction as LegacyMigrationAction
from models import MigrationAction, MigrationObservation, MigrationState


class MigrationEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self) -> None:
        super().__init__()
        self._task_env: Optional[SchemaMigrationEnv] = None
        self._current_task = "add_columns"

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: str = "add_columns",
        **kwargs,
    ) -> MigrationObservation:
        self._current_task = task
        self._task_env = SchemaMigrationEnv(task_name=task)
        legacy_obs = self._task_env.reset()
        return self._wrap_observation(legacy_obs, done=False, reward=None)

    def step(self, action: MigrationAction, **kwargs) -> MigrationObservation:
        if self._task_env is None:
            self._task_env = SchemaMigrationEnv(task_name=self._current_task)
            self._task_env.reset()

        legacy_action = LegacyMigrationAction(action_type=str(action.action_type), sql=action.sql)
        legacy_obs, reward_obj, done, _info = self._task_env.step(legacy_action)
        return self._wrap_observation(legacy_obs, done=done, reward=float(reward_obj.value))

    @property
    def state(self) -> MigrationState:
        if self._task_env is None:
            return MigrationState(task=self._current_task)

        state = self._task_env.state()
        return MigrationState(
            step_count=state.get("step", 0),
            task=state.get("task", self._current_task),
            difficulty=state.get("difficulty", "easy"),
            env_done=state.get("done", False),
            cumulative_reward=state.get("cumulative_reward", 0.0),
            rollback_count=state.get("rollback_count", 0),
            successful_executions=state.get("successful_executions", 0),
            syntax_failures=state.get("syntax_failures", 0),
            execution_failures=state.get("execution_failures", 0),
        )

    @staticmethod
    def _wrap_observation(legacy_obs, done: bool, reward: Optional[float]) -> MigrationObservation:
        return MigrationObservation(
            done=done,
            reward=reward,
            current_schema=legacy_obs.current_schema,
            migration_spec=legacy_obs.migration_spec,
            requirements=legacy_obs.requirements,
            migration_hints=legacy_obs.migration_hints,
            migration_buffer=legacy_obs.migration_buffer,
            execution_history=legacy_obs.execution_history,
            step_history=legacy_obs.step_history,
            execution_logs=legacy_obs.execution_logs,
            last_result=legacy_obs.last_result,
            step=legacy_obs.step,
            max_steps=legacy_obs.max_steps,
            partial_score=legacy_obs.partial_score,
            hint=legacy_obs.hint,
        )
