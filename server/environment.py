"""
MigrationEnvironment — OpenEnv SDK-compliant wrapper for SchemaMigrationEnv.

Inherits from openenv.core.env_server.interfaces.Environment.
Wraps the legacy SchemaMigrationEnv (env/legacy_environment.py) which
contains all real SQLite migration logic.

The close() method is intentionally a no-op so that a singleton instance
can persist state across HTTP reset → step → step → submit sequences.
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple, List

# Ensure project root is importable when this module is loaded via uvicorn
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from openenv.core.env_server.interfaces import Environment

from models import MigrationAction, MigrationObservation, MigrationState
from env.legacy_environment import SchemaMigrationEnv
# Legacy internal action type — needed to call the legacy step()
from env.models import MigrationAction as _LegacyAction


class MigrationEnvironment(Environment):
    """
    OpenEnv-compliant wrapper environment for database schema migrations.

    Wraps SchemaMigrationEnv to expose the standard Environment interface:
        reset(**kwargs) -> MigrationObservation
        step(action)    -> MigrationObservation  (done + reward embedded)
        state property  -> MigrationState

    SUPPORTS_CONCURRENT_SESSIONS = False:
        A single global instance is used for all HTTP calls so that state
        persists across the multi-step inference.py episode loop.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self) -> None:
        super().__init__()
        self._task_env: Optional[SchemaMigrationEnv] = None
        self._current_task: str = "add_columns"

    # ------------------------------------------------------------------
    # Environment interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task: str = "add_columns",
        **kwargs,
    ) -> MigrationObservation:
        """Reset with the given task. task kwarg is forwarded from /reset requests."""
        self._current_task = task
        self._task_env = SchemaMigrationEnv(task_name=task)
        legacy_obs = self._task_env.reset()
        return self._wrap_obs(legacy_obs, done=False, reward=None)

    def step(
        self,
        action: MigrationAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> MigrationObservation:
        """Execute one action. done and reward are embedded in the returned observation."""
        if self._task_env is None:
            # Lazily initialise if step is called before reset
            self._task_env = SchemaMigrationEnv(task_name=self._current_task)
            self._task_env.reset()

        # Convert SDK action → legacy action (same fields, different base class)
        legacy_action = _LegacyAction(
            action_type=str(action.action_type),
            sql=action.sql,
        )
        legacy_obs, reward_obj, done, _info = self._task_env.step(legacy_action)
        return self._wrap_obs(legacy_obs, done=done, reward=float(reward_obj.value))

    @property
    def state(self) -> MigrationState:
        """Return current episode metadata as MigrationState."""
        if self._task_env is None:
            return MigrationState(task=self._current_task)

        s = self._task_env.state()
        return MigrationState(
            step_count=s.get("step", 0),
            task=s.get("task", self._current_task),
            difficulty=s.get("difficulty", "easy"),
            env_done=s.get("done", False),
            cumulative_reward=s.get("cumulative_reward", 0.0),
            rollback_count=s.get("rollback_count", 0),
        )

    def grade(self) -> Tuple[float, List[str]]:
        """Run the grader on the current DB without ending the episode."""
        if self._task_env is None:
            return 0.05, ["No active episode — call reset first"]
        return self._task_env.grade()

    def close(self) -> None:
        """No-op: preserves state across HTTP calls (singleton pattern)."""
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_obs(legacy_obs, done: bool, reward) -> MigrationObservation:
        """Convert legacy MigrationObservation to SDK-compliant MigrationObservation."""
        return MigrationObservation(
            # SDK-inherited fields
            done=done,
            reward=reward,
            # Fields mirrored from legacy observation
            current_schema=legacy_obs.current_schema,
            migration_spec=legacy_obs.migration_spec,
            requirements=legacy_obs.requirements,
            migration_buffer=legacy_obs.migration_buffer,
            execution_history=legacy_obs.execution_history,
            last_result=legacy_obs.last_result,
            step=legacy_obs.step,
            max_steps=legacy_obs.max_steps,
            partial_score=legacy_obs.partial_score,
            hint=legacy_obs.hint,
        )
