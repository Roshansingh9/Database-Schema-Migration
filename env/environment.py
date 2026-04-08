"""
SchemaMigrationEnv — OpenEnv-compliant environment for database schema migration tasks.

Implements the full OpenEnv interface:
  reset()        → MigrationObservation
  step(action)   → (MigrationObservation, MigrationReward, bool, dict)
  state()        → dict  (current internal state snapshot)

The agent interacts with a real SQLite database.
Every action produces observable consequences in the live schema.
"""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from env.database import MigrationDB
from env.models import (
    ActionType,
    ExecutionResult,
    MigrationAction,
    MigrationObservation,
    MigrationReward,
    RewardBreakdown,
)
from tasks.task_definitions import TASKS, Task


class SchemaMigrationEnv:
    """
    OpenEnv-compliant environment for database schema migration.

    Usage
    -----
    env = SchemaMigrationEnv(task_name="add_columns")
    obs = env.reset()
    while True:
        action = agent.act(obs)
        obs, reward, done, info = env.step(action)
        if done:
            break
    """

    ENV_NAME = "schema-migration-openenv"
    VERSION = "1.0.0"

    def __init__(self, task_name: str = "add_columns") -> None:
        if task_name not in TASKS:
            raise ValueError(f"Unknown task '{task_name}'. Available: {list(TASKS.keys())}")
        self._task: Task = TASKS[task_name]
        self._db = MigrationDB()
        self._step_count: int = 0
        self._done: bool = False
        self._migration_buffer: str = ""
        self._execution_history: List[ExecutionResult] = []
        self._pre_snapshot: str = ""
        self._cumulative_reward: float = 0.0
        self._rollback_count: int = 0
        self._last_obs: Optional[MigrationObservation] = None
        self._episode_start: float = 0.0

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self) -> MigrationObservation:
        """
        Reset the environment to a fresh episode.
        Returns the initial observation.
        """
        self._db.close()
        self._db = MigrationDB()
        self._db.init(self._task.seed_sql)
        self._pre_snapshot = self._db.snapshot_sql()
        self._step_count = 0
        self._done = False
        self._migration_buffer = ""
        self._execution_history = []
        self._cumulative_reward = 0.0
        self._rollback_count = 0
        self._episode_start = time.time()

        obs = self._make_observation(last_result=None)
        self._last_obs = obs
        return obs

    def step(
        self, action: MigrationAction
    ) -> Tuple[MigrationObservation, MigrationReward, bool, Dict[str, Any]]:
        """
        Execute one action and return (observation, reward, done, info).

        Reward signal is provided at EVERY step (partial rewards) so the
        agent receives continuous feedback throughout the trajectory.
        """
        if self._done:
            obs = self._last_obs or self.reset()
            reward = MigrationReward(value=0.0, done=True, message="Episode already finished.")
            return obs, reward, True, {"error": "episode_done"}

        self._step_count += 1
        result, step_reward = self._dispatch(action)
        self._execution_history.append(result)

        # Check step budget
        budget_exceeded = self._step_count >= self._task.max_steps
        should_end = self._done or budget_exceeded

        # Budget penalty
        if budget_exceeded and not self._done:
            step_reward -= 0.05
            self._done = True

        self._cumulative_reward += step_reward  # uncapped — clamped only at display time

        obs = self._make_observation(last_result=result)
        obs.partial_score = max(0.0, min(1.0, self._cumulative_reward))
        self._last_obs = obs

        reward = MigrationReward(
            value=round(step_reward, 4),
            done=should_end,
            message=result.message,
        )

        info: Dict[str, Any] = {
            "step": self._step_count,
            "max_steps": self._task.max_steps,
            "action_type": action.action_type,
            "execution_success": result.success,
            "cumulative_reward": round(self._cumulative_reward, 4),
        }

        return obs, reward, should_end, info

    def state(self) -> Dict[str, Any]:
        """
        Return the current internal state as a plain dict.
        Used by OpenEnv for introspection and validation.
        """
        schema = self._db.get_schema()
        return {
            "env_name": self.ENV_NAME,
            "version": self.VERSION,
            "task": self._task.name,
            "difficulty": self._task.difficulty,
            "step": self._step_count,
            "max_steps": self._task.max_steps,
            "done": self._done,
            "migration_buffer": self._migration_buffer,
            "schema": [t.model_dump() for t in schema],
            "execution_history_length": len(self._execution_history),
            "cumulative_reward": round(self._cumulative_reward, 4),
            "rollback_count": self._rollback_count,
            "elapsed_seconds": round(time.time() - self._episode_start, 2),
        }

    def grade(self) -> Tuple[float, List[str]]:
        """
        Run the task-specific grader against the current DB state.
        Returns (score_0_to_1, explanation_notes).
        Called automatically on SUBMIT, but can also be called manually.
        """
        score, notes = self._task.grader(self._db, self._pre_snapshot)
        return score, notes

    def available_tasks(self) -> List[str]:
        return list(TASKS.keys())

    # ------------------------------------------------------------------
    # Action dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, action: MigrationAction) -> Tuple[ExecutionResult, float]:
        """Route action to the appropriate handler. Returns (result, step_reward)."""
        atype = ActionType(action.action_type)

        if atype == ActionType.WRITE_MIGRATION:
            return self._handle_write(action)
        elif atype == ActionType.EXECUTE:
            return self._handle_execute(action)
        elif atype == ActionType.ROLLBACK:
            return self._handle_rollback()
        elif atype == ActionType.INSPECT_SCHEMA:
            return self._handle_inspect()
        elif atype == ActionType.RUN_QUERY:
            return self._handle_query(action)
        elif atype == ActionType.SUBMIT:
            return self._handle_submit()
        else:
            return ExecutionResult(success=False, message=f"Unknown action: {atype}"), -0.05

    def _handle_write(self, action: MigrationAction) -> Tuple[ExecutionResult, float]:
        """Append SQL to the migration buffer. +small reward for non-empty SQL."""
        if not action.sql or not action.sql.strip():
            return ExecutionResult(success=False, message="WRITE_MIGRATION requires non-empty sql"), -0.02

        # Basic syntax pre-check: try parsing without executing
        test_db = MigrationDB()
        test_db.init(self._db.snapshot_sql())
        test_result = test_db.execute_sql(action.sql)
        test_db.close()

        if not test_result.success:
            # Syntax error — warn agent but don't crash the episode
            return ExecutionResult(
                success=False,
                message=f"Syntax check failed: {test_result.message}. SQL not added to buffer.",
            ), -0.03

        self._migration_buffer += ("\n" if self._migration_buffer else "") + action.sql.strip()
        return ExecutionResult(
            success=True,
            message=f"SQL added to migration buffer ({len(self._migration_buffer)} chars total)",
        ), 0.02

    def _handle_execute(self, action: MigrationAction) -> Tuple[ExecutionResult, float]:
        """Execute the migration buffer (or inline SQL) against the live DB."""
        sql_to_run = action.sql or self._migration_buffer
        if not sql_to_run or not sql_to_run.strip():
            return ExecutionResult(success=False, message="Nothing to execute — buffer is empty"), -0.02

        result = self._db.execute_sql(sql_to_run)
        if result.success:
            self._migration_buffer = ""  # Clear buffer on success
            return result, 0.05          # Reward for successful execution
        else:
            return result, -0.05         # Penalty for execution failure

    def _handle_rollback(self) -> Tuple[ExecutionResult, float]:
        """
        Restore the database to the pre-migration snapshot.
        Penalised to discourage infinite rollback loops.
        """
        self._rollback_count += 1
        if self._rollback_count > 5:
            return ExecutionResult(
                success=False,
                message="Too many rollbacks (>5). Penalty applied.",
            ), -0.1

        # Restore from pre-episode snapshot
        self._db.close()
        self._db = MigrationDB()
        self._db.init(self._pre_snapshot)
        self._migration_buffer = ""
        penalty = -0.03 * self._rollback_count
        return ExecutionResult(
            success=True,
            message=f"Database rolled back to initial state (rollback #{self._rollback_count})",
        ), penalty

    def _handle_inspect(self) -> Tuple[ExecutionResult, float]:
        """Read the current schema. Free action — no reward/penalty."""
        schema = self._db.get_schema()
        schema_text = "; ".join(
            f"{t.name}({', '.join(c.name + ':' + c.type for c in t.columns)})"
            for t in schema
        )
        return ExecutionResult(
            success=True,
            message=f"Schema: {schema_text}",
            rows_affected=len(schema),
        ), 0.0

    def _handle_query(self, action: MigrationAction) -> Tuple[ExecutionResult, float]:
        """Run a SELECT query. Free action — used for verification."""
        if not action.sql or not action.sql.strip():
            return ExecutionResult(success=False, message="RUN_QUERY requires sql"), -0.01
        result = self._db.run_query(action.sql)
        return result, 0.0

    def _handle_submit(self) -> Tuple[ExecutionResult, float]:
        """
        Finalize the episode. Run the grader and return the full score.
        This is the only action that triggers the definitive grade.
        """
        self._done = True
        score, notes = self._task.grader(self._db, self._pre_snapshot)

        breakdown = RewardBreakdown(
            total=score,
            notes=notes,
        )

        # Overwrite cumulative reward with the authoritative grader score
        self._cumulative_reward = score

        return ExecutionResult(
            success=True,
            message=(
                f"Episode complete. Final score: {score:.4f}\n"
                + "\n".join(notes)
            ),
        ), score  # The step reward IS the final score

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _make_observation(
        self, last_result: Optional[ExecutionResult]
    ) -> MigrationObservation:
        schema = self._db.get_schema()
        hint = self._get_hint()
        return MigrationObservation(
            current_schema=schema,
            migration_spec=self._task.spec,
            requirements=self._task.requirements,
            migration_buffer=self._migration_buffer,
            execution_history=list(self._execution_history),
            last_result=last_result,
            step=self._step_count,
            max_steps=self._task.max_steps,
            partial_score=self._cumulative_reward,
            hint=hint,
        )

    def _get_hint(self) -> Optional[str]:
        """Provide a contextual hint if the agent appears stuck."""
        if self._step_count == 0:
            return None
        if self._rollback_count >= 3:
            return "Hint: Consider using INSPECT_SCHEMA to understand the current state before writing SQL."
        recent_failures = sum(
            1 for r in self._execution_history[-3:] if not r.success
        )
        if recent_failures >= 3:
            return "Hint: Multiple recent failures. Try INSPECT_SCHEMA or RUN_QUERY to verify current state."
        steps_remaining = self._task.max_steps - self._step_count
        if steps_remaining <= 5 and not self._done:
            return f"Hint: Only {steps_remaining} steps remaining. Consider SUBMIT to lock in partial credit."
        return None
