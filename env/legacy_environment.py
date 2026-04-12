"""
OpenEnv-compatible environment for database schema migration tasks.

This is the legacy runtime that owns the real SQLite migration workflow. The
validator-facing OpenEnv wrapper lives in server/environment.py.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from env.database import MigrationDB
from env.legacy_models import (
    ActionType,
    ExecutionResult,
    MigrationAction,
    MigrationObservation,
    MigrationReward,
    RewardBreakdown,
)
from tasks.task_definitions import TASKS, Task, build_seed_metrics


class SchemaMigrationEnv:
    ENV_NAME = "schema-migration-openenv"
    VERSION = "1.1.0"

    def __init__(self, task_name: str = "add_columns") -> None:
        if task_name not in TASKS:
            raise ValueError(f"Unknown task '{task_name}'. Available: {list(TASKS.keys())}")
        self._task: Task = TASKS[task_name]
        self._db = MigrationDB()
        self._step_count = 0
        self._done = False
        self._migration_buffer = ""
        self._execution_history: List[ExecutionResult] = []
        self._step_history: List[Dict[str, Any]] = []
        self._execution_logs: List[str] = []
        self._pre_snapshot = ""
        self._seed_metrics: Dict[str, Any] = {}
        self._cumulative_reward = 0.0
        self._rollback_count = 0
        self._last_obs: Optional[MigrationObservation] = None
        self._episode_start = 0.0
        self._snapshot_stack: List[str] = []
        self._syntax_failures = 0
        self._execution_failures = 0
        self._successful_executions = 0

    def reset(self) -> MigrationObservation:
        self._db.close()
        self._db = MigrationDB()
        self._db.init(self._task.seed_sql)
        self._pre_snapshot = self._db.snapshot_sql()
        self._seed_metrics = build_seed_metrics(self._pre_snapshot)
        self._step_count = 0
        self._done = False
        self._migration_buffer = ""
        self._execution_history = []
        self._step_history = []
        self._execution_logs = []
        self._cumulative_reward = 0.0
        self._rollback_count = 0
        self._episode_start = time.time()
        self._snapshot_stack = [self._pre_snapshot]
        self._syntax_failures = 0
        self._execution_failures = 0
        self._successful_executions = 0
        obs = self._make_observation(last_result=None)
        self._last_obs = obs
        return obs

    def step(self, action: MigrationAction) -> Tuple[MigrationObservation, MigrationReward, bool, Dict[str, Any]]:
        if self._done:
            obs = self._last_obs or self.reset()
            reward = MigrationReward(value=0.0, done=True, message="Episode already finished.")
            return obs, reward, True, {"error": "episode_done"}

        self._step_count += 1
        started = time.time()
        result, step_reward, breakdown = self._dispatch(action)
        self._execution_history.append(result)
        self._execution_logs.append(result.message)
        self._step_history.append(
            {
                "step": self._step_count,
                "action_type": str(action.action_type),
                "sql": action.sql,
                "success": result.success,
                "message": result.message,
                "elapsed_ms": round((time.time() - started) * 1000, 2),
            }
        )

        budget_exceeded = self._step_count >= self._task.max_steps and not self._done
        if budget_exceeded:
            step_reward -= 0.05
            self._done = True
            self._execution_logs.append("Step budget exceeded before submit")

        self._cumulative_reward += step_reward
        done = self._done

        obs = self._make_observation(last_result=result)
        obs.partial_score = max(0.0, min(1.0, self._cumulative_reward))
        self._last_obs = obs
        reward = MigrationReward(
            value=round(max(-1.0, min(1.0, step_reward)), 4),
            breakdown=breakdown,
            done=done,
            message=result.message,
        )
        info = {
            "step": self._step_count,
            "max_steps": self._task.max_steps,
            "action_type": str(action.action_type),
            "execution_success": result.success,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "rollback_count": self._rollback_count,
        }
        return obs, reward, done, info

    def state(self) -> Dict[str, Any]:
        return {
            "env_name": self.ENV_NAME,
            "version": self.VERSION,
            "task": self._task.name,
            "difficulty": self._task.difficulty,
            "step": self._step_count,
            "max_steps": self._task.max_steps,
            "done": self._done,
            "migration_buffer": self._migration_buffer,
            "schema": [table.model_dump() for table in self._db.get_schema()],
            "execution_history_length": len(self._execution_history),
            "step_history": self._step_history,
            "execution_logs": self._execution_logs,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "rollback_count": self._rollback_count,
            "successful_executions": self._successful_executions,
            "syntax_failures": self._syntax_failures,
            "execution_failures": self._execution_failures,
            "elapsed_seconds": round(time.time() - self._episode_start, 2),
        }

    def grade(self) -> Tuple[float, List[str]]:
        score, notes, _ = self._task.grader(self._db, self._pre_snapshot, self._seed_metrics)
        return score, notes

    def grade_detailed(self) -> Tuple[float, List[str], Dict[str, float]]:
        return self._task.grader(self._db, self._pre_snapshot, self._seed_metrics)

    def available_tasks(self) -> List[str]:
        return list(TASKS.keys())

    def _dispatch(self, action: MigrationAction) -> Tuple[ExecutionResult, float, Optional[RewardBreakdown]]:
        atype = ActionType(action.action_type)
        if atype == ActionType.WRITE_MIGRATION:
            return self._handle_write(action)
        if atype == ActionType.EXECUTE:
            return self._handle_execute(action)
        if atype == ActionType.ROLLBACK:
            return self._handle_rollback()
        if atype == ActionType.INSPECT_SCHEMA:
            return self._handle_inspect()
        if atype == ActionType.RUN_QUERY:
            return self._handle_query(action)
        if atype == ActionType.SUBMIT:
            return self._handle_submit()
        return ExecutionResult(success=False, message=f"Unknown action: {atype}"), -0.05, None

    def _handle_write(self, action: MigrationAction) -> Tuple[ExecutionResult, float, Optional[RewardBreakdown]]:
        if not action.sql or not action.sql.strip():
            self._syntax_failures += 1
            return ExecutionResult(success=False, message="write_migration requires non-empty sql"), -0.02, None

        valid, message = self._db.validate_sql(action.sql)
        if not valid:
            self._syntax_failures += 1
            return ExecutionResult(success=False, message=message), -0.04, None

        dry_run = self._db.execute_sql(action.sql, apply_changes=False)
        if not dry_run.success:
            self._syntax_failures += 1
            return ExecutionResult(success=False, message=f"Validation failed: {dry_run.message}"), -0.04, None

        self._migration_buffer += ("\n" if self._migration_buffer else "") + action.sql.strip()
        return ExecutionResult(success=True, message="SQL added to migration buffer"), 0.02, None

    def _handle_execute(self, action: MigrationAction) -> Tuple[ExecutionResult, float, Optional[RewardBreakdown]]:
        sql_to_run = action.sql or self._migration_buffer
        if not sql_to_run or not sql_to_run.strip():
            self._execution_failures += 1
            return ExecutionResult(success=False, message="Nothing to execute; migration buffer is empty"), -0.02, None

        before_snapshot = self._db.snapshot_sql()
        result = self._db.execute_sql(sql_to_run)
        if result.success:
            self._snapshot_stack.append(before_snapshot)
            self._migration_buffer = ""
            self._successful_executions += 1
            return result, 0.05, None

        self._execution_failures += 1
        return result, -0.05, None

    def _handle_rollback(self) -> Tuple[ExecutionResult, float, Optional[RewardBreakdown]]:
        self._rollback_count += 1
        if len(self._snapshot_stack) <= 1:
            return ExecutionResult(success=False, message="No executed migration available to rollback"), -0.03, None
        previous_snapshot = self._snapshot_stack.pop()
        self._db.init(previous_snapshot)
        self._migration_buffer = ""
        penalty = min(0.2, 0.03 * self._rollback_count)
        return ExecutionResult(success=True, message=f"Rolled back last migration (rollback #{self._rollback_count})"), -penalty, None

    def _handle_inspect(self) -> Tuple[ExecutionResult, float, Optional[RewardBreakdown]]:
        schema = self._db.get_schema()
        summary = "; ".join(
            f"{table.object_type} {table.name}({', '.join(col.name + ':' + col.type for col in table.columns)})"
            for table in schema
        )
        return ExecutionResult(success=True, message=f"Schema: {summary}", rows_affected=len(schema)), 0.0, None

    def _handle_query(self, action: MigrationAction) -> Tuple[ExecutionResult, float, Optional[RewardBreakdown]]:
        if not action.sql or not action.sql.strip():
            return ExecutionResult(success=False, message="run_query requires sql"), -0.01, None
        result = self._db.run_query(action.sql)
        return result, 0.0, None

    def _handle_submit(self) -> Tuple[ExecutionResult, float, Optional[RewardBreakdown]]:
        self._done = True
        score, notes, raw_breakdown = self.grade_detailed()
        efficiency_penalty = min(
            0.35,
            raw_breakdown.get("efficiency_penalty", 0.0)
            + 0.03 * self._rollback_count
            + 0.01 * max(0, self._step_count - max(1, self._task.max_steps // 2)),
        )
        raw_total = max(
            0.0,
            min(
                1.0,
                0.20 * raw_breakdown.get("syntax_score", 0.0)
                + 0.20 * raw_breakdown.get("execution_score", 0.0)
                + 0.35 * raw_breakdown.get("correctness_score", 0.0)
                + 0.25 * raw_breakdown.get("integrity_score", 0.0)
                - efficiency_penalty,
            ),
        )
        total = max(0.05, min(0.95, round(0.05 + raw_total * 0.90, 4)))
        breakdown = RewardBreakdown(
            syntax_score=raw_breakdown.get("syntax_score", 0.0),
            execution_score=raw_breakdown.get("execution_score", 0.0),
            correctness_score=raw_breakdown.get("correctness_score", 0.0),
            integrity_score=raw_breakdown.get("integrity_score", 0.0),
            efficiency_penalty=efficiency_penalty,
            total=total,
            notes=notes,
        )
        self._cumulative_reward = breakdown.total
        return (
            ExecutionResult(
                success=True,
                message="Episode complete. Final score: {:.4f}\n{}".format(breakdown.total, "\n".join(notes)),
                metadata={"notes": notes, "score": breakdown.total},
            ),
            breakdown.total,
            breakdown,
        )

    def _make_observation(self, last_result: Optional[ExecutionResult]) -> MigrationObservation:
        return MigrationObservation(
            current_schema=self._db.get_schema(),
            migration_spec=self._task.spec,
            requirements=self._task.requirements,
            migration_hints=self._task.hints,
            migration_buffer=self._migration_buffer,
            execution_history=list(self._execution_history),
            step_history=list(self._step_history),
            execution_logs=list(self._execution_logs[-20:]),
            last_result=last_result,
            step=self._step_count,
            max_steps=self._task.max_steps,
            partial_score=self._cumulative_reward,
            hint=self._get_hint(),
        )

    def _get_hint(self) -> Optional[str]:
        if self._step_count == 0:
            return None
        if self._rollback_count >= 2:
            return "Rollback restores only the last executed migration. Inspect the schema before retrying."
        recent_failures = sum(1 for result in self._execution_history[-3:] if not result.success)
        if recent_failures >= 2:
            return "Multiple recent failures detected. Use inspect_schema or run_query to verify the live state before executing more SQL."
        steps_remaining = self._task.max_steps - self._step_count
        if steps_remaining <= 4 and not self._done:
            return f"Only {steps_remaining} steps remain. Run your verification queries before submit."
        return None
