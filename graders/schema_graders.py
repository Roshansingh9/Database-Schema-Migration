"""
OpenEnv rubric adapters for the Database Schema Migration Agent.
"""

from __future__ import annotations

import os
import sys
from typing import Any

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

try:
    from openenv.core.rubrics.base import Rubric as _RubricBase
except ImportError:
    class _RubricBase:
        def forward(self, action: Any, observation: Any) -> float:
            raise NotImplementedError
        def __call__(self, action: Any, observation: Any) -> float:
            return self.forward(action, observation)

from env.database import MigrationDB
from tasks.task_definitions import TASKS, build_seed_metrics


def _score_from_seed(task_name: str) -> float:
    task = TASKS[task_name]
    db = MigrationDB()
    db.init(task.seed_sql)
    pre = db.snapshot_sql()
    metrics = build_seed_metrics(pre)
    score, _, _ = task.grader(db, pre, metrics)
    db.close()
    return score


def _extract_score(observation: Any, task_name: str) -> float:
    db = getattr(observation, "_db", None) or getattr(observation, "db", None)
    if db is not None:
        pre = getattr(observation, "_pre_snapshot", "")
        metrics = build_seed_metrics(pre) if pre else build_seed_metrics(TASKS[task_name].seed_sql)
        score, _, _ = TASKS[task_name].grader(db, pre or TASKS[task_name].seed_sql, metrics)
        return score
    if isinstance(observation, dict):
        ps = observation.get("partial_score")
        if ps is not None and 0.0 < float(ps) < 1.0:
            return float(ps)
    elif hasattr(observation, "partial_score"):
        ps = float(observation.partial_score)
        if 0.0 < ps < 1.0:
            return ps
    return _score_from_seed(task_name)


class _BaseGrader(_RubricBase):
    task_name = ""

    def forward(self, action: Any, observation: Any) -> float:
        return _extract_score(observation, self.task_name)


class AddColumnsGrader(_BaseGrader):
    task_name = "add_columns"


class NormalizeOrdersGrader(_BaseGrader):
    task_name = "normalize_orders"


class RefactorEmployeesGrader(_BaseGrader):
    task_name = "refactor_employees"


__all__ = ["AddColumnsGrader", "NormalizeOrdersGrader", "RefactorEmployeesGrader"]
