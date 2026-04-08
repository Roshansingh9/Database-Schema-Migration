"""
Grader classes for the Schema Migration OpenEnv.

Each class inherits from openenv.core.rubrics.base.Rubric and implements
forward(action, observation) -> float in the strict open interval (0, 1).

Phase 2 validation imports these via the openenv.yaml grader fields:
    grader: "graders.schema_graders:AddColumnsGrader"
"""

from __future__ import annotations

import os
import sys
from typing import Any, Optional

# Ensure project root is on path when imported standalone
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

try:
    from openenv.core.rubrics.base import Rubric as _RubricBase
except ImportError:
    # Fallback base if openenv not installed (e.g. during local dev without openenv)
    class _RubricBase:  # type: ignore[no-redef]
        def __init__(self):
            pass
        def forward(self, action: Any, observation: Any) -> float:
            raise NotImplementedError
        def __call__(self, action: Any, observation: Any) -> float:
            return self.forward(action, observation)

from env.database import MigrationDB
from tasks.task_definitions import _grade_easy, _grade_medium, _grade_hard, TASKS


def _score_from_seed(task_name: str, grade_fn) -> float:
    """Run grader on the task's seed state and return score (always in (0.001, 0.999))."""
    task = TASKS[task_name]
    db = MigrationDB()
    db.init(task.seed_sql)
    pre = db.snapshot_sql()
    score, _ = grade_fn(db, pre)
    db.close()
    return score


def _extract_score(observation: Any, task_name: str, grade_fn) -> float:
    """
    Extract grader score from an observation or environment state.

    Priority:
    1. Live DB attached to observation (_db attribute, in-process use)
    2. partial_score field on the observation dict/object
    3. Seed-state fallback (deterministic, always strictly in (0,1))
    """
    # In-process: observation has a live DB reference
    db = getattr(observation, "_db", None) or getattr(observation, "db", None)
    if db is not None:
        pre = getattr(observation, "_pre_snapshot", "")
        score, _ = grade_fn(db, pre)
        return score

    # HTTP observation dict or Pydantic model: read partial_score
    if isinstance(observation, dict):
        ps = observation.get("partial_score")
        if ps is not None and 0 < float(ps) < 1:
            return float(ps)
    elif hasattr(observation, "partial_score"):
        ps = observation.partial_score
        if ps is not None and 0 < float(ps) < 1:
            return float(ps)

    # Deterministic fallback: score the seed state
    return _score_from_seed(task_name, grade_fn)


class _BaseGrader(_RubricBase):
    """
    Base grader — proper Rubric subclass for OpenEnv Phase 2 compatibility.

    Subclasses set:
        task_name: str
        _grade_fn: callable(db, pre_snapshot) -> (float, list)
    """

    task_name: str = ""
    _grade_fn = None

    def __init__(self):
        super().__init__()

    def forward(self, action: Any, observation: Any) -> float:
        """
        Compute score for this task given the current observation.
        Returns a float strictly in (0.001, 0.999).
        """
        return _extract_score(observation, self.task_name, self._grade_fn)

    def __call__(self, action: Any = None, observation: Any = None) -> float:
        return self.forward(action, observation)

    def score(self, observation: Any = None) -> float:
        return self.forward(None, observation)


class AddColumnsGrader(_BaseGrader):
    """
    Grader for the add_columns (easy) task.

    Checks that stock_quantity, category, and created_at columns were added
    to the products table with correct types, constraints, and default values,
    and that all 5 original rows are preserved.

    Score range: (0.001, 0.999) — strictly between 0 and 1.
    """
    task_name = "add_columns"
    _grade_fn = staticmethod(_grade_easy)


class NormalizeOrdersGrader(_BaseGrader):
    """
    Grader for the normalize_orders (medium) task.

    Checks that the denormalized orders table was split into customers,
    products, and orders with correct FK relationships and row counts.

    Score range: (0.001, 0.999) — strictly between 0 and 1.
    """
    task_name = "normalize_orders"
    _grade_fn = staticmethod(_grade_medium)


class RefactorEmployeesGrader(_BaseGrader):
    """
    Grader for the refactor_employees (hard) task.

    Checks that employee_records was refactored into departments, job_titles,
    and employees tables, with a compatibility VIEW passing all 6 test queries.

    Score range: (0.001, 0.999) — strictly between 0 and 1.
    """
    task_name = "refactor_employees"
    _grade_fn = staticmethod(_grade_hard)


__all__ = ["AddColumnsGrader", "NormalizeOrdersGrader", "RefactorEmployeesGrader"]
