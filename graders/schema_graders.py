"""
Grader classes for the Schema Migration OpenEnv.

Each grader wraps the task-specific scoring function and exposes it
as a callable class with a score(observation) -> float interface.

Grader scores are always strictly between 0.001 and 0.999 — never
exactly 0.0 or 1.0 — satisfying the OpenEnv Phase 2 score range check.
"""

from __future__ import annotations

import sys
import os

# Ensure project root is on path when imported standalone
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from env.database import MigrationDB
from tasks.task_definitions import _grade_easy, _grade_medium, _grade_hard, TASKS


class _BaseGrader:
    """
    Callable grader base class compatible with the OpenEnv grader protocol.

    Interface:
        grader = AddColumnsGrader()
        score = grader(observation)  # returns float in (0, 1) exclusive
    """

    task_name: str
    _grade_fn = None

    def __call__(self, observation=None) -> float:
        """
        Run the grader against the current task state.

        When called from the OpenEnv framework, observation may contain
        the database state. Falls back to running the grader on a fresh
        initial DB (returns initial partial score) if no live DB is given.
        """
        # If observation has a live db reference, use it
        db = getattr(observation, "_db", None) or getattr(observation, "db", None)
        pre = getattr(observation, "_pre_snapshot", "") if db else ""

        if db is None:
            # Fallback: run grader on the seed state
            task = TASKS[self.task_name]
            db = MigrationDB()
            db.init(task.seed_sql)
            pre = db.snapshot_sql()
            score, _ = self._grade_fn(db, pre)
            db.close()
        else:
            score, _ = self._grade_fn(db, pre)

        return score

    def score(self, observation=None) -> float:
        return self(observation)

    def forward(self, action, observation) -> float:
        """OpenEnv Rubric-compatible forward method."""
        return self(observation)


class AddColumnsGrader(_BaseGrader):
    """
    Grader for the add_columns (easy) task.

    Checks that stock_quantity, category, and created_at columns were
    added to the products table with correct types, constraints, and
    default values, and that all 5 original rows are preserved.
    """
    task_name = "add_columns"
    _grade_fn = staticmethod(_grade_easy)


class NormalizeOrdersGrader(_BaseGrader):
    """
    Grader for the normalize_orders (medium) task.

    Checks that the denormalized orders table was split into customers,
    products, and orders tables with correct FK relationships, row counts,
    and no duplicate entries.
    """
    task_name = "normalize_orders"
    _grade_fn = staticmethod(_grade_medium)


class RefactorEmployeesGrader(_BaseGrader):
    """
    Grader for the refactor_employees (hard) task.

    Checks that employee_records was refactored into departments,
    job_titles, and employees tables, with a compatibility VIEW that
    passes all 6 test queries and maintains FK integrity.
    """
    task_name = "refactor_employees"
    _grade_fn = staticmethod(_grade_hard)


__all__ = ["AddColumnsGrader", "NormalizeOrdersGrader", "RefactorEmployeesGrader"]
