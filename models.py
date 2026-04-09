"""
Root-level models for Schema Migration OpenEnv.

MigrationAction, MigrationObservation, and MigrationState inherit from the
official openenv-core base classes for Phase 2 validator compliance.

Internal helper types (TableSchema, ExecutionResult, etc.) remain in
env/models.py and are imported here for convenience.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field

from openenv.core.env_server.types import Action, Observation, State

# Re-export internal helper types so server/environment.py can import from one place
from env.models import (
    ActionType,
    ExecutionResult,
    TableColumn,
    TableSchema,
)


class MigrationAction(Action):
    """Action for the Schema Migration environment.

    Inherits metadata: Dict from openenv Action base.
    extra='forbid' is inherited — only action_type and sql are accepted.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
        use_enum_values=True,   # store ActionType as its string value
    )

    action_type: ActionType = Field(..., description="The type of action to perform")
    sql: Optional[str] = Field(
        default=None,
        description="SQL statement for write_migration, run_query, or inline execute",
    )


class MigrationObservation(Observation):
    """Observation from the Schema Migration environment.

    Inherits from openenv Observation:
        done: bool = False
        reward: float | int | bool | None = None
        metadata: Dict[str, Any] = {}
    """

    current_schema: List[TableSchema] = Field(default_factory=list)
    migration_spec: str = ""
    requirements: List[str] = Field(default_factory=list)
    migration_buffer: str = ""
    execution_history: List[ExecutionResult] = Field(default_factory=list)
    last_result: Optional[ExecutionResult] = None
    step: int = 0
    max_steps: int = 20
    partial_score: float = 0.0
    hint: Optional[str] = None


class MigrationState(State):
    """State for the Schema Migration environment.

    Inherits from openenv State:
        episode_id: Optional[str] = None
        step_count: int = 0
    State has extra='allow' so additional fields are accepted.
    """

    task: str = "add_columns"
    difficulty: str = "easy"
    env_done: bool = False
    cumulative_reward: float = 0.0
    rollback_count: int = 0


__all__ = [
    "MigrationAction",
    "MigrationObservation",
    "MigrationState",
    # Re-exports from env.models
    "ActionType",
    "ExecutionResult",
    "TableColumn",
    "TableSchema",
]
