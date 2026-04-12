"""
Validator-facing OpenEnv models for schema-migration-openenv.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

try:
    from openenv.core.env_server import Action, Observation, State
except ImportError:
    class Action(BaseModel):
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class Observation(BaseModel):
        done: bool = False
        reward: Optional[float] = None
        metadata: Dict[str, Any] = Field(default_factory=dict)

    class State(BaseModel):
        episode_id: Optional[str] = None
        step_count: int = 0

from env.legacy_models import ActionType, ExecutionResult, TableColumn, TableSchema


class MigrationAction(Action):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
        use_enum_values=True,
    )

    action_type: ActionType = Field(..., description="The type of action to perform")
    sql: Optional[str] = Field(default=None, description="SQL for write_migration, execute, or run_query")


class MigrationObservation(Observation):
    current_schema: List[TableSchema] = Field(default_factory=list)
    migration_spec: str = ""
    requirements: List[str] = Field(default_factory=list)
    migration_hints: Dict[str, Any] = Field(default_factory=dict)
    migration_buffer: str = ""
    execution_history: List[ExecutionResult] = Field(default_factory=list)
    step_history: List[Dict[str, Any]] = Field(default_factory=list)
    execution_logs: List[str] = Field(default_factory=list)
    last_result: Optional[ExecutionResult] = None
    step: int = 0
    max_steps: int = 20
    partial_score: float = 0.0
    hint: Optional[str] = None


class MigrationState(State):
    task: str = "add_columns"
    difficulty: str = "easy"
    env_done: bool = False
    cumulative_reward: float = 0.0
    rollback_count: int = 0
    successful_executions: int = 0
    syntax_failures: int = 0
    execution_failures: int = 0


__all__ = [
    "MigrationAction",
    "MigrationObservation",
    "MigrationState",
    "ActionType",
    "ExecutionResult",
    "TableColumn",
    "TableSchema",
]
