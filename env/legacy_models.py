"""
Legacy internal models for the schema migration runtime.

These models are intentionally separate from the root-level OpenEnv models so
we can preserve the original environment logic while exposing the validator-
friendly wrappers in the project root.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ActionType(str, Enum):
    WRITE_MIGRATION = "write_migration"
    EXECUTE = "execute"
    ROLLBACK = "rollback"
    INSPECT_SCHEMA = "inspect_schema"
    RUN_QUERY = "run_query"
    SUBMIT = "submit"


class MigrationAction(BaseModel):
    action_type: ActionType = Field(..., description="The type of action to perform")
    sql: Optional[str] = Field(
        default=None,
        description="SQL statement(s) for write_migration, execute, or run_query actions",
    )

    model_config = ConfigDict(use_enum_values=True)


class TableColumn(BaseModel):
    name: str
    type: str
    nullable: bool
    primary_key: bool = False
    foreign_key: Optional[str] = Field(default=None, description="References 'table.column'")
    default: Optional[str] = None


class TableSchema(BaseModel):
    name: str
    columns: List[TableColumn]
    row_count: int = 0
    object_type: str = "table"
    definition: Optional[str] = None


class ExecutionResult(BaseModel):
    success: bool
    message: str
    rows_affected: int = 0
    query_result: Optional[List[Dict[str, Any]]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MigrationObservation(BaseModel):
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


class RewardBreakdown(BaseModel):
    syntax_score: float = Field(0.0, ge=0.0, le=1.0)
    execution_score: float = Field(0.0, ge=0.0, le=1.0)
    correctness_score: float = Field(0.0, ge=0.0, le=1.0)
    integrity_score: float = Field(0.0, ge=0.0, le=1.0)
    efficiency_penalty: float = Field(0.0, ge=0.0, le=1.0)
    total: float = Field(0.0, ge=0.0, le=1.0)
    notes: List[str] = Field(default_factory=list)


class MigrationReward(BaseModel):
    value: float = Field(0.0, ge=-1.0, le=1.0)
    breakdown: Optional[RewardBreakdown] = None
    done: bool = False
    message: str = ""
