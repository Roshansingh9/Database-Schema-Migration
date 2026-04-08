"""
Typed Pydantic models for the Schema Migration OpenEnv environment.
Observation, Action, and Reward — fully compliant with OpenEnv spec.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Action types
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    WRITE_MIGRATION = "write_migration"   # Write a SQL migration statement
    EXECUTE         = "execute"           # Execute the current migration buffer
    ROLLBACK        = "rollback"          # Roll back the last executed migration
    INSPECT_SCHEMA  = "inspect_schema"    # Read current schema (returns table list + columns)
    RUN_QUERY       = "run_query"         # Run a SELECT query to inspect data
    SUBMIT          = "submit"            # Finalize and score the episode


class MigrationAction(BaseModel):
    """
    The action an agent takes in one step.

    action_type : one of the ActionType enum values
    sql         : required for WRITE_MIGRATION, EXECUTE (if sql provided inline),
                  and RUN_QUERY
    """
    action_type: ActionType = Field(..., description="The type of action to perform")
    sql: Optional[str] = Field(
        default=None,
        description="SQL statement(s) for WRITE_MIGRATION or RUN_QUERY actions",
    )

    model_config = ConfigDict(use_enum_values=True)


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class TableColumn(BaseModel):
    name: str
    type: str
    nullable: bool
    primary_key: bool = False
    foreign_key: Optional[str] = Field(
        default=None,
        description="References 'table.column' if this is a FK column",
    )


class TableSchema(BaseModel):
    name: str
    columns: List[TableColumn]
    row_count: int = 0


class ExecutionResult(BaseModel):
    success: bool
    message: str
    rows_affected: int = 0
    query_result: Optional[List[Dict[str, Any]]] = None  # for RUN_QUERY


class MigrationObservation(BaseModel):
    """
    Everything the agent can see at each step.

    current_schema      : list of tables with columns as they exist NOW in the db
    migration_spec      : natural-language description of what the migration must achieve
    requirements        : structured list of specific requirements to satisfy
    migration_buffer    : SQL the agent has written but not yet executed
    execution_history   : log of all executions and their outcomes this episode
    last_result         : outcome of the most recent action
    step                : current step number (0-indexed)
    max_steps           : episode step budget
    partial_score       : running reward accumulated so far (0.0–1.0 scale)
    hint               : optional contextual hint (None unless agent is stuck)
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


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class RewardBreakdown(BaseModel):
    """Granular breakdown so the agent (and judges) can see exactly why it scored."""
    syntax_score: float = Field(0.0, ge=0.0, le=1.0)
    execution_score: float = Field(0.0, ge=0.0, le=1.0)
    correctness_score: float = Field(0.0, ge=0.0, le=1.0)
    integrity_score: float = Field(0.0, ge=0.0, le=1.0)
    efficiency_penalty: float = Field(0.0, ge=0.0, le=1.0)
    total: float = Field(0.0, ge=0.0, le=1.0)
    notes: List[str] = Field(default_factory=list)


class MigrationReward(BaseModel):
    """
    Reward returned from step().

    value     : scalar reward for THIS step (float in [-0.5, 0.5])
    breakdown : detailed scoring components (populated at SUBMIT time)
    done      : whether the episode is complete
    """
    value: float = Field(0.0, ge=-1.0, le=1.0)
    breakdown: Optional[RewardBreakdown] = None
    done: bool = False
    message: str = ""
