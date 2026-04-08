"""
SQLite-backed database engine for the Schema Migration environment.

All databases run fully in-memory — no files, no cleanup needed.
Each episode gets a fresh DB snapshot cloned from the task's seed schema.
"""

from __future__ import annotations

import sqlite3
import textwrap
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple

from env.models import ExecutionResult, TableColumn, TableSchema


class MigrationDB:
    """
    Wraps an in-memory SQLite connection with helpers for schema introspection,
    safe SQL execution, and state snapshots.
    """

    def __init__(self) -> None:
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def init(self, seed_sql: str) -> None:
        """
        Create a fresh in-memory DB and populate it with seed_sql.
        seed_sql should be CREATE TABLE + INSERT statements.
        """
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.row_factory = sqlite3.Row
        try:
            for stmt in self._split_statements(seed_sql):
                if stmt.strip():
                    self._conn.execute(stmt)
            self._conn.commit()
        except sqlite3.Error as exc:
            raise RuntimeError(f"Failed to initialise DB from seed SQL: {exc}") from exc

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def snapshot_sql(self) -> str:
        """
        Dump the entire DB as SQL (CREATE + INSERT) so we can restore it later.
        Used to save pre-migration state for rollback and correctness checks.
        """
        lines: List[str] = []
        assert self._conn
        for row in self._conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"
        ):
            lines.append(row["sql"] + ";")
            tname = row["name"]
            rows = self._conn.execute(f'SELECT * FROM "{tname}"').fetchall()
            for r in rows:
                vals = ", ".join(self._sql_literal(v) for v in r)
                lines.append(f'INSERT INTO "{tname}" VALUES ({vals});')
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute_sql(self, sql: str, readonly: bool = False) -> ExecutionResult:
        """
        Execute one or more semicolon-delimited SQL statements.
        Returns an ExecutionResult with success, message, rows_affected.
        """
        assert self._conn
        statements = self._split_statements(sql)
        if not statements:
            return ExecutionResult(success=False, message="Empty SQL — nothing to execute")

        total_affected = 0
        try:
            for stmt in statements:
                stmt = stmt.strip()
                if not stmt:
                    continue
                upper = stmt.upper().lstrip()

                # Guard: block dangerous operations in readonly mode
                if readonly:
                    for forbidden in ("DROP", "DELETE", "TRUNCATE", "UPDATE", "INSERT", "ALTER", "CREATE"):
                        if upper.startswith(forbidden):
                            return ExecutionResult(
                                success=False,
                                message=f"Read-only mode: {forbidden} statements are not allowed",
                            )

                cur = self._conn.execute(stmt)
                total_affected += cur.rowcount if cur.rowcount > 0 else 0

            self._conn.commit()
            return ExecutionResult(
                success=True,
                message="OK",
                rows_affected=total_affected,
            )
        except sqlite3.Error as exc:
            try:
                self._conn.rollback()
            except Exception:
                pass
            return ExecutionResult(success=False, message=str(exc), rows_affected=0)

    def run_query(self, sql: str, max_rows: int = 50) -> ExecutionResult:
        """
        Run a SELECT query and return results as list of dicts.
        Limited to max_rows to prevent runaway output.
        """
        assert self._conn
        sql = sql.strip()
        upper = sql.upper()
        if not (upper.startswith("SELECT") or upper.startswith("PRAGMA") or upper.startswith("WITH")):
            return ExecutionResult(
                success=False,
                message="run_query only accepts SELECT / PRAGMA / WITH statements",
            )
        try:
            cur = self._conn.execute(sql)
            rows = cur.fetchmany(max_rows)
            cols = [d[0] for d in cur.description] if cur.description else []
            result = [dict(zip(cols, row)) for row in rows]
            return ExecutionResult(
                success=True,
                message=f"{len(result)} row(s) returned",
                rows_affected=len(result),
                query_result=result,
            )
        except sqlite3.Error as exc:
            return ExecutionResult(success=False, message=str(exc))

    # ------------------------------------------------------------------
    # Schema introspection
    # ------------------------------------------------------------------

    def get_schema(self) -> List[TableSchema]:
        """Return the current schema as a list of TableSchema models."""
        assert self._conn
        tables: List[TableSchema] = []
        for row in self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ):
            tname = row[0]
            columns = self._get_columns(tname)
            count = self._conn.execute(f'SELECT COUNT(*) FROM "{tname}"').fetchone()[0]
            tables.append(TableSchema(name=tname, columns=columns, row_count=count))
        return tables

    def table_exists(self, name: str) -> bool:
        assert self._conn
        r = self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        return r is not None

    def column_exists(self, table: str, column: str) -> bool:
        if not self.table_exists(table):
            return False
        cols = self._get_columns(table)
        return any(c.name == column for c in cols)

    def get_row_count(self, table: str) -> int:
        assert self._conn
        if not self.table_exists(table):
            return -1
        return self._conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]

    def get_checksum(self, table: str) -> Optional[str]:
        """Simple row-level checksum for data integrity checks."""
        assert self._conn
        if not self.table_exists(table):
            return None
        try:
            rows = self._conn.execute(f'SELECT * FROM "{table}" ORDER BY rowid').fetchall()
            return str(hash(str([tuple(r) for r in rows])))
        except Exception:
            return None

    def query_returns(self, sql: str) -> Optional[List[Any]]:
        """Run a query and return the first column of all rows, or None on error."""
        assert self._conn
        try:
            rows = self._conn.execute(sql).fetchall()
            return [row[0] for row in rows]
        except Exception:
            return None

    def query_scalar(self, sql: str) -> Optional[Any]:
        """Run a query and return a single scalar value, or None on error."""
        assert self._conn
        try:
            row = self._conn.execute(sql).fetchone()
            return row[0] if row else None
        except Exception:
            return None

    def fk_violations(self) -> int:
        """Returns number of FK violations currently in the database."""
        assert self._conn
        try:
            rows = self._conn.execute("PRAGMA foreign_key_check").fetchall()
            return len(rows)
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_columns(self, table: str) -> List[TableColumn]:
        assert self._conn
        cols: List[TableColumn] = []
        fk_targets: Dict[str, str] = {}

        # FK info
        for fk in self._conn.execute(f'PRAGMA foreign_key_list("{table}")'):
            fk_targets[fk[3]] = f"{fk[2]}.{fk[4]}"  # from_col -> table.to_col

        for info in self._conn.execute(f'PRAGMA table_info("{table}")'):
            col = TableColumn(
                name=info[1],
                type=info[2] or "TEXT",
                nullable=not bool(info[3]),
                primary_key=bool(info[5]),
                foreign_key=fk_targets.get(info[1]),
            )
            cols.append(col)
        return cols

    @staticmethod
    def _split_statements(sql: str) -> List[str]:
        """Split SQL on semicolons while respecting string literals."""
        stmts: List[str] = []
        buf = ""
        in_str = False
        str_char = ""
        for ch in sql:
            if in_str:
                buf += ch
                if ch == str_char:
                    in_str = False
            elif ch in ("'", '"'):
                in_str = True
                str_char = ch
                buf += ch
            elif ch == ";":
                buf = buf.strip()
                if buf:
                    stmts.append(buf)
                buf = ""
            else:
                buf += ch
        buf = buf.strip()
        if buf:
            stmts.append(buf)
        return stmts

    @staticmethod
    def _sql_literal(value: Any) -> str:
        if value is None:
            return "NULL"
        if isinstance(value, (int, float)):
            return str(value)
        return "'" + str(value).replace("'", "''") + "'"
