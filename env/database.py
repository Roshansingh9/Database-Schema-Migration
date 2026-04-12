"""
SQLite-backed execution engine for the Schema Migration environment.

The live database stays in-memory, but migration execution happens inside an
isolated subprocess seeded from a SQL snapshot. This gives us real execution,
strong rollback semantics, and protection against runaway queries.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from env.models import ExecutionResult, TableColumn, TableSchema

MAX_QUERY_ROWS = 50
DEFAULT_TIMEOUT_SECONDS = 3.0

_FORBIDDEN_SQL_PREFIXES = (
    "ATTACH",
    "DETACH",
    "VACUUM",
    "REINDEX",
    "ANALYZE",
    "LOAD_EXTENSION",
    "BEGIN",
    "COMMIT",
    "ROLLBACK",
    "SAVEPOINT",
    "RELEASE",
)
_FORBIDDEN_SQL_SUBSTRINGS = (
    "DROP DATABASE",
    "PRAGMA WRITABLE_SCHEMA",
    "LOAD_EXTENSION",
)
_READ_ONLY_PREFIXES = ("SELECT", "WITH", "PRAGMA")

_WORKER_SCRIPT = r'''
import json
import sqlite3
import sys


def sql_literal(value):
    if value is None:
        return "NULL"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def snapshot_conn(conn):
    lines = []
    objects = conn.execute(
        """
        SELECT name, type, sql
        FROM sqlite_master
        WHERE type IN ('table', 'view', 'index', 'trigger')
          AND name NOT LIKE 'sqlite_%'
          AND sql IS NOT NULL
        ORDER BY
          CASE type
            WHEN 'table' THEN 1
            WHEN 'view' THEN 2
            WHEN 'index' THEN 3
            WHEN 'trigger' THEN 4
          END,
          name
        """
    ).fetchall()
    table_names = [row[0] for row in objects if row[1] == 'table']
    for _, _, sql in objects:
        lines.append(sql + ';')
    for table in table_names:
        rows = conn.execute(f'SELECT * FROM "{table}"').fetchall()
        for row in rows:
            values = ', '.join(sql_literal(value) for value in row)
            lines.append(f'INSERT INTO "{table}" VALUES ({values});')
    return '\n'.join(lines)


payload = json.loads(sys.stdin.read())
conn = sqlite3.connect(':memory:')
conn.row_factory = sqlite3.Row
conn.execute('PRAGMA foreign_keys = ON')
try:
    conn.executescript(payload['snapshot_sql'])
    conn.commit()

    total_affected = 0
    query_result = None
    message = 'OK'
    if payload['readonly']:
        sql = payload['statements'][0].strip()
        cur = conn.execute(sql)
        rows = cur.fetchmany(payload['max_rows'])
        cols = [d[0] for d in cur.description] if cur.description else []
        query_result = [dict(zip(cols, row)) for row in rows]
        message = f"{len(query_result)} row(s) returned"
    else:
        script = ''.join(stmt.rstrip() + ';\n' for stmt in payload['statements'] if stmt.strip())
        before = conn.total_changes
        conn.executescript(script)
        conn.commit()
        total_affected = max(0, conn.total_changes - before)

    result = {
        'success': True,
        'message': message,
        'rows_affected': len(query_result or []) if payload['readonly'] else total_affected,
        'query_result': query_result,
    }
    if not payload['readonly']:
        result['snapshot_sql'] = snapshot_conn(conn)
except sqlite3.Error as exc:
    try:
        conn.rollback()
    except sqlite3.Error:
        pass
    result = {'success': False, 'message': str(exc), 'rows_affected': 0}
finally:
    conn.close()

sys.stdout.write(json.dumps(result))
'''


class MigrationDB:
    """Wrap an in-memory SQLite connection with helpers for safe execution."""

    def __init__(self, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        self._conn: Optional[sqlite3.Connection] = None
        self._timeout_seconds = timeout_seconds

    def init(self, seed_sql: str) -> None:
        self.close()
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = OFF")
        try:
            for stmt in self._split_statements(seed_sql):
                if stmt.strip():
                    self._conn.execute(stmt)
            self._conn.commit()
            self._conn.execute("PRAGMA foreign_keys = ON")
        except sqlite3.Error as exc:
            raise RuntimeError(f"Failed to initialise DB from seed SQL: {exc}") from exc

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def snapshot_sql(self) -> str:
        assert self._conn
        return self._snapshot_conn(self._conn)

    def execute_sql(
        self,
        sql: str,
        readonly: bool = False,
        timeout_seconds: Optional[float] = None,
        apply_changes: bool = True,
        max_rows: int = MAX_QUERY_ROWS,
    ) -> ExecutionResult:
        statements = self._split_statements(sql)
        if not statements:
            return ExecutionResult(success=False, message="Empty SQL: nothing to execute")

        valid, message = self.validate_sql(sql, readonly=readonly)
        if not valid:
            return ExecutionResult(success=False, message=message, rows_affected=0)

        result = self._run_in_subprocess(
            snapshot_sql=self.snapshot_sql(),
            statements=statements,
            readonly=readonly,
            timeout_seconds=timeout_seconds or self._timeout_seconds,
            max_rows=max_rows,
        )
        if result.success and not readonly and apply_changes:
            self.init(result.metadata["snapshot_sql"])
        return result

    def run_query(self, sql: str, max_rows: int = MAX_QUERY_ROWS) -> ExecutionResult:
        return self.execute_sql(sql, readonly=True, max_rows=max_rows)

    def validate_sql(self, sql: str, readonly: bool = False) -> Tuple[bool, str]:
        statements = self._split_statements(sql)
        if not statements:
            return False, "Empty SQL: nothing to validate"
        for stmt in statements:
            cleaned = stmt.strip()
            upper = cleaned.upper().lstrip()
            if any(token in upper for token in _FORBIDDEN_SQL_SUBSTRINGS):
                return False, f"Blocked unsafe SQL statement: {cleaned.split()[0]}"
            if upper.startswith(_FORBIDDEN_SQL_PREFIXES):
                return False, f"Blocked unsafe SQL statement: {cleaned.split()[0]}"
            if readonly and not upper.startswith(_READ_ONLY_PREFIXES):
                return False, "Read-only queries must start with SELECT, WITH, or PRAGMA"
        return True, "SQL accepted"

    def get_schema(self) -> List[TableSchema]:
        assert self._conn
        objects: List[TableSchema] = []
        for row in self._conn.execute(
            """
            SELECT name, type
            FROM sqlite_master
            WHERE type IN ('table', 'view') AND name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """
        ):
            name = row["name"]
            objects.append(
                TableSchema(
                    name=name,
                    columns=self._get_columns(name),
                    row_count=self._safe_count(name),
                    object_type=row["type"],
                    definition=self.get_object_definition(name),
                )
            )
        return objects

    def get_tables(self) -> List[str]:
        assert self._conn
        rows = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        return [row[0] for row in rows]

    def get_views(self) -> List[str]:
        assert self._conn
        rows = self._conn.execute("SELECT name FROM sqlite_master WHERE type='view' ORDER BY name").fetchall()
        return [row[0] for row in rows]

    def get_object_definition(self, name: str) -> Optional[str]:
        assert self._conn
        row = self._conn.execute("SELECT sql FROM sqlite_master WHERE name=?", (name,)).fetchone()
        return row[0] if row else None

    def table_exists(self, name: str) -> bool:
        assert self._conn
        return self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        ).fetchone() is not None

    def view_exists(self, name: str) -> bool:
        assert self._conn
        return self._conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='view' AND name=?",
            (name,),
        ).fetchone() is not None

    def column_exists(self, table: str, column: str) -> bool:
        return any(col.name == column for col in self._get_columns(table))

    def get_row_count(self, table: str) -> int:
        assert self._conn
        if not self.table_exists(table):
            return -1
        return self._conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]

    def get_checksum(self, table: str) -> Optional[str]:
        assert self._conn
        if not self.table_exists(table):
            return None
        try:
            rows = self._conn.execute(f'SELECT * FROM "{table}" ORDER BY rowid').fetchall()
            return self._checksum_rows(rows)
        except sqlite3.Error:
            return None

    def snapshot_table_metrics(self) -> Dict[str, Dict[str, Any]]:
        return {
            table: {"row_count": self.get_row_count(table), "checksum": self.get_checksum(table)}
            for table in self.get_tables()
        }

    def query_returns(self, sql: str) -> Optional[List[Any]]:
        result = self.run_query(sql)
        if not result.success or not result.query_result:
            return None
        key = next(iter(result.query_result[0]), None)
        return [] if key is None else [row[key] for row in result.query_result]

    def query_scalar(self, sql: str) -> Optional[Any]:
        result = self.run_query(sql)
        if not result.success or not result.query_result:
            return None
        row = result.query_result[0]
        return row[next(iter(row))] if row else None

    def fk_violations(self) -> int:
        assert self._conn
        try:
            return len(self._conn.execute("PRAGMA foreign_key_check").fetchall())
        except sqlite3.Error:
            return 0

    def schema_diff(self, expected_objects: Sequence[str]) -> Dict[str, List[str]]:
        current = {schema.name for schema in self.get_schema()}
        expected = set(expected_objects)
        return {"missing": sorted(expected - current), "unexpected": sorted(current - expected)}

    def compare_query_results(self, left_sql: str, right_sql: str) -> Dict[str, Any]:
        left = self.run_query(left_sql)
        right = self.run_query(right_sql)
        return {
            "match": left.success and right.success and left.query_result == right.query_result,
            "left": left.query_result,
            "right": right.query_result,
            "left_message": left.message,
            "right_message": right.message,
        }

    def _safe_count(self, name: str) -> int:
        assert self._conn
        try:
            return self._conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
        except sqlite3.Error:
            return -1

    def _get_columns(self, table: str) -> List[TableColumn]:
        assert self._conn
        cols: List[TableColumn] = []
        fk_targets: Dict[str, str] = {}
        try:
            for fk in self._conn.execute(f'PRAGMA foreign_key_list("{table}")'):
                fk_targets[fk[3]] = f"{fk[2]}.{fk[4]}"
            for info in self._conn.execute(f'PRAGMA table_info("{table}")'):
                cols.append(
                    TableColumn(
                        name=info[1],
                        type=info[2] or "TEXT",
                        nullable=not bool(info[3]),
                        primary_key=bool(info[5]),
                        foreign_key=fk_targets.get(info[1]),
                        default=info[4],
                    )
                )
        except sqlite3.Error:
            return []
        return cols

    def _run_in_subprocess(
        self,
        snapshot_sql: str,
        statements: Sequence[str],
        readonly: bool,
        timeout_seconds: float,
        max_rows: int,
    ) -> ExecutionResult:
        payload = {
            "snapshot_sql": snapshot_sql,
            "statements": list(statements),
            "readonly": readonly,
            "max_rows": max_rows,
        }
        try:
            completed = subprocess.run(
                [sys.executable, "-c", _WORKER_SCRIPT],
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(success=False, message=f"SQL execution timed out after {timeout_seconds:.1f}s", rows_affected=0)

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            return ExecutionResult(success=False, message=stderr or "Migration subprocess failed", rows_affected=0)

        try:
            payload = json.loads(completed.stdout or "{}")
        except json.JSONDecodeError:
            return ExecutionResult(success=False, message="Migration subprocess returned invalid JSON", rows_affected=0)

        metadata = {}
        if "snapshot_sql" in payload:
            metadata["snapshot_sql"] = payload.pop("snapshot_sql")
        return ExecutionResult(
            success=payload.get("success", False),
            message=payload.get("message", "Unknown execution result"),
            rows_affected=payload.get("rows_affected", 0),
            query_result=payload.get("query_result"),
            metadata=metadata,
        )

    @staticmethod
    def _snapshot_conn(conn: sqlite3.Connection) -> str:
        lines: List[str] = []
        objects = conn.execute(
            """
            SELECT name, type, sql
            FROM sqlite_master
            WHERE type IN ('table', 'view', 'index', 'trigger')
              AND name NOT LIKE 'sqlite_%'
              AND sql IS NOT NULL
            ORDER BY
              CASE type
                WHEN 'table' THEN 1
                WHEN 'view' THEN 2
                WHEN 'index' THEN 3
                WHEN 'trigger' THEN 4
              END,
              name
            """
        ).fetchall()
        table_names = [row["name"] for row in objects if row["type"] == "table"]
        for row in objects:
            lines.append(row["sql"] + ";")
        for table in table_names:
            rows = conn.execute(f'SELECT * FROM "{table}"').fetchall()
            for row in rows:
                values = ", ".join(MigrationDB._sql_literal(value) for value in row)
                lines.append(f'INSERT INTO "{table}" VALUES ({values});')
        return "\n".join(lines)

    @staticmethod
    def _checksum_rows(rows: Sequence[sqlite3.Row]) -> str:
        payload = json.dumps([list(row) for row in rows], sort_keys=True, default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _split_statements(sql: str) -> List[str]:
        stmts: List[str] = []
        buf = ""
        in_str = False
        str_char = ""
        in_line_comment = False
        in_block_comment = False
        i = 0
        while i < len(sql):
            ch = sql[i]
            nxt = sql[i + 1] if i + 1 < len(sql) else ""
            if in_line_comment:
                if ch == "\n":
                    in_line_comment = False
                    buf += ch
            elif in_block_comment:
                if ch == "*" and nxt == "/":
                    in_block_comment = False
                    i += 1
            elif in_str:
                buf += ch
                if ch == str_char:
                    if i + 1 < len(sql) and sql[i + 1] == str_char:
                        buf += sql[i + 1]
                        i += 1
                    else:
                        in_str = False
            elif ch == "-" and nxt == "-":
                in_line_comment = True
                i += 1
            elif ch == "/" and nxt == "*":
                in_block_comment = True
                i += 1
            elif ch in ("'", '"'):
                in_str = True
                str_char = ch
                buf += ch
            elif ch == ";":
                chunk = buf.strip()
                if chunk:
                    stmts.append(chunk)
                buf = ""
            else:
                buf += ch
            i += 1
        chunk = buf.strip()
        if chunk:
            stmts.append(chunk)
        return stmts

    @staticmethod
    def _sql_literal(value: Any) -> str:
        if value is None:
            return "NULL"
        if isinstance(value, (int, float)):
            return str(value)
        return "'" + str(value).replace("'", "''") + "'"


