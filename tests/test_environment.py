"""
tests/test_environment.py

Validates the full environment lifecycle, all three task graders,
reward signals, and edge cases.

Run with: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from env.database import MigrationDB
from env.environment import SchemaMigrationEnv
from env.models import ActionType, MigrationAction


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def easy_env():
    env = SchemaMigrationEnv(task_name="add_columns")
    env.reset()
    return env


@pytest.fixture
def medium_env():
    env = SchemaMigrationEnv(task_name="normalize_orders")
    env.reset()
    return env


@pytest.fixture
def hard_env():
    env = SchemaMigrationEnv(task_name="refactor_employees")
    env.reset()
    return env


# ---------------------------------------------------------------------------
# Environment lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:

    def test_reset_returns_observation(self, easy_env):
        obs = easy_env.reset()
        assert obs.current_schema is not None
        assert len(obs.current_schema) > 0
        assert obs.migration_spec != ""
        assert obs.step == 0

    def test_state_returns_dict(self, easy_env):
        s = easy_env.state()
        assert s["task"] == "add_columns"
        assert s["step"] == 0
        assert s["done"] is False

    def test_step_increments_counter(self, easy_env):
        action = MigrationAction(action_type=ActionType.INSPECT_SCHEMA)
        obs, reward, done, info = easy_env.step(action)
        assert obs.step == 1
        assert info["step"] == 1
        assert not done

    def test_submit_ends_episode(self, easy_env):
        action = MigrationAction(action_type=ActionType.SUBMIT)
        obs, reward, done, info = easy_env.step(action)
        assert done is True
        assert easy_env._done is True

    def test_step_after_done_returns_done(self, easy_env):
        easy_env.step(MigrationAction(action_type=ActionType.SUBMIT))
        _, _, done, _ = easy_env.step(MigrationAction(action_type=ActionType.INSPECT_SCHEMA))
        assert done is True

    def test_reset_clears_state(self, easy_env):
        easy_env.step(MigrationAction(action_type=ActionType.SUBMIT))
        easy_env.reset()
        assert easy_env._done is False
        assert easy_env._step_count == 0


# ---------------------------------------------------------------------------
# Database operations
# ---------------------------------------------------------------------------

class TestDatabase:

    def test_initial_schema_has_products(self, easy_env):
        schema = {t.name: t for t in easy_env._db.get_schema()}
        assert "products" in schema
        assert schema["products"].row_count == 5

    def test_execute_adds_column(self, easy_env):
        sql = "ALTER TABLE products ADD COLUMN stock_quantity INTEGER NOT NULL DEFAULT 0"
        result = easy_env._db.execute_sql(sql)
        assert result.success, result.message
        assert easy_env._db.column_exists("products", "stock_quantity")

    def test_rollback_restores_state(self, easy_env):
        easy_env._db.execute_sql(
            "ALTER TABLE products ADD COLUMN temp_col INTEGER"
        )
        assert easy_env._db.column_exists("products", "temp_col")
        action = MigrationAction(action_type=ActionType.ROLLBACK)
        easy_env.step(action)
        assert not easy_env._db.column_exists("products", "temp_col")

    def test_run_query_returns_data(self, easy_env):
        result = easy_env._db.run_query("SELECT COUNT(*) as cnt FROM products")
        assert result.success
        assert result.query_result is not None
        assert result.query_result[0]["cnt"] == 5

    def test_fk_violations_start_at_zero(self, medium_env):
        # Fresh DB should have no FK violations
        violations = medium_env._db.fk_violations()
        assert violations == 0


# ---------------------------------------------------------------------------
# Reward signals
# ---------------------------------------------------------------------------

class TestRewards:

    def test_inspect_schema_zero_reward(self, easy_env):
        _, reward, _, _ = easy_env.step(
            MigrationAction(action_type=ActionType.INSPECT_SCHEMA)
        )
        assert reward.value == 0.0

    def test_write_valid_sql_positive_reward(self, easy_env):
        _, reward, _, _ = easy_env.step(
            MigrationAction(
                action_type=ActionType.WRITE_MIGRATION,
                sql="ALTER TABLE products ADD COLUMN stock_quantity INTEGER NOT NULL DEFAULT 0",
            )
        )
        assert reward.value > 0

    def test_write_invalid_sql_negative_reward(self, easy_env):
        _, reward, _, _ = easy_env.step(
            MigrationAction(
                action_type=ActionType.WRITE_MIGRATION,
                sql="NOTVALIDSQL $$$",
            )
        )
        assert reward.value < 0

    def test_execute_success_positive_reward(self, easy_env):
        easy_env.step(MigrationAction(
            action_type=ActionType.WRITE_MIGRATION,
            sql="ALTER TABLE products ADD COLUMN stock_quantity INTEGER NOT NULL DEFAULT 0",
        ))
        _, reward, _, _ = easy_env.step(
            MigrationAction(action_type=ActionType.EXECUTE)
        )
        assert reward.value > 0

    def test_empty_execute_negative_reward(self, easy_env):
        # Empty buffer — nothing to execute
        _, reward, _, _ = easy_env.step(
            MigrationAction(action_type=ActionType.EXECUTE)
        )
        assert reward.value < 0

    def test_rollback_penalty_increases(self, easy_env):
        penalties = []
        for _ in range(3):
            _, reward, _, _ = easy_env.step(
                MigrationAction(action_type=ActionType.ROLLBACK)
            )
            penalties.append(reward.value)
        # Each successive rollback should be more penalized
        assert penalties[0] >= penalties[1] >= penalties[2]


# ---------------------------------------------------------------------------
# Grader: Easy task
# ---------------------------------------------------------------------------

class TestGraderEasy:

    def test_zero_score_without_migration(self, easy_env):
        score, notes = easy_env.grade()
        # No migration done — should score partially (original cols exist, rows intact)
        # but missing the 3 new columns
        assert 0.0 <= score <= 1.0

    def test_full_score_with_correct_migration(self):
        env = SchemaMigrationEnv(task_name="add_columns")
        env.reset()

        migrations = [
            "ALTER TABLE products ADD COLUMN stock_quantity INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE products ADD COLUMN category TEXT",
            "ALTER TABLE products ADD COLUMN created_at TEXT NOT NULL DEFAULT '2024-01-01'",
        ]
        for sql in migrations:
            env._db.execute_sql(sql)

        score, notes = env.grade()
        assert score >= 0.85, f"Expected >=0.85, got {score}. Notes: {notes}"

    def test_score_in_valid_range(self, easy_env):
        score, _ = easy_env.grade()
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Grader: Medium task
# ---------------------------------------------------------------------------

class TestGraderMedium:

    def test_zero_score_no_migration(self, medium_env):
        score, _ = medium_env.grade()
        # Original table exists but no customers/products tables → low score
        assert score < 0.3

    def test_partial_score_with_tables_only(self, medium_env):
        medium_env._db.execute_sql("""
            CREATE TABLE customers (
                customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                city TEXT
            )
        """)
        score, _ = medium_env.grade()
        assert score > 0.0

    def test_full_score_correct_normalization(self):
        env = SchemaMigrationEnv(task_name="normalize_orders")
        env.reset()

        # Full normalization
        env._db.execute_sql("""
            CREATE TABLE customers (
                customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                city TEXT
            );
            INSERT INTO customers (name, email, city) VALUES
                ('Alice Smith',  'alice@example.com', 'Mumbai'),
                ('Bob Jones',    'bob@example.com',   'Delhi'),
                ('Carol White',  'carol@example.com', 'Bangalore');

            CREATE TABLE products (
                product_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                price REAL NOT NULL
            );
            INSERT INTO products (name, price) VALUES
                ('Laptop',   999.99),
                ('Mouse',     29.99),
                ('Keyboard',  79.99),
                ('Monitor',  349.99),
                ('Headset',   89.99);

            CREATE TABLE new_orders (
                order_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
                product_id  INTEGER NOT NULL REFERENCES products(product_id),
                quantity    INTEGER NOT NULL DEFAULT 1,
                order_date  TEXT NOT NULL
            );
            INSERT INTO new_orders (customer_id, product_id, quantity, order_date)
            SELECT c.customer_id, p.product_id, o.quantity, o.order_date
            FROM orders o
            JOIN customers c ON c.email = o.customer_email
            JOIN products  p ON p.name  = o.product_name;

            DROP TABLE orders;
            ALTER TABLE new_orders RENAME TO orders;
        """)

        score, notes = env.grade()
        assert score >= 0.85, f"Expected >=0.85, got {score}. Notes: {notes}"


# ---------------------------------------------------------------------------
# Grader: Hard task
# ---------------------------------------------------------------------------

class TestGraderHard:

    def test_zero_score_no_migration(self, hard_env):
        score, _ = hard_env.grade()
        # employee_records exists as a table, not a view → low score
        assert score < 0.3

    def test_score_range_valid(self, hard_env):
        score, _ = hard_env.grade()
        assert 0.0 <= score <= 1.0

    def test_full_score_correct_refactor(self):
        env = SchemaMigrationEnv(task_name="refactor_employees")
        env.reset()

        # Full refactor
        env._db.execute_sql("""
            CREATE TABLE departments (
                dept_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE
            );
            INSERT INTO departments (name) VALUES
                ('Engineering'), ('Product'), ('Design'), ('HR');

            CREATE TABLE job_titles (
                title_id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL UNIQUE
            );
            INSERT INTO job_titles (title) VALUES
                ('Senior Engineer'), ('Engineering Lead'), ('Product Manager'),
                ('UX Designer'), ('Engineer'), ('HR Manager'), ('Product Analyst'),
                ('UI Designer'), ('HR Analyst');

            CREATE TABLE employees (
                emp_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name  TEXT NOT NULL,
                email      TEXT NOT NULL UNIQUE,
                dept_id    INTEGER NOT NULL REFERENCES departments(dept_id),
                title_id   INTEGER NOT NULL REFERENCES job_titles(title_id),
                salary     REAL NOT NULL,
                manager_id INTEGER REFERENCES employees(emp_id),
                hire_date  TEXT NOT NULL,
                is_active  INTEGER NOT NULL DEFAULT 1
            );
            INSERT INTO employees (full_name, email, dept_id, title_id, salary, manager_id, hire_date, is_active)
            SELECT
                er.full_name, er.email,
                d.dept_id,
                jt.title_id,
                er.salary,
                NULL,
                er.hire_date,
                er.is_active
            FROM employee_records er
            JOIN departments d  ON d.name  = er.department
            JOIN job_titles jt  ON jt.title = er.job_title;

            UPDATE employees SET manager_id = (
                SELECT e2.emp_id FROM employees e2
                JOIN employee_records er2 ON er2.email = e2.email
                JOIN employee_records er1 ON er1.manager_email = er2.email
                WHERE er1.email = employees.email
            ) WHERE EXISTS (
                SELECT 1 FROM employee_records er
                WHERE er.email = employees.email AND er.manager_email IS NOT NULL
            );

            DROP TABLE employee_records;

            CREATE VIEW employee_records AS
            SELECT
                e.emp_id,
                e.full_name,
                e.email,
                d.name   AS department,
                jt.title AS job_title,
                e.salary,
                mgr.email AS manager_email,
                e.hire_date,
                e.is_active
            FROM employees e
            JOIN departments d  ON d.dept_id  = e.dept_id
            JOIN job_titles jt  ON jt.title_id = e.title_id
            LEFT JOIN employees mgr ON mgr.emp_id = e.manager_id;
        """)

        score, notes = env.grade()
        print("\nHard task notes:", notes)
        assert score >= 0.80, f"Expected >=0.80, got {score}. Notes: {notes}"


# ---------------------------------------------------------------------------
# API (integration test — requires running server)
# ---------------------------------------------------------------------------

class TestServer:
    """Lightweight smoke test for the HTTP API. Skipped if server is not running."""

    @pytest.mark.skip(reason="Requires running server — run manually")
    def test_health_endpoint(self):
        import requests
        resp = requests.get("http://localhost:7860/health", timeout=5)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    @pytest.mark.skip(reason="Requires running server — run manually")
    def test_reset_and_step(self):
        import requests
        base = "http://localhost:7860"
        obs = requests.post(f"{base}/reset", json={"task": "add_columns"}).json()
        assert "current_schema" in obs
        result = requests.post(f"{base}/step", json={"action_type": "inspect_schema"}).json()
        assert "observation" in result
        assert "reward" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
