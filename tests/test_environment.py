import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import SchemaMigrationEnv
from env.models import ActionType, MigrationAction


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


class TestLifecycle:
    def test_reset_returns_observation(self, easy_env):
        obs = easy_env.reset()
        assert obs.current_schema
        assert obs.migration_spec
        assert obs.step == 0
        assert obs.migration_hints

    def test_state_returns_dict(self, easy_env):
        state = easy_env.state()
        assert state["task"] == "add_columns"
        assert state["step"] == 0
        assert state["done"] is False

    def test_submit_ends_episode(self, easy_env):
        _, reward, done, _ = easy_env.step(MigrationAction(action_type=ActionType.SUBMIT))
        assert done is True
        assert reward.breakdown is not None

    def test_step_after_done_returns_done(self, easy_env):
        easy_env.step(MigrationAction(action_type=ActionType.SUBMIT))
        _, _, done, _ = easy_env.step(MigrationAction(action_type=ActionType.INSPECT_SCHEMA))
        assert done is True


class TestDatabase:
    def test_initial_schema_has_products(self, easy_env):
        schema = {t.name: t for t in easy_env._db.get_schema()}
        assert "products" in schema
        assert schema["products"].row_count == 5

    def test_execute_adds_column(self, easy_env):
        result = easy_env._db.execute_sql("ALTER TABLE products ADD COLUMN stock_quantity INTEGER NOT NULL DEFAULT 0")
        assert result.success, result.message
        assert easy_env._db.column_exists("products", "stock_quantity")

    def test_rollback_restores_last_executed_migration(self, easy_env):
        easy_env.step(MigrationAction(action_type=ActionType.WRITE_MIGRATION, sql="ALTER TABLE products ADD COLUMN temp_col INTEGER"))
        easy_env.step(MigrationAction(action_type=ActionType.EXECUTE))
        assert easy_env._db.column_exists("products", "temp_col")
        easy_env.step(MigrationAction(action_type=ActionType.ROLLBACK))
        assert not easy_env._db.column_exists("products", "temp_col")

    def test_run_query_returns_data(self, easy_env):
        result = easy_env._db.run_query("SELECT COUNT(*) AS cnt FROM products")
        assert result.success
        assert result.query_result[0]["cnt"] == 5


class TestRewards:
    def test_inspect_schema_zero_reward(self, easy_env):
        _, reward, _, _ = easy_env.step(MigrationAction(action_type=ActionType.INSPECT_SCHEMA))
        assert reward.value == 0.0

    def test_write_valid_sql_positive_reward(self, easy_env):
        _, reward, _, _ = easy_env.step(MigrationAction(action_type=ActionType.WRITE_MIGRATION, sql="ALTER TABLE products ADD COLUMN stock_quantity INTEGER NOT NULL DEFAULT 0"))
        assert reward.value > 0

    def test_write_invalid_sql_negative_reward(self, easy_env):
        _, reward, _, _ = easy_env.step(MigrationAction(action_type=ActionType.WRITE_MIGRATION, sql="DROP DATABASE prod"))
        assert reward.value < 0

    def test_execute_success_positive_reward(self, easy_env):
        easy_env.step(MigrationAction(action_type=ActionType.WRITE_MIGRATION, sql="ALTER TABLE products ADD COLUMN stock_quantity INTEGER NOT NULL DEFAULT 0"))
        _, reward, _, _ = easy_env.step(MigrationAction(action_type=ActionType.EXECUTE))
        assert reward.value > 0

    def test_rollback_penalty_increases(self, easy_env):
        easy_env.step(MigrationAction(action_type=ActionType.WRITE_MIGRATION, sql="ALTER TABLE products ADD COLUMN c1 INTEGER"))
        easy_env.step(MigrationAction(action_type=ActionType.EXECUTE))
        easy_env.step(MigrationAction(action_type=ActionType.WRITE_MIGRATION, sql="ALTER TABLE products ADD COLUMN c2 INTEGER"))
        easy_env.step(MigrationAction(action_type=ActionType.EXECUTE))
        penalties = []
        for _ in range(2):
            _, reward, _, _ = easy_env.step(MigrationAction(action_type=ActionType.ROLLBACK))
            penalties.append(reward.value)
        assert penalties[0] >= penalties[1]


class TestGraderEasy:
    def test_score_without_migration_is_partial(self, easy_env):
        score, notes = easy_env.grade()
        assert 0.0 <= score <= 1.0
        assert notes

    def test_full_score_with_correct_migration(self):
        env = SchemaMigrationEnv(task_name="add_columns")
        env.reset()
        env._db.execute_sql("ALTER TABLE products ADD COLUMN stock_quantity INTEGER NOT NULL DEFAULT 0")
        env._db.execute_sql("ALTER TABLE products ADD COLUMN category TEXT")
        env._db.execute_sql("ALTER TABLE products ADD COLUMN created_at TEXT NOT NULL DEFAULT '2024-01-01'")
        score, notes = env.grade()
        assert score >= 0.83, notes


class TestGraderMedium:
    def test_zero_score_no_migration(self, medium_env):
        score, _ = medium_env.grade()
        assert score < 0.75

    def test_full_score_correct_normalization(self):
        env = SchemaMigrationEnv(task_name="normalize_orders")
        env.reset()
        result = env._db.execute_sql(
            """
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
                order_id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
                product_id INTEGER NOT NULL REFERENCES products(product_id),
                quantity INTEGER NOT NULL DEFAULT 1,
                order_date TEXT NOT NULL
            );
            INSERT INTO new_orders (customer_id, product_id, quantity, order_date)
            SELECT c.customer_id, p.product_id, o.quantity, o.order_date
            FROM orders o
            JOIN customers c ON c.email = o.customer_email
            JOIN products p ON p.name = o.product_name;
            DROP TABLE orders;
            ALTER TABLE new_orders RENAME TO orders;
            """
        )
        assert result.success, result.message
        score, notes = env.grade()
        assert score >= 0.75, notes


class TestGraderHard:
    def test_zero_score_no_migration(self, hard_env):
        score, _ = hard_env.grade()
        assert score < 0.75

    def test_full_score_correct_refactor(self):
        env = SchemaMigrationEnv(task_name="refactor_employees")
        env.reset()
        result = env._db.execute_sql(
            """
            CREATE TABLE departments (
                dept_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE
            );
            INSERT INTO departments (name) VALUES ('Engineering'), ('Product'), ('Design'), ('HR');

            CREATE TABLE job_titles (
                title_id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL UNIQUE
            );
            INSERT INTO job_titles (title) VALUES
                ('Senior Engineer'), ('Engineering Lead'), ('Product Manager'), ('UX Designer'),
                ('Engineer'), ('HR Manager'), ('Product Analyst'), ('UI Designer'), ('HR Analyst');

            CREATE TABLE employees (
                emp_id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                dept_id INTEGER NOT NULL REFERENCES departments(dept_id),
                title_id INTEGER NOT NULL REFERENCES job_titles(title_id),
                salary REAL NOT NULL,
                manager_id INTEGER REFERENCES employees(emp_id),
                hire_date TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1
            );
            INSERT INTO employees (emp_id, full_name, email, dept_id, title_id, salary, manager_id, hire_date, is_active)
            SELECT
                er.emp_id,
                er.full_name,
                er.email,
                d.dept_id,
                jt.title_id,
                er.salary,
                NULL,
                er.hire_date,
                er.is_active
            FROM employee_records er
            JOIN departments d ON d.name = er.department
            JOIN job_titles jt ON jt.title = er.job_title;
            UPDATE employees
            SET manager_id = (
                SELECT manager.emp_id
                FROM employee_records source
                JOIN employees manager ON manager.email = source.manager_email
                WHERE source.email = employees.email
            )
            WHERE EXISTS (
                SELECT 1 FROM employee_records source
                JOIN employees manager ON manager.email = source.manager_email
                WHERE source.email = employees.email
            );
            DROP TABLE employee_records;
            CREATE VIEW employee_records AS
            SELECT
                e.emp_id,
                e.full_name,
                e.email,
                d.name AS department,
                jt.title AS job_title,
                e.salary,
                mgr.email AS manager_email,
                e.hire_date,
                e.is_active
            FROM employees e
            JOIN departments d ON d.dept_id = e.dept_id
            JOIN job_titles jt ON jt.title_id = e.title_id
            LEFT JOIN employees mgr ON mgr.emp_id = e.manager_id;
            """
        )
        assert result.success, result.message
        score, notes = env.grade()
        assert score >= 0.7, notes


class TestServer:
    @pytest.mark.skip(reason="Requires running server")
    def test_health_endpoint(self):
        import requests
        response = requests.get("http://localhost:7860/health", timeout=5)
        assert response.status_code == 200



