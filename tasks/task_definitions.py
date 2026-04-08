"""
Task definitions for the Schema Migration environment.

Each task is a dataclass containing:
  - name / difficulty / description
  - seed_sql   : the starting database state
  - spec       : natural-language migration specification shown to the agent
  - requirements : bullet-point checklist shown in the observation
  - grader     : a callable(db: MigrationDB, pre_snapshot: str) -> float in [0,1]
  - max_steps  : step budget
  - test_queries: list of (description, sql, expected_value) tuples used in Hard task
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from env.database import MigrationDB


# ---------------------------------------------------------------------------
# Helper: lightweight scorer
# ---------------------------------------------------------------------------

def _score(*components: Tuple[str, float, float]) -> Tuple[float, List[str]]:
    """
    components: (label, earned, max)
    Returns (total_score_0_to_1, notes_list)
    """
    total_earned = sum(c[1] for c in components)
    total_max = sum(c[2] for c in components)
    score = round(max(0.0, min(1.0, total_earned / total_max if total_max > 0 else 0.0)), 4)
    notes = [f"{c[0]}: {c[1]:.2f}/{c[2]:.2f}" for c in components]
    return score, notes


# ===========================================================================
# TASK 1 — EASY: Add columns to an existing table
# ===========================================================================

_EASY_SEED = textwrap.dedent("""
    CREATE TABLE products (
        id      INTEGER PRIMARY KEY AUTOINCREMENT,
        name    TEXT    NOT NULL,
        price   REAL    NOT NULL
    );

    INSERT INTO products (name, price) VALUES
        ('Laptop',   999.99),
        ('Mouse',     29.99),
        ('Keyboard',  79.99),
        ('Monitor',  349.99),
        ('Headset',   89.99);
""").strip()

_EASY_SPEC = textwrap.dedent("""
    The e-commerce team needs three new fields on the `products` table:

    1. `stock_quantity`  — INTEGER, NOT NULL, DEFAULT 0
       Tracks how many units are in stock.

    2. `category`        — TEXT, nullable (NULL is allowed)
       Product category string (e.g. 'Electronics', 'Accessories').

    3. `created_at`      — TEXT, NOT NULL, DEFAULT '2024-01-01'
       ISO-8601 date string recording when the product was added.

    All existing rows must survive the migration with their original data intact.
    The new columns should appear alongside the existing ones.
""").strip()

_EASY_REQS = [
    "Add column: stock_quantity INTEGER NOT NULL DEFAULT 0",
    "Add column: category TEXT (nullable)",
    "Add column: created_at TEXT NOT NULL DEFAULT '2024-01-01'",
    "Preserve all 5 existing product rows",
    "Existing columns (id, name, price) must be unchanged",
]


def _grade_easy(db: MigrationDB, _pre: str) -> Tuple[float, List[str]]:
    notes: List[str] = []
    earned = 0.0

    # 1. Table still exists
    if not db.table_exists("products"):
        return 0.0, ["FAIL: products table missing"]

    # 2. Row count preserved
    rc = db.get_row_count("products")
    if rc == 5:
        earned += 1.5
        notes.append("row_count: 1.5/1.5 (5 rows intact)")
    else:
        notes.append(f"row_count: 0/1.5 ({rc} rows, expected 5)")

    # 3. New columns exist with correct types / constraints
    col_checks = [
        ("stock_quantity", "INTEGER", False),   # (name, type_contains, nullable)
        ("category",       "TEXT",    True),
        ("created_at",     "TEXT",    False),
    ]
    for cname, ctype, nullable in col_checks:
        if db.column_exists("products", cname):
            # Just check existence — type enforcement is best-effort in SQLite
            earned += 1.0
            notes.append(f"{cname}: 1.0/1.0 (present)")
        else:
            notes.append(f"{cname}: 0/1.0 (missing)")

    # 4. Original columns intact
    schema_tables = {t.name: t for t in db.get_schema()}
    if "products" in schema_tables:
        existing_names = {c.name for c in schema_tables["products"].columns}
        for col in ("id", "name", "price"):
            if col in existing_names:
                earned += 0.5
                notes.append(f"original {col}: 0.5/0.5 (intact)")
            else:
                notes.append(f"original {col}: 0/0.5 (MISSING)")

    # 5. Default values — check stock_quantity and created_at for existing rows
    result = db.run_query("SELECT stock_quantity, created_at FROM products LIMIT 1")
    if result.success and result.query_result:
        row = result.query_result[0]
        if row.get("stock_quantity") == 0:
            earned += 0.5
            notes.append("default stock_quantity=0: 0.5/0.5")
        else:
            notes.append(f"default stock_quantity: 0/0.5 (got {row.get('stock_quantity')})")
        if row.get("created_at") == "2024-01-01":
            earned += 0.5
            notes.append("default created_at='2024-01-01': 0.5/0.5")
        else:
            notes.append(f"default created_at: 0/0.5 (got {row.get('created_at')})")

    total_max = 1.5 + 3 * 1.0 + 3 * 0.5 + 0.5 + 0.5  # = 7.5
    score = round(max(0.0, min(1.0, earned / total_max)), 4)
    return score, notes


# ===========================================================================
# TASK 2 — MEDIUM: Normalize a denormalized table (1NF → 3NF)
# ===========================================================================

_MEDIUM_SEED = textwrap.dedent("""
    -- Denormalized orders table: customer data repeated in every row
    CREATE TABLE orders (
        order_id        INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_name   TEXT    NOT NULL,
        customer_email  TEXT    NOT NULL,
        customer_city   TEXT,
        product_name    TEXT    NOT NULL,
        product_price   REAL    NOT NULL,
        quantity        INTEGER NOT NULL DEFAULT 1,
        order_date      TEXT    NOT NULL
    );

    INSERT INTO orders VALUES
        (1,'Alice Smith','alice@example.com','Mumbai',   'Laptop',   999.99,1,'2024-01-10'),
        (2,'Alice Smith','alice@example.com','Mumbai',   'Mouse',     29.99,2,'2024-01-10'),
        (3,'Bob Jones',  'bob@example.com',  'Delhi',    'Keyboard',  79.99,1,'2024-01-11'),
        (4,'Alice Smith','alice@example.com','Mumbai',   'Monitor',  349.99,1,'2024-01-15'),
        (5,'Carol White','carol@example.com','Bangalore','Headset',   89.99,3,'2024-01-16'),
        (6,'Bob Jones',  'bob@example.com',  'Delhi',    'Laptop',   999.99,1,'2024-01-17'),
        (7,'Carol White','carol@example.com','Bangalore','Mouse',     29.99,1,'2024-01-18');
""").strip()

_MEDIUM_SPEC = textwrap.dedent("""
    The orders table is denormalized — customer data is repeated for every order.
    Normalize the database into three tables:

    1. `customers` table
       - customer_id  INTEGER PRIMARY KEY AUTOINCREMENT
       - name         TEXT NOT NULL
       - email        TEXT NOT NULL UNIQUE
       - city         TEXT

    2. `products` table
       - product_id   INTEGER PRIMARY KEY AUTOINCREMENT
       - name         TEXT NOT NULL UNIQUE
       - price        REAL NOT NULL

    3. `orders` table (replace the existing one)
       - order_id     INTEGER PRIMARY KEY AUTOINCREMENT
       - customer_id  INTEGER NOT NULL REFERENCES customers(customer_id)
       - product_id   INTEGER NOT NULL REFERENCES products(product_id)
       - quantity     INTEGER NOT NULL DEFAULT 1
       - order_date   TEXT NOT NULL

    ALL 7 original order records must be preserved (no data loss).
    Unique customers: Alice Smith, Bob Jones, Carol White (3 rows in customers).
    Unique products: Laptop, Mouse, Keyboard, Monitor, Headset (5 rows in products).
    The old denormalized `orders` table must be replaced by the normalized version.
""").strip()

_MEDIUM_REQS = [
    "Create customers table with correct schema + 3 rows",
    "Create products table with correct schema + 5 rows",
    "Replace orders table with normalized version (FKs to customers + products)",
    "Preserve all 7 order records",
    "FK constraints must hold (no orphaned references)",
    "No duplicate customers or products",
]


def _grade_medium(db: MigrationDB, _pre: str) -> Tuple[float, List[str]]:
    notes: List[str] = []
    earned = 0.0

    # 1. All three tables exist
    for tname in ("customers", "products", "orders"):
        if db.table_exists(tname):
            earned += 0.5
            notes.append(f"table {tname}: 0.5/0.5")
        else:
            notes.append(f"table {tname}: 0/0.5 MISSING")

    # 2. Customer count
    nc = db.get_row_count("customers")
    if nc == 3:
        earned += 1.5
        notes.append("customers rows: 1.5/1.5")
    elif nc is not None and nc > 0:
        earned += 0.5
        notes.append(f"customers rows: 0.5/1.5 (got {nc}, expected 3)")
    else:
        notes.append(f"customers rows: 0/1.5 (got {nc})")

    # 3. Product count
    np_ = db.get_row_count("products")
    if np_ == 5:
        earned += 1.5
        notes.append("products rows: 1.5/1.5")
    elif np_ is not None and np_ > 0:
        earned += 0.5
        notes.append(f"products rows: 0.5/1.5 (got {np_}, expected 5)")
    else:
        notes.append(f"products rows: 0/1.5 (got {np_})")

    # 4. Order count preserved
    no = db.get_row_count("orders")
    if no == 7:
        earned += 2.0
        notes.append("orders rows: 2.0/2.0 (7 rows preserved)")
    elif no is not None and no > 0:
        earned += 0.5
        notes.append(f"orders rows: 0.5/2.0 (got {no}, expected 7)")
    else:
        notes.append(f"orders rows: 0/2.0 (got {no})")

    # 5. FK columns exist in orders
    for fk_col in ("customer_id", "product_id"):
        if db.column_exists("orders", fk_col):
            earned += 0.5
            notes.append(f"orders.{fk_col}: 0.5/0.5")
        else:
            notes.append(f"orders.{fk_col}: 0/0.5 MISSING")

    # 6. FK integrity — no orphaned references
    violations = db.fk_violations()
    if violations == 0:
        earned += 1.5
        notes.append("FK integrity: 1.5/1.5 (no violations)")
    else:
        notes.append(f"FK integrity: 0/1.5 ({violations} violation(s))")

    # 7. No duplicate emails in customers
    result = db.run_query("SELECT COUNT(*) as cnt FROM customers")
    uniq = db.run_query("SELECT COUNT(DISTINCT email) as cnt FROM customers")
    if (result.success and uniq.success and result.query_result and uniq.query_result
            and result.query_result[0]["cnt"] == uniq.query_result[0]["cnt"]):
        earned += 0.5
        notes.append("no duplicate customers: 0.5/0.5")
    else:
        notes.append("no duplicate customers: 0/0.5")

    total_max = 1.5 + 1.5 + 1.5 + 2.0 + 1.0 + 1.5 + 0.5  # = 9.5
    score = round(max(0.0, min(1.0, earned / total_max)), 4)
    return score, notes


# ===========================================================================
# TASK 3 — HARD: Multi-table refactor with compatibility view
# ===========================================================================

_HARD_SEED = textwrap.dedent("""
    -- Legacy monolithic schema used by an internal analytics system
    CREATE TABLE employee_records (
        emp_id          INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name       TEXT    NOT NULL,
        email           TEXT    NOT NULL UNIQUE,
        department      TEXT    NOT NULL,
        job_title       TEXT    NOT NULL,
        salary          REAL    NOT NULL,
        manager_email   TEXT,
        hire_date       TEXT    NOT NULL,
        is_active       INTEGER NOT NULL DEFAULT 1
    );

    INSERT INTO employee_records VALUES
        (1,'Priya Sharma',  'priya@corp.com',  'Engineering','Senior Engineer',  120000,'ravi@corp.com',  '2020-03-15',1),
        (2,'Ravi Patel',    'ravi@corp.com',   'Engineering','Engineering Lead',  150000,NULL,            '2018-06-01',1),
        (3,'Anjali Nair',   'anjali@corp.com', 'Product',    'Product Manager',   130000,'ceo@corp.com',  '2019-11-20',1),
        (4,'Kiran Desai',   'kiran@corp.com',  'Design',     'UX Designer',        90000,'anjali@corp.com','2021-07-10',1),
        (5,'Meera Iyer',    'meera@corp.com',  'Engineering','Engineer',           95000,'ravi@corp.com', '2022-01-05',1),
        (6,'Suresh Babu',   'suresh@corp.com', 'HR',         'HR Manager',        100000,NULL,            '2017-09-12',1),
        (7,'Divya Menon',   'divya@corp.com',  'Product',    'Product Analyst',    80000,'anjali@corp.com','2023-02-28',1),
        (8,'Arjun Reddy',   'arjun@corp.com',  'Design',     'UI Designer',        85000,'kiran@corp.com','2022-08-19',0),
        (9,'Lakshmi Rao',   'lakshmi@corp.com','HR',         'HR Analyst',         75000,'suresh@corp.com','2023-05-01',1),
       (10,'Vikram Singh',  'vikram@corp.com', 'Engineering','Senior Engineer',   118000,'ravi@corp.com', '2020-12-10',1);
""").strip()

_HARD_SPEC = textwrap.dedent("""
    Refactor the legacy `employee_records` table into a proper normalized schema:

    NEW TABLES REQUIRED:
    ─────────────────────
    1. `departments`
       - dept_id    INTEGER PRIMARY KEY AUTOINCREMENT
       - name       TEXT NOT NULL UNIQUE

    2. `job_titles`
       - title_id   INTEGER PRIMARY KEY AUTOINCREMENT
       - title      TEXT NOT NULL UNIQUE

    3. `employees`
       - emp_id     INTEGER PRIMARY KEY AUTOINCREMENT
       - full_name  TEXT NOT NULL
       - email      TEXT NOT NULL UNIQUE
       - dept_id    INTEGER NOT NULL REFERENCES departments(dept_id)
       - title_id   INTEGER NOT NULL REFERENCES job_titles(title_id)
       - salary     REAL NOT NULL
       - manager_id INTEGER REFERENCES employees(emp_id)   -- self-referential FK
       - hire_date  TEXT NOT NULL
       - is_active  INTEGER NOT NULL DEFAULT 1

    COMPATIBILITY VIEW (CRITICAL):
    ───────────────────────────────
    Create a VIEW named `employee_records` (same name as the old table) that
    reproduces the EXACT columns the old table had, so existing application
    queries continue to work unchanged:

        emp_id, full_name, email, department, job_title, salary,
        manager_email, hire_date, is_active

    VERIFICATION:
    ─────────────
    These test queries must return the same results as before:
    Q1: SELECT COUNT(*) FROM employee_records           → must return 10
    Q2: SELECT COUNT(*) FROM employee_records WHERE is_active=1  → must return 9
    Q3: SELECT department FROM employee_records WHERE email='priya@corp.com'  → 'Engineering'
    Q4: SELECT salary FROM employee_records WHERE email='ravi@corp.com'  → 150000.0
    Q5: SELECT COUNT(DISTINCT department) FROM employee_records  → 4
    Q6: SELECT COUNT(*) FROM employees WHERE manager_id IS NULL  → 3 (Ravi, Suresh have no manager; Anjali's manager 'ceo@corp.com' is not in the table)

    ALL 10 original employee records must be preserved.
    The old `employee_records` TABLE must be replaced by the VIEW.
""").strip()

_HARD_REQS = [
    "Create departments table (4 unique departments as rows)",
    "Create job_titles table (9 unique job titles as rows)",
    "Create employees table with FK to departments, job_titles, self-ref manager_id",
    "Migrate all 10 employees with correct dept_id, title_id, manager_id",
    "Drop old employee_records TABLE; create employee_records VIEW with identical columns",
    "Q1–Q6 test queries all return correct results through the compatibility view",
    "FK integrity: zero violations",
]

# Test queries: (description, sql, expected_scalar_or_list)
_HARD_TEST_QUERIES: List[Tuple[str, str, Any]] = [
    ("total row count via view",        "SELECT COUNT(*) FROM employee_records", 10),
    ("active employees via view",       "SELECT COUNT(*) FROM employee_records WHERE is_active=1", 9),
    ("department lookup via view",      "SELECT department FROM employee_records WHERE email='priya@corp.com'", "Engineering"),
    ("salary lookup via view",          "SELECT salary FROM employee_records WHERE email='ravi@corp.com'", 150000.0),
    ("distinct departments via view",   "SELECT COUNT(DISTINCT department) FROM employee_records", 4),
    ("employees with no manager",       "SELECT COUNT(*) FROM employees WHERE manager_id IS NULL", 3),
]


def _grade_hard(db: MigrationDB, _pre: str) -> Tuple[float, List[str]]:
    notes: List[str] = []
    earned = 0.0

    # 1. Three new tables exist
    for tname in ("departments", "job_titles", "employees"):
        if db.table_exists(tname):
            earned += 0.5
            notes.append(f"table {tname}: 0.5/0.5")
        else:
            notes.append(f"table {tname}: 0/0.5 MISSING")

    # 2. Department count
    nd = db.get_row_count("departments")
    if nd == 4:
        earned += 1.0
        notes.append("departments rows: 1.0/1.0")
    elif nd is not None and nd > 0:
        earned += 0.3
        notes.append(f"departments rows: 0.3/1.0 (got {nd}, expected 4)")
    else:
        notes.append(f"departments rows: 0/1.0 (got {nd})")

    # 3. Job title count
    njt = db.get_row_count("job_titles")
    if njt == 9:
        earned += 1.0
        notes.append("job_titles rows: 1.0/1.0")
    elif njt is not None and njt > 0:
        earned += 0.3
        notes.append(f"job_titles rows: 0.3/1.0 (got {njt}, expected 9)")
    else:
        notes.append(f"job_titles rows: 0/1.0 (got {njt})")

    # 4. Employee count
    ne = db.get_row_count("employees")
    if ne == 10:
        earned += 1.5
        notes.append("employees rows: 1.5/1.5")
    elif ne is not None and ne > 0:
        earned += 0.5
        notes.append(f"employees rows: 0.5/1.5 (got {ne})")
    else:
        notes.append(f"employees rows: 0/1.5 (got {ne})")

    # 5. FK integrity
    violations = db.fk_violations()
    if violations == 0:
        earned += 1.0
        notes.append("FK integrity: 1.0/1.0")
    else:
        notes.append(f"FK integrity: 0/1.0 ({violations} violation(s))")

    # 6. Test queries through compatibility view (most critical)
    query_score = 0.0
    per_query = 2.0 / len(_HARD_TEST_QUERIES)
    for desc, sql, expected in _HARD_TEST_QUERIES:
        val = db.query_scalar(sql)
        if val == expected or (isinstance(expected, float) and abs(float(val or 0) - expected) < 0.01):
            query_score += per_query
            notes.append(f"Q '{desc}': PASS")
        else:
            notes.append(f"Q '{desc}': FAIL (got {val!r}, expected {expected!r})")
    earned += query_score
    notes.append(f"test queries total: {query_score:.2f}/2.0")

    # 7. employee_records is now a VIEW (not a table)
    assert db._conn
    obj_type = db._conn.execute(
        "SELECT type FROM sqlite_master WHERE name='employee_records'"
    ).fetchone()
    if obj_type and obj_type[0] == "view":
        earned += 1.0
        notes.append("employee_records is a VIEW: 1.0/1.0")
    elif obj_type and obj_type[0] == "table":
        notes.append("employee_records is still a TABLE (not converted to view): 0/1.0")
    else:
        notes.append("employee_records not found: 0/1.0")

    total_max = 1.5 + 1.0 + 1.0 + 1.5 + 1.0 + 2.0 + 1.0  # = 9.0
    score = round(max(0.0, min(1.0, earned / total_max)), 4)
    return score, notes


# ===========================================================================
# Task registry
# ===========================================================================

@dataclass
class Task:
    name: str
    difficulty: str          # "easy" | "medium" | "hard"
    description: str
    seed_sql: str
    spec: str
    requirements: List[str]
    grader: Callable          # (db: MigrationDB, pre_snapshot: str) -> Tuple[float, List[str]]
    max_steps: int = 20


TASKS: Dict[str, Task] = {
    "add_columns": Task(
        name="add_columns",
        difficulty="easy",
        description="Add three new columns to the products table with correct types and defaults.",
        seed_sql=_EASY_SEED,
        spec=_EASY_SPEC,
        requirements=_EASY_REQS,
        grader=_grade_easy,
        max_steps=15,
    ),
    "normalize_orders": Task(
        name="normalize_orders",
        difficulty="medium",
        description="Normalize a denormalized orders table into customers, products, and orders tables.",
        seed_sql=_MEDIUM_SEED,
        spec=_MEDIUM_SPEC,
        requirements=_MEDIUM_REQS,
        grader=_grade_medium,
        max_steps=25,
    ),
    "refactor_employees": Task(
        name="refactor_employees",
        difficulty="hard",
        description=(
            "Refactor a legacy employee_records table into a normalized schema "
            "and create a compatibility view so existing queries keep working."
        ),
        seed_sql=_HARD_SEED,
        spec=_HARD_SPEC,
        requirements=_HARD_REQS,
        grader=_grade_hard,
        max_steps=35,
    ),
}

