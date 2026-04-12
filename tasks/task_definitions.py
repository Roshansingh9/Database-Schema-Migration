"""
Task definitions and grading helpers for the Schema Migration environment.

Each task contains a real SQLite seed schema plus deterministic grading logic
that validates the migrated database by executing live queries.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from env.database import MigrationDB


@dataclass
class Task:
    name: str
    difficulty: str
    description: str
    seed_sql: str
    spec: str
    requirements: List[str]
    hints: Dict[str, Any]
    grader: Callable[[MigrationDB, str, Dict[str, Any]], Tuple[float, List[str], Dict[str, float]]]
    max_steps: int = 20
    expected_objects: List[str] = field(default_factory=list)


def build_seed_metrics(pre_snapshot: str) -> Dict[str, Any]:
    db = MigrationDB()
    db.init(pre_snapshot)
    metrics = {
        "tables": db.get_tables(),
        "table_metrics": db.snapshot_table_metrics(),
        "schema": {table.name: table for table in db.get_schema()},
    }
    db.close()
    return metrics


def _normalize_score(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 4)


def _strict_openenv_score(raw: float) -> float:
    score = round(0.05 + max(0.0, min(1.0, raw)) * 0.90, 4)
    return max(0.05, min(0.95, score))


def _finalize_breakdown(
    syntax_score: float,
    execution_score: float,
    correctness_score: float,
    integrity_score: float,
    efficiency_penalty: float,
    notes: List[str],
) -> Tuple[float, List[str], Dict[str, float]]:
    raw_total = _normalize_score(
        0.20 * syntax_score
        + 0.20 * execution_score
        + 0.35 * correctness_score
        + 0.25 * integrity_score
        - efficiency_penalty
    )
    total = _strict_openenv_score(raw_total)
    breakdown = {
        "syntax_score": _normalize_score(syntax_score),
        "execution_score": _normalize_score(execution_score),
        "correctness_score": _normalize_score(correctness_score),
        "integrity_score": _normalize_score(integrity_score),
        "efficiency_penalty": _normalize_score(efficiency_penalty),
        "total": total,
    }
    return total, notes, breakdown


def _legacy_integrity_score(db: MigrationDB, seed_metrics: Dict[str, Any], protected_tables: List[str]) -> Tuple[float, List[str]]:
    notes: List[str] = []
    checks = 0
    passed = 0
    for table in protected_tables:
        checks += 1
        before = seed_metrics["table_metrics"].get(table)
        after_row_count = db.get_row_count(table)
        if before and after_row_count == before["row_count"]:
            passed += 1
            notes.append(f"row count preserved for {table}")
        else:
            notes.append(f"row count mismatch for {table}: expected {before['row_count'] if before else 'missing'}, got {after_row_count}")

        checks += 1
        after_checksum = db.get_checksum(table)
        if before and after_checksum == before["checksum"]:
            passed += 1
            notes.append(f"checksum preserved for {table}")
        else:
            notes.append(f"checksum mismatch for {table}")
    return (passed / checks if checks else 1.0), notes


def _redundant_column_penalty(db: MigrationDB, table: str, banned_columns: List[str]) -> float:
    if not db.table_exists(table):
        return 0.0
    schema = {s.name: s for s in db.get_schema()}
    current = schema.get(table)
    if current is None:
        return 0.0
    present = sum(1 for col in banned_columns if any(c.name == col for c in current.columns))
    return min(0.2, 0.05 * present)


def _object_score(db: MigrationDB, expected_objects: List[str]) -> Tuple[float, List[str]]:
    diff = db.schema_diff(expected_objects)
    notes = [f"missing objects: {diff['missing']}" if diff["missing"] else "no expected objects missing"]
    if diff["unexpected"]:
        notes.append(f"unexpected objects: {diff['unexpected']}")
    total = len(expected_objects) + max(len(diff["unexpected"]), 0)
    if total == 0:
        return 1.0, notes
    score = max(0.0, 1.0 - (len(diff["missing"]) + 0.25 * len(diff["unexpected"])) / total)
    return score, notes


_EASY_SEED = textwrap.dedent(
    """
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
    """
).strip()

_EASY_SPEC = textwrap.dedent(
    """
    The e-commerce team needs three new fields on the products table.

    Add:
    - stock_quantity INTEGER NOT NULL DEFAULT 0
    - category TEXT NULL
    - created_at TEXT NOT NULL DEFAULT '2024-01-01'

    Preserve all existing rows and keep the original id, name, and price data unchanged.
    """
).strip()

_EASY_REQS = [
    "Add stock_quantity INTEGER NOT NULL DEFAULT 0 to products",
    "Add category TEXT nullable to products",
    "Add created_at TEXT NOT NULL DEFAULT '2024-01-01' to products",
    "Preserve all 5 existing product rows without changing id/name/price values",
    "Do not add unrelated columns",
]


def _grade_easy(db: MigrationDB, pre_snapshot: str, seed_metrics: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, float]]:
    notes: List[str] = []
    syntax_score = 1.0
    execution_score = 1.0 if db.table_exists("products") else 0.0

    schema = {table.name: table for table in db.get_schema()}
    products = schema.get("products")
    correctness = 0.0
    if products:
        expected = {
            "stock_quantity": ("INTEGER", False),
            "category": ("TEXT", True),
            "created_at": ("TEXT", False),
        }
        present = 0
        for name, (col_type, nullable) in expected.items():
            column = next((c for c in products.columns if c.name == name), None)
            if column and col_type in column.type.upper() and column.nullable == nullable:
                present += 1
                notes.append(f"column {name} present with expected shape")
            elif column:
                notes.append(f"column {name} present but constraints differ")
            else:
                notes.append(f"column {name} missing")
        correctness += present / len(expected)

        original_query = db.compare_query_results(
            "SELECT id, name, price FROM products ORDER BY id",
            "SELECT id, name, price FROM products ORDER BY id",
        )
        if original_query["match"]:
            correctness += 0.25
            notes.append("original columns remain queryable")

        default_check = db.run_query(
            "SELECT MIN(stock_quantity) AS min_stock, MAX(stock_quantity) AS max_stock, "
            "MIN(created_at) AS min_created, MAX(created_at) AS max_created FROM products"
        )
        if default_check.success and default_check.query_result:
            row = default_check.query_result[0]
            if row.get("min_stock") == 0 and row.get("max_stock") == 0:
                correctness += 0.1
                notes.append("stock_quantity defaults preserved on existing rows")
            if row.get("min_created") == "2024-01-01" and row.get("max_created") == "2024-01-01":
                correctness += 0.1
                notes.append("created_at defaults preserved on existing rows")

    integrity_score, integrity_notes = _legacy_integrity_score(db, seed_metrics, ["products"])
    notes.extend(integrity_notes)
    efficiency_penalty = _redundant_column_penalty(db, "products", ["stock", "product_category", "created_on"])
    if efficiency_penalty:
        notes.append("redundant columns detected on products")
    correctness = min(1.0, correctness)
    return _finalize_breakdown(
        syntax_score=syntax_score,
        execution_score=execution_score,
        correctness_score=correctness,
        integrity_score=integrity_score,
        efficiency_penalty=efficiency_penalty,
        notes=notes,
    )


_MEDIUM_SEED = textwrap.dedent(
    """
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
    """
).strip()

_MEDIUM_SPEC = textwrap.dedent(
    """
    Normalize the legacy orders table into customers, products, and a new normalized orders table.

    Required target schema:
    - customers(customer_id PK, name, email UNIQUE, city)
    - products(product_id PK, name UNIQUE, price)
    - orders(order_id PK, customer_id FK, product_id FK, quantity, order_date)

    Preserve all 7 original orders, deduplicate customers and products, and ensure foreign keys are valid.
    """
).strip()

_MEDIUM_REQS = [
    "Create customers with 3 unique rows",
    "Create products with 5 unique rows",
    "Replace the denormalized orders table with a normalized one using foreign keys",
    "Preserve all 7 order records and their quantities and dates",
    "Do not keep denormalized customer_* or product_* columns in the final orders table",
]


def _grade_medium(db: MigrationDB, pre_snapshot: str, seed_metrics: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, float]]:
    notes: List[str] = []
    object_score, object_notes = _object_score(db, ["customers", "products", "orders"])
    notes.extend(object_notes)
    execution_score = object_score

    correctness = 0.0
    if db.get_row_count("customers") == 3:
        correctness += 0.2
        notes.append("customers row count correct")
    else:
        notes.append(f"customers row count incorrect: {db.get_row_count('customers')}")

    if db.get_row_count("products") == 5:
        correctness += 0.2
        notes.append("products row count correct")
    else:
        notes.append(f"products row count incorrect: {db.get_row_count('products')}")

    if db.get_row_count("orders") == 7:
        correctness += 0.2
        notes.append("orders row count correct")
    else:
        notes.append(f"orders row count incorrect: {db.get_row_count('orders')}")

    email_uniqueness = db.query_scalar("SELECT COUNT(*) = COUNT(DISTINCT email) FROM customers")
    if email_uniqueness == 1:
        correctness += 0.1
        notes.append("customers.email is effectively unique")

    order_projection = db.run_query(
        """
        SELECT c.name AS customer_name, c.email AS customer_email, c.city AS customer_city,
               p.name AS product_name, p.price AS product_price, o.quantity, o.order_date
        FROM orders o
        JOIN customers c ON c.customer_id = o.customer_id
        JOIN products p ON p.product_id = o.product_id
        ORDER BY o.order_id
        """
    )
    seed_db = MigrationDB()
    seed_db.init(pre_snapshot)
    original_projection = seed_db.run_query(
        """
        SELECT customer_name, customer_email, customer_city, product_name, product_price, quantity, order_date
        FROM orders ORDER BY order_id
        """
    )
    seed_db.close()
    if order_projection.success and original_projection.success and order_projection.query_result == original_projection.query_result:
        correctness += 0.3
        notes.append("normalized join reproduces the original order data exactly")
    else:
        notes.append("normalized join does not match the original order data")

    integrity = 1.0
    fk_violations = db.fk_violations()
    if fk_violations != 0:
        integrity -= 0.4
        notes.append(f"foreign key violations detected: {fk_violations}")
    else:
        notes.append("foreign key integrity check passed")

    orders_schema = {s.name: s for s in db.get_schema()}.get("orders")
    banned_columns = ["customer_name", "customer_email", "customer_city", "product_name", "product_price"]
    efficiency_penalty = _redundant_column_penalty(db, "orders", banned_columns)
    if orders_schema:
        fk_cols = {c.name for c in orders_schema.columns if c.foreign_key}
        if {"customer_id", "product_id"}.issubset(fk_cols):
            integrity = min(1.0, integrity + 0.1)
            notes.append("orders foreign key columns are present")
    return _finalize_breakdown(
        syntax_score=1.0,
        execution_score=execution_score,
        correctness_score=min(1.0, correctness),
        integrity_score=min(1.0, integrity),
        efficiency_penalty=efficiency_penalty,
        notes=notes,
    )


_HARD_SEED = textwrap.dedent(
    """
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
    """
).strip()

_HARD_SPEC = textwrap.dedent(
    """
    Refactor employee_records into normalized departments, job_titles, and employees tables.

    Final required objects:
    - departments(dept_id PK, name UNIQUE)
    - job_titles(title_id PK, title UNIQUE)
    - employees(emp_id PK, full_name, email UNIQUE, dept_id FK, title_id FK, salary, manager_id FK, hire_date, is_active)
    - employee_records VIEW exposing: emp_id, full_name, email, department, job_title, salary, manager_email, hire_date, is_active

    The compatibility view must preserve the results of legacy read queries.
    """
).strip()

_HARD_REQS = [
    "Create departments with 4 unique rows",
    "Create job_titles with 9 unique rows",
    "Create employees with valid department/title/self-manager references",
    "Replace the old employee_records table with a compatibility view of the same name",
    "Preserve legacy query behavior through the view",
]

_HARD_COMPAT_QUERIES: List[Tuple[str, str, Any]] = [
    ("total row count", "SELECT COUNT(*) FROM employee_records", 10),
    ("active employees", "SELECT COUNT(*) FROM employee_records WHERE is_active=1", 9),
    ("priya department", "SELECT department FROM employee_records WHERE email='priya@corp.com'", "Engineering"),
    ("ravi salary", "SELECT salary FROM employee_records WHERE email='ravi@corp.com'", 150000.0),
    ("distinct departments", "SELECT COUNT(DISTINCT department) FROM employee_records", 4),
    ("employees with no manager", "SELECT COUNT(*) FROM employees WHERE manager_id IS NULL", 3),
]


def _grade_hard(db: MigrationDB, pre_snapshot: str, seed_metrics: Dict[str, Any]) -> Tuple[float, List[str], Dict[str, float]]:
    notes: List[str] = []
    execution_score, object_notes = _object_score(db, ["departments", "job_titles", "employees", "employee_records"])
    notes.extend(object_notes)

    correctness = 0.0
    if db.get_row_count("departments") == 4:
        correctness += 0.1
        notes.append("departments row count correct")
    if db.get_row_count("job_titles") == 9:
        correctness += 0.1
        notes.append("job_titles row count correct")
    if db.get_row_count("employees") == 10:
        correctness += 0.15
        notes.append("employees row count correct")

    if db.view_exists("employee_records") and not db.table_exists("employee_records"):
        correctness += 0.15
        notes.append("employee_records correctly replaced by a compatibility view")
    else:
        notes.append("employee_records is not a compatibility view")

    seed_db = MigrationDB()
    seed_db.init(pre_snapshot)
    legacy_projection = seed_db.run_query(
        "SELECT emp_id, full_name, email, department, job_title, salary, manager_email, hire_date, is_active FROM employee_records ORDER BY emp_id"
    )
    seed_db.close()
    current_projection = db.run_query(
        "SELECT emp_id, full_name, email, department, job_title, salary, manager_email, hire_date, is_active FROM employee_records ORDER BY emp_id"
    )
    if legacy_projection.success and current_projection.success and legacy_projection.query_result == current_projection.query_result:
        correctness += 0.3
        notes.append("compatibility view reproduces the full legacy result set")
    else:
        notes.append("compatibility view does not reproduce the full legacy result set")

    per_query = 0.2 / len(_HARD_COMPAT_QUERIES)
    for label, sql, expected in _HARD_COMPAT_QUERIES:
        value = db.query_scalar(sql)
        if value == expected:
            correctness += per_query
            notes.append(f"compatibility query passed: {label}")
        else:
            notes.append(f"compatibility query failed: {label} (got {value!r}, expected {expected!r})")

    integrity = 1.0
    fk_violations = db.fk_violations()
    if fk_violations:
        integrity -= 0.4
        notes.append(f"foreign key violations detected: {fk_violations}")
    else:
        notes.append("foreign key integrity check passed")

    manager_ids = db.run_query(
        "SELECT COUNT(*) AS cnt FROM employees e LEFT JOIN employees m ON m.emp_id = e.manager_id WHERE e.manager_id IS NOT NULL AND m.emp_id IS NULL"
    )
    if manager_ids.success and manager_ids.query_result and manager_ids.query_result[0]["cnt"] == 0:
        integrity += 0.1
        notes.append("self-referential manager links are valid")

    efficiency_penalty = 0.0
    if db.table_exists("employees"):
        efficiency_penalty += _redundant_column_penalty(db, "employees", ["department", "job_title", "manager_email"])
    return _finalize_breakdown(
        syntax_score=1.0,
        execution_score=execution_score,
        correctness_score=min(1.0, correctness),
        integrity_score=min(1.0, integrity),
        efficiency_penalty=min(0.3, efficiency_penalty),
        notes=notes,
    )


TASKS: Dict[str, Task] = {
    "add_columns": Task(
        name="add_columns",
        difficulty="easy",
        description="Add three new columns to products while preserving all data.",
        seed_sql=_EASY_SEED,
        spec=_EASY_SPEC,
        requirements=_EASY_REQS,
        hints={
            "legacy_tables": ["products"],
            "target_objects": ["products"],
            "validation_queries": ["SELECT id, name, price, stock_quantity, category, created_at FROM products ORDER BY id"],
        },
        grader=_grade_easy,
        max_steps=15,
        expected_objects=["products"],
    ),
    "normalize_orders": Task(
        name="normalize_orders",
        difficulty="medium",
        description="Normalize orders into customers, products, and foreign-key-backed orders.",
        seed_sql=_MEDIUM_SEED,
        spec=_MEDIUM_SPEC,
        requirements=_MEDIUM_REQS,
        hints={
            "legacy_tables": ["orders"],
            "target_objects": ["customers", "products", "orders"],
            "validation_queries": [
                "SELECT COUNT(*) FROM customers",
                "SELECT COUNT(*) FROM products",
                "SELECT COUNT(*) FROM orders",
            ],
        },
        grader=_grade_medium,
        max_steps=25,
        expected_objects=["customers", "products", "orders"],
    ),
    "refactor_employees": Task(
        name="refactor_employees",
        difficulty="hard",
        description="Refactor employee_records into normalized tables plus a compatibility view.",
        seed_sql=_HARD_SEED,
        spec=_HARD_SPEC,
        requirements=_HARD_REQS,
        hints={
            "legacy_tables": ["employee_records"],
            "target_objects": ["departments", "job_titles", "employees", "employee_records"],
            "validation_queries": [query for _, query, _ in _HARD_COMPAT_QUERIES],
        },
        grader=_grade_hard,
        max_steps=35,
        expected_objects=["departments", "job_titles", "employees", "employee_records"],
    ),
}


