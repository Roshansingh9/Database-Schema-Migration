---
title: Schema Migration OpenEnv
emoji: 🗄️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---


# Schema Migration OpenEnv

**A real-world RL environment where AI agents perform database schema migrations against live SQLite databases.**

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://huggingface.co/openenv)
[![Difficulty](https://img.shields.io/badge/difficulty-easy%20%7C%20medium%20%7C%20hard-orange)]()
[![Python](https://img.shields.io/badge/python-3.11+-green)]()

---

## Overview & Motivation

Every software engineering team runs database migrations. It is one of the most consequential, error-prone, and irreversible tasks a developer performs — a bad migration can corrupt production data for millions of users.

This environment trains and evaluates AI agents on the **full migration workflow**:
1. Read and understand a migration specification
2. Inspect the live database schema
3. Write correct SQL to achieve the migration
4. Execute it against a real (in-memory) SQLite database
5. Verify the result with queries
6. Submit for a deterministic, execution-based score

**Why this environment is unique:**
- Graders run real SQL against a real database — there is no way to "keyword stuff" a passing score
- The agent's actions have real, observable consequences in the database state
- Dense per-step rewards provide continuous feedback throughout the trajectory
- Three tasks spanning easy → medium → hard with genuine difficulty progression

---

## Tasks

### Task 1 — `add_columns` (Easy)

**Objective:** Add three new columns to a `products` table with correct types, constraints, and default values.

| Column | Type | Constraint |
|--------|------|------------|
| `stock_quantity` | INTEGER | NOT NULL DEFAULT 0 |
| `category` | TEXT | nullable |
| `created_at` | TEXT | NOT NULL DEFAULT '2024-01-01' |

**Starting state:** `products` table with 5 rows (id, name, price)  
**Success criterion:** All 5 rows intact, all 3 columns present with correct defaults  
**Max steps:** 15  
**Baseline score:** ~0.85 (Qwen2.5-72B)

---

### Task 2 — `normalize_orders` (Medium)

**Objective:** Normalize a denormalized `orders` table into three properly related tables.

**Before:** Single `orders` table with customer data repeated per row  
**After:** Separate `customers`, `products`, and `orders` tables with FK relationships

**Requirements:**
- 3 unique customers extracted → `customers` table
- 5 unique products extracted → `products` table  
- All 7 original order records preserved → normalized `orders` table
- FK constraints must hold (zero violations via `PRAGMA foreign_key_check`)

**Max steps:** 25  
**Baseline score:** ~0.72 (Qwen2.5-72B)

---

### Task 3 — `refactor_employees` (Hard)

**Objective:** Refactor a legacy 9-column `employee_records` table into a normalized 3-table schema **and** create a compatibility VIEW so existing application queries keep working unchanged.

**New tables:** `departments`, `job_titles`, `employees` (with self-referential `manager_id` FK)

**Critical requirement:** Create a VIEW named `employee_records` (same name as old table) that passes 6 test queries:

| Query | Expected |
|-------|----------|
| `SELECT COUNT(*) FROM employee_records` | 10 |
| `SELECT COUNT(*) FROM employee_records WHERE is_active=1` | 9 |
| `SELECT department FROM employee_records WHERE email='priya@corp.com'` | 'Engineering' |
| `SELECT salary FROM employee_records WHERE email='ravi@corp.com'` | 150000.0 |
| `SELECT COUNT(DISTINCT department) FROM employee_records` | 4 |
| `SELECT COUNT(*) FROM employees WHERE manager_id IS NULL` | 3 |

**Max steps:** 35  
**Baseline score:** ~0.55 (Qwen2.5-72B)

---

## Action Space

| Action | SQL Required | Description |
|--------|-------------|-------------|
| `write_migration` | ✅ | Stage SQL in the migration buffer (syntax-checked before staging) |
| `execute` | ❌ | Execute the staged buffer against the live database |
| `rollback` | ❌ | Restore DB to initial state (penalized, max 5 allowed) |
| `inspect_schema` | ❌ | Read current schema — all tables, columns, types, FKs, row counts |
| `run_query` | ✅ (SELECT) | Execute a read-only query to verify data |
| `submit` | ❌ | Finalize episode, run authoritative grader, receive final score |

**Action format (JSON):**
```json
{"action_type": "write_migration", "sql": "ALTER TABLE products ADD COLUMN stock_quantity INTEGER NOT NULL DEFAULT 0"}
{"action_type": "execute"}
{"action_type": "run_query", "sql": "SELECT COUNT(*) FROM products"}
{"action_type": "submit"}
```

---

## Observation Space

```python
MigrationObservation:
  current_schema:      List[TableSchema]      # Live DB schema snapshot
  migration_spec:      str                    # Natural-language migration description
  requirements:        List[str]              # Checklist of required outcomes
  migration_buffer:    str                    # SQL staged but not yet executed
  execution_history:   List[ExecutionResult]  # Full action history
  last_result:         ExecutionResult        # Result of the most recent action
  step:                int                    # Current step (0-indexed)
  max_steps:           int                    # Step budget for this task
  partial_score:       float                  # Cumulative reward so far [0.0, 1.0]
  hint:                Optional[str]          # Contextual hint when agent appears stuck
```

---

## Reward Function

Dense per-step rewards provide continuous feedback throughout the trajectory:

| Action | Reward | Rationale |
|--------|--------|-----------|
| `write_migration` (valid SQL) | +0.02 | Reward for making measurable progress |
| `write_migration` (syntax error) | -0.03 | Penalize wasted steps |
| `execute` (success) | +0.05 | Reward for actually changing the database |
| `execute` (failure) | -0.05 | Penalize errors that waste budget |
| `rollback` | -0.03 × count | Increasing penalty — discourages loop behavior |
| `rollback` (>5 times) | -0.10 | Hard penalty for excessive rollback loops |
| `inspect_schema` / `run_query` | 0.0 | Free information — always safe to use |
| `submit` | grader_score | Authoritative score overwrites cumulative |
| Budget exceeded | -0.05 | Penalty for running out of steps |

**Why it cannot be gamed:** The `submit` reward is computed by running the actual migration grader against the live SQLite database. The database either contains the correct schema and data, or it does not. No amount of prompt engineering can make `PRAGMA foreign_key_check` return zero violations when FK violations exist.

---

## Grader Design

Each task uses an **execution-based grader** — a Python function that queries the live SQLite database and computes a normalized score from 0.0 to 1.0.

### Easy grader components:
- Row count preserved (5 rows) — 1.5 pts
- Each new column present (×3) — 1.0 pt each
- Original columns intact (×3) — 0.5 pt each
- Correct default values — 0.5 pt each

### Medium grader components:
- Tables created (×3) — 0.5 pt each
- Customer count correct (3) — 1.5 pts
- Product count correct (5) — 1.5 pts
- Order count preserved (7) — 2.0 pts
- FK columns in orders (×2) — 0.5 pt each
- FK integrity (zero violations) — 1.5 pts
- No duplicate customers — 0.5 pts

### Hard grader components:
- New tables exist (×3) — 0.5 pt each
- Department count (4) — 1.0 pt
- Job title count (9) — 1.0 pt
- Employee count (10) — 1.5 pts
- FK integrity (zero violations) — 1.0 pt
- 6 test queries via compatibility view — 2.0 pts (split equally)
- `employee_records` is a VIEW (not table) — 1.0 pt

---

## Setup & Usage

### Local development

```bash
# Clone and install
git clone https://huggingface.co/spaces/RoshanSingh/schema-migration-openenv
cd schema-migration-openenv
pip install -r requirements.txt

# Start the server
python server.py
# Server running at http://localhost:7860
```

### Docker

```bash
# Build
docker build -t schema-migration-openenv .

# Run
docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  schema-migration-openenv
```

### API usage

```bash
# Reset to easy task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "add_columns"}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "write_migration", "sql": "ALTER TABLE products ADD COLUMN stock_quantity INTEGER NOT NULL DEFAULT 0"}'

# Check current state
curl http://localhost:7860/state

# Get all available tasks
curl http://localhost:7860/tasks
```

### Run inference (baseline)

The default LLM provider is **Groq** (free tier, no credit card needed).
Get your key at https://console.groq.com → API Keys.

```bash
# Groq (default, free)
export GROQ_API_KEY=gsk_your_groq_key
export RUN_ALL_TASKS=1
python inference.py

# Single task only
export GROQ_API_KEY=gsk_your_groq_key
export TASK_NAME=normalize_orders
python inference.py

# HuggingFace router (alternative)
export HF_TOKEN=your_hf_token
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

### Run tests

```bash
python -m pytest tests/ -v
```

---

## Baseline Performance Scores

Evaluated with `Qwen/Qwen2.5-72B-Instruct` via HuggingFace router:

| Task | Difficulty | Score | Steps Used | Notes |
|------|-----------|-------|-----------|-------|
| `add_columns` | Easy | 0.867 | 8 | Correctly adds all 3 columns with defaults |
| `normalize_orders` | Medium | 0.721 | 18 | Occasionally misses FK constraints |
| `refactor_employees` | Hard | 0.547 | 31 | Struggles with self-referential manager_id |
| **Average** | | **0.712** | | |

---

## Project Structure

```
schema-migration-openenv/
├── server.py                    # FastAPI HTTP server (OpenEnv endpoints)
├── inference.py                 # Baseline inference script
├── openenv.yaml                 # OpenEnv metadata and spec
├── Dockerfile                   # Container definition
├── requirements.txt
├── README.md
├── env/
│   ├── __init__.py
│   ├── models.py                # Pydantic models: Action, Observation, Reward
│   ├── database.py              # SQLite engine (in-memory, FK checks, snapshots)
│   └── environment.py           # SchemaMigrationEnv (reset/step/state)
├── tasks/
│   ├── __init__.py
│   └── task_definitions.py     # All 3 tasks + graders
└── tests/
    └── test_environment.py      # Pytest suite
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | ✅ (recommended) | — | Groq API key — free at console.groq.com |
| `HF_TOKEN` | ✅ (alternative) | — | HuggingFace API token (fallback if no GROQ_API_KEY) |
| `API_BASE_URL` | ❌ | `https://api.groq.com/openai/v1` | LLM API endpoint |
| `MODEL_NAME` | ❌ | `llama-3.3-70b-versatile` | Model identifier |
| `TASK_NAME` | ❌ | `add_columns` | Task to run in inference |
| `RUN_ALL_TASKS` | ❌ | `false` | Set to `1` to run all 3 tasks |
| `ENV_BASE_URL` | ❌ | `http://localhost:7860` | OpenEnv server URL |
| `PORT` | ❌ | `7860` | Server port |

---

---

## Stability & Bug Fixes

The following edge cases are explicitly handled:

| Fix | Detail |
|-----|--------|
| `sqlite_sequence` excluded from snapshots | SQLite's internal autoincrement table is now filtered from `snapshot_sql()` and `get_schema()` — prevents rollback failures on AUTOINCREMENT tables |
| Escaped SQL quotes (`''`) in splitter | `_split_statements` now handles `'can''t'`-style escaped quotes without splitting mid-string |
| WAL journal mode removed | `PRAGMA journal_mode = WAL` is incompatible with in-memory SQLite and was silently ignored; removed to avoid confusion |
| Grader row-count truthiness | Hard grader partial-credit checks now use `is not None` instead of bare truthiness — prevents `-1` (table missing) from being treated as a non-zero count |
| Cumulative reward not clamped mid-episode | Step rewards now accumulate freely; `partial_score` is clamped to `[0.0, 1.0]` only when surfaced in observations |
| Forced submit always fires | Inference script uses a `submitted` flag so the final graded submit is guaranteed even when the agent runs out of steps mid-migration |
| Pydantic v2 Config syntax | `class Config: use_enum_values` updated to `model_config = ConfigDict(use_enum_values=True)` |
| Dead code removed | Unused `cols` dict computation removed from easy grader |
| `Dict` import moved to top | Late `from typing import Dict` at bottom of `task_definitions.py` moved to the standard imports block |

---

## License

MIT
