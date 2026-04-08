"""
inference.py — Baseline inference script for Schema Migration OpenEnv.

Uses the OpenAI API client (via HuggingFace router or any OpenAI-compatible endpoint).
Emits structured stdout logs in the mandatory [START] / [STEP] / [END] format.

Environment variables:
  HF_TOKEN      — HuggingFace API token (required)
  API_BASE_URL  — LLM endpoint (default: https://router.huggingface.co/v1)
  MODEL_NAME    — Model identifier (default: Qwen/Qwen2.5-72B-Instruct)
  TASK_NAME     — Task to run: add_columns | normalize_orders | refactor_employees
  ENV_BASE_URL  — OpenEnv server URL (default: http://localhost:7860)

Run:
  python inference.py
  TASK_NAME=normalize_orders python inference.py
  TASK_NAME=refactor_employees python inference.py
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

# Load .env file if present (so you don't have to `set` vars every terminal)
try:
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(_env_path):
        with open(_env_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _, _v = _line.partition("=")
                    os.environ.setdefault(_k.strip(), _v.strip())
except Exception:
    pass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Auto-detect provider from environment variables.
# Priority: GROQ > OPENROUTER > HF_TOKEN/API_KEY
# Override any value with the explicit env vars API_BASE_URL / MODEL_NAME / API_KEY.
if os.getenv("GROQ_API_KEY"):
    API_KEY      = os.getenv("GROQ_API_KEY", "")
    API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
    MODEL_NAME   = os.getenv("MODEL_NAME",   "llama-3.3-70b-versatile")
elif os.getenv("OPENROUTER_API_KEY"):
    API_KEY      = os.getenv("OPENROUTER_API_KEY", "")
    API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")
    MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/llama-3.3-70b-instruct:free")
else:
    API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME    = os.getenv("TASK_NAME", "add_columns")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK    = "schema-migration-openenv"

# Step budgets per task (must finish within 20 min total across all tasks)
MAX_STEPS: Dict[str, int] = {
    "add_columns":       12,
    "normalize_orders":  20,
    "refactor_employees": 28,
}

TEMPERATURE  = 0.2   # Low temperature for SQL generation — we want deterministic output
MAX_TOKENS   = 800
SUCCESS_THRESHOLD = 0.7


# ---------------------------------------------------------------------------
# Mandatory log functions — exact format required by evaluation system
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Environment client
# ---------------------------------------------------------------------------

def env_reset(task: str) -> Dict[str, Any]:
    resp = requests.post(f"{ENV_BASE_URL}/reset", json={"task": task}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_step(action_type: str, sql: Optional[str] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"action_type": action_type}
    if sql:
        payload["sql"] = sql
    resp = requests.post(f"{ENV_BASE_URL}/step", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = textwrap.dedent("""
    You are an expert database engineer performing SQL schema migrations.
    You interact with a live SQLite database through a step-by-step API.

    AVAILABLE ACTIONS (respond with JSON, exactly one action per turn):
    {
      "action_type": "write_migration",
      "sql": "<your SQL here>"
    }
    OR one of these (no sql field needed):
    {"action_type": "execute"}
    {"action_type": "rollback"}
    {"action_type": "inspect_schema"}
    {"action_type": "run_query", "sql": "SELECT ..."}
    {"action_type": "submit"}

    WORKFLOW:
    1. Read the migration_spec and requirements carefully.
    2. Use inspect_schema or run_query to understand the current state.
    3. Write your migration SQL with write_migration (one logical block at a time).
    4. Execute it with execute to apply it to the live DB.
    5. Verify with run_query that the results are correct.
    6. When confident, call submit to finalize and receive your score.

    IMPORTANT RULES:
    - Always check your work with run_query before submitting.
    - SQLite syntax: use ADD COLUMN (not ADD COLUMNS), TEXT/INTEGER/REAL types.
    - For FK constraints in SQLite, you must CREATE the new table with FKs (ALTER TABLE cannot add FKs).
    - To replace a table: create new_table, INSERT INTO new_table SELECT ... FROM old_table, DROP TABLE old_table.
    - To create a VIEW: CREATE VIEW view_name AS SELECT ...
    - Respond with ONLY the JSON action object. No explanation, no markdown, just the JSON.
""").strip()


def build_user_prompt(obs: Dict[str, Any], step: int, last_reward: float) -> str:
    schema_lines = []
    for tbl in obs.get("current_schema", []):
        cols = ", ".join(
            f"{c['name']} {c['type']}{'?' if c['nullable'] else ''}"
            + (" PK" if c.get('primary_key') else "")
            + (f" FK→{c['foreign_key']}" if c.get('foreign_key') else "")
            for c in tbl["columns"]
        )
        schema_lines.append(f"  {tbl['name']}({cols}) [{tbl['row_count']} rows]")

    last_result = obs.get("last_result")
    last_msg = last_result["message"][:300] if last_result else "None"
    last_qr = ""
    if last_result and last_result.get("query_result"):
        last_qr = f"\nQuery result: {json.dumps(last_result['query_result'][:5])}"

    buffer = obs.get("migration_buffer", "")
    hint = obs.get("hint", "")
    reqs = "\n".join(f"  - {r}" for r in obs.get("requirements", []))

    return textwrap.dedent(f"""
        STEP {step} | Partial score so far: {obs.get('partial_score', 0):.3f} | Last reward: {last_reward:.2f}
        Steps remaining: {obs.get('max_steps', 20) - step}

        MIGRATION SPEC:
        {obs.get('migration_spec', '')}

        REQUIREMENTS:
        {reqs}

        CURRENT DATABASE SCHEMA:
        {chr(10).join(schema_lines) or '  (empty)'}

        MIGRATION BUFFER (staged but not executed):
        {buffer or '  (empty)'}

        LAST ACTION RESULT:
        {last_msg}{last_qr}

        {('HINT: ' + hint) if hint else ''}

        What is your next action? Respond with JSON only.
    """).strip()


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def parse_action(response_text: str) -> Dict[str, Any]:
    """Extract JSON action from model response. Handles common formatting issues."""
    text = response_text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(
            l for l in lines
            if not l.startswith("```")
        ).strip()
    # Find first { ... }
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    # Fallback: inspect schema if we can't parse
    return {"action_type": "inspect_schema"}


def run_task(client: OpenAI, task_name: str) -> float:
    """Run one full episode. Returns the final score."""
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    # Reset environment
    obs = env_reset(task_name)
    max_steps = MAX_STEPS.get(task_name, 20)
    rewards: List[float] = []
    last_reward = 0.0
    final_score = 0.0
    submitted = False
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    fatal = False
    for step in range(1, max_steps + 1):
        user_prompt = build_user_prompt(obs, step, last_reward)
        messages.append({"role": "user", "content": user_prompt})

        # Trim history: keep system prompt + last 8 messages (4 exchanges)
        # Prevents token usage from growing unboundedly across long episodes
        if len(messages) > 9:
            messages = [messages[0]] + messages[-8:]

        # Get action from model — retry up to 3× on transient rate limits (429)
        error_msg: Optional[str] = None
        action_dict: Dict[str, Any] = {"action_type": "inspect_schema"}
        for attempt in range(4):  # 1 try + 3 retries
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                )
                raw_response = completion.choices[0].message.content or ""
                messages.append({"role": "assistant", "content": raw_response})
                action_dict = parse_action(raw_response)
                error_msg = None
                break  # success
            except Exception as exc:
                exc_str = str(exc)
                error_msg = exc_str[:200]
                # Hard-fatal: quota exhausted / auth failure — stop immediately
                if any(c in exc_str for c in ("402", "401", "403", "insufficient_quota", "depleted")):
                    print(f"[FATAL] LLM API error: {error_msg}", flush=True)
                    action_dict = {"action_type": "__fatal__"}
                    fatal = True
                    break
                # Soft rate-limit (429): backoff and retry
                if "429" in exc_str and attempt < 3:
                    wait = 5 * (2 ** attempt)  # 5s, 10s, 20s
                    print(f"[RATELIMIT] 429 on attempt {attempt+1}, waiting {wait}s...", flush=True)
                    time.sleep(wait)
                    continue
                # Other error or retries exhausted
                action_dict = {"action_type": "inspect_schema"}
                break

        if action_dict.get("action_type") == "__fatal__":
            break

        action_type = action_dict.get("action_type", "inspect_schema")
        sql = action_dict.get("sql")

        # Execute action
        try:
            result = env_step(action_type, sql)
            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            obs = result.get("observation", obs)
            last_reward = reward
            rewards.append(reward)

            # Extract score from final submission
            if done and action_type == "submit":
                final_score = obs.get("partial_score", 0.0)
                submitted = True

            log_step(
                step=step,
                action=action_type,
                reward=reward,
                done=done,
                error=error_msg,
            )

            if done:
                break

        except Exception as exc:
            error_msg = str(exc)[:100]
            log_step(step=step, action=action_type, reward=0.0, done=False, error=error_msg)
            rewards.append(0.0)

    # If agent didn't explicitly submit (budget exceeded or loop exited early), force one.
    # But NOT on fatal auth/quota errors — the env was just reset and has no progress;
    # force-submitting would score a stale previous episode.
    if not submitted and not fatal:
        try:
            result = env_step("submit")
            final_score = result.get("observation", {}).get("partial_score", 0.0)
        except Exception:
            pass

    success = final_score >= SUCCESS_THRESHOLD
    log_end(
        success=success,
        steps=len(rewards),
        score=final_score,
        rewards=rewards,
    )
    return final_score


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not API_KEY:
        print("ERROR: Set GROQ_API_KEY (recommended, free) or HF_TOKEN.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    # Determine which task(s) to run
    run_all = os.getenv("RUN_ALL_TASKS", "").lower() in ("1", "true", "yes")
    tasks_to_run = list(MAX_STEPS.keys()) if run_all else [TASK_NAME]

    scores: Dict[str, float] = {}
    for task in tasks_to_run:
        print(f"\n{'='*60}", flush=True)
        print(f"Running task: {task}", flush=True)
        print(f"{'='*60}", flush=True)
        score = run_task(client, task)
        scores[task] = score
        print(f"\nTask '{task}' final score: {score:.3f}", flush=True)
        time.sleep(3)  # Brief pause between tasks to let rate limits breathe

    print(f"\n{'='*60}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    for task, score in scores.items():
        status = "PASS" if score >= SUCCESS_THRESHOLD else "FAIL"
        print(f"  {task:30s} {score:.3f}  [{status}]", flush=True)
    if scores:
        avg = sum(scores.values()) / len(scores)
        print(f"\n  {'Average':30s} {avg:.3f}", flush=True)


if __name__ == "__main__":
    main()
