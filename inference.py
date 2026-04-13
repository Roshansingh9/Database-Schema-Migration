"""
inference.py - Baseline inference script for schema-migration-openenv.

Hackathon-aligned environment variables:
  HF_TOKEN      - primary API key for Hugging Face / OpenAI-compatible routing
  API_KEY       - optional fallback alias for local testing
  API_BASE_URL  - LLM endpoint (default: https://router.huggingface.co/v1)
  MODEL_NAME    - model identifier (default: Qwen/Qwen2.5-72B-Instruct)
  TASK_NAME     - optional single task override
  ENV_BASE_URL  - OpenEnv server URL (default: http://localhost:7860)

Stdout is limited to the required [START] / [STEP] / [END] lines.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests
from openai import OpenAI

# Load .env file if present.
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

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME = os.getenv("TASK_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK = "schema-migration-openenv"

MAX_STEPS: Dict[str, int] = {
    "add_columns": 12,
    "normalize_orders": 20,
    "refactor_employees": 28,
}

TEMPERATURE = 0.2
MAX_TOKENS = 800
SUCCESS_THRESHOLD = 0.7
SERVER_PROC: Optional[subprocess.Popen] = None


def _is_local_env_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.hostname in {"127.0.0.1", "localhost"}


def _wait_for_server(base_url: str, timeout_s: float = 20.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            resp = requests.get(f"{base_url}/health", timeout=2)
            if resp.ok:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def ensure_local_server() -> Optional[subprocess.Popen]:
    try:
        if requests.get(f"{ENV_BASE_URL}/health", timeout=2).ok:
            return None
    except Exception:
        pass

    if not _is_local_env_url(ENV_BASE_URL):
        return None

    server_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "server.app:app", "--host", "127.0.0.1", "--port", "7860"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if _wait_for_server(ENV_BASE_URL):
        return server_proc

    server_proc.terminate()
    server_proc.wait(timeout=10)
    raise RuntimeError(f"Could not start local environment server at {ENV_BASE_URL}")


def _ensure_running_server() -> None:
    global SERVER_PROC

    if not _is_local_env_url(ENV_BASE_URL):
        return

    try:
        if requests.get(f"{ENV_BASE_URL}/health", timeout=2).ok:
            return
    except Exception:
        pass

    if SERVER_PROC is not None and SERVER_PROC.poll() is None:
        try:
            SERVER_PROC.terminate()
            SERVER_PROC.wait(timeout=10)
        except Exception:
            try:
                SERVER_PROC.kill()
            except Exception:
                pass

    SERVER_PROC = ensure_local_server()


def _request_env(method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> requests.Response:
    for attempt in range(2):
        try:
            if method == "GET":
                resp = requests.get(f"{ENV_BASE_URL}{path}", timeout=30)
            else:
                resp = requests.post(f"{ENV_BASE_URL}{path}", json=payload, timeout=30)
            resp.raise_for_status()
            return resp
        except requests.RequestException:
            if attempt == 0 and _is_local_env_url(ENV_BASE_URL):
                _ensure_running_server()
                continue
            raise

    raise RuntimeError(f"Unable to reach environment endpoint: {path}")


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
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def env_reset(task: str) -> Dict[str, Any]:
    resp = _request_env("POST", "/reset", {"task": task})
    data = resp.json()
    return data.get("observation", data)


def env_step(action_type: str, sql: Optional[str] = None) -> Dict[str, Any]:
    action: Dict[str, Any] = {"action_type": action_type}
    if sql:
        action["sql"] = sql
    resp = _request_env("POST", "/step", {"action": action})
    return resp.json()


def env_grade(task_name: str) -> float:
    try:
        resp = _request_env("POST", "/grade", {})
        return float(resp.json().get("score", 0.0))
    except Exception:
        try:
            from env.database import MigrationDB
            from tasks.task_definitions import TASKS, build_seed_metrics

            task = TASKS[task_name]
            db = MigrationDB()
            db.init(task.seed_sql)
            pre_snapshot = db.snapshot_sql()
            metrics = build_seed_metrics(pre_snapshot)
            score, _, _ = task.grader(db, pre_snapshot, metrics)
            db.close()
            return float(score)
        except Exception:
            return 0.05


SYSTEM_PROMPT = textwrap.dedent(
    """
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
    3. Write your migration SQL with write_migration.
    4. Execute it with execute.
    5. Verify with run_query.
    6. Submit when confident.

    Respond with ONLY the JSON action object.
    """
).strip()


def build_user_prompt(obs: Dict[str, Any], step: int, last_reward: float) -> str:
    schema_lines = []
    for tbl in obs.get("current_schema", []):
        cols = ", ".join(
            f"{c['name']} {c['type']}{'?' if c['nullable'] else ''}"
            + (" PK" if c.get("primary_key") else "")
            + (f" FK->{c['foreign_key']}" if c.get("foreign_key") else "")
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

    return textwrap.dedent(
        f"""
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
        """
    ).strip()


def parse_action(response_text: str) -> Dict[str, Any]:
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(line for line in lines if not line.startswith("```")).strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return {"action_type": "inspect_schema"}


def is_fatal_llm_error(exc_str: str) -> bool:
    fatal_markers = (
        "401",
        "402",
        "403",
        "410",
        "insufficient_quota",
        "depleted",
        "deprecated",
        "no longer supported",
        "invalid_request_error",
        "model_not_found",
        "your request was blocked",
        "blocked",
    )
    lowered = exc_str.lower()
    return any(marker in lowered for marker in fatal_markers)


def run_task(client: OpenAI, task_name: str) -> float:
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

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
        if len(messages) > 9:
            messages = [messages[0]] + messages[-8:]

        error_msg: Optional[str] = None
        action_dict: Dict[str, Any] = {"action_type": "inspect_schema"}
        for attempt in range(4):
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
                break
            except Exception as exc:
                exc_str = str(exc)
                error_msg = exc_str[:200]
                if is_fatal_llm_error(exc_str):
                    print(f"[FATAL] LLM API error: {error_msg}", file=sys.stderr, flush=True)
                    action_dict = {"action_type": "__fatal__"}
                    fatal = True
                    break
                if "429" in exc_str and attempt < 3:
                    wait = 5 * (2 ** attempt)
                    print(f"[RATELIMIT] 429 on attempt {attempt + 1}, waiting {wait}s...", file=sys.stderr, flush=True)
                    time.sleep(wait)
                    continue
                action_dict = {"action_type": "inspect_schema"}
                break

        if action_dict.get("action_type") == "__fatal__":
            break

        action_type = action_dict.get("action_type", "inspect_schema")
        sql = action_dict.get("sql")

        try:
            result = env_step(action_type, sql)
            reward = result.get("reward", 0.0)
            done = result.get("done", False)
            obs = result.get("observation", obs)
            last_reward = reward
            rewards.append(reward)

            if done and action_type == "submit":
                final_score = obs.get("partial_score", 0.0)
                submitted = True

            log_step(step=step, action=action_type, reward=reward, done=done, error=error_msg)
            if done:
                break
        except Exception as exc:
            error_msg = str(exc)[:100]
            log_step(step=step, action=action_type, reward=0.0, done=False, error=error_msg)
            rewards.append(0.0)

    if not submitted and not fatal:
        try:
            result = env_step("submit")
            final_score = result.get("observation", {}).get("partial_score", 0.0)
        except Exception:
            pass
    elif fatal and final_score == 0.0:
        final_score = env_grade(task_name)
        print(f"[INFO] LLM unavailable; reporting seed-state grader score: {final_score:.4f}", file=sys.stderr, flush=True)

    success = final_score >= SUCCESS_THRESHOLD
    log_end(success=success, steps=len(rewards), score=final_score, rewards=rewards)
    return final_score


def main() -> None:
    global SERVER_PROC

    if not API_KEY:
        print("ERROR: Set HF_TOKEN (preferred) or API_KEY.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    if TASK_NAME:
        tasks_to_run = [TASK_NAME]
    else:
        tasks_to_run = list(MAX_STEPS.keys())

    try:
        SERVER_PROC = ensure_local_server()
        for task in tasks_to_run:
            run_task(client, task)
            time.sleep(3)
    finally:
        if SERVER_PROC is not None and SERVER_PROC.poll() is None:
            SERVER_PROC.terminate()
            try:
                SERVER_PROC.wait(timeout=10)
            except subprocess.TimeoutExpired:
                SERVER_PROC.kill()


if __name__ == "__main__":
    main()
