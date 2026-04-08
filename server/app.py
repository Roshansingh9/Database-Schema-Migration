"""
server/app.py — ASGI entry point for multi-mode / package-based deployment.

Exports the FastAPI `app` object so this module works as:
  uvicorn server.app:app --host 0.0.0.0 --port 7860

Also provides a `main()` CLI entry point registered in pyproject.toml:
  schema-migration-openenv
"""

from __future__ import annotations

import importlib.util
import os
import sys

# Ensure the project root is importable (needed when installed as a package)
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

# Load root server.py as a named module — avoids the server/ package shadowing it
_spec = importlib.util.spec_from_file_location(
    "__root_server__",
    os.path.join(_root, "server.py"),
)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

# Re-export the FastAPI application
app = _mod.app


def main() -> None:
    """CLI entry point: start the server (used by project.scripts)."""
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
