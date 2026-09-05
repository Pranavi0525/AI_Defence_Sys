#!/usr/bin/env python3
"""
web_prototype/run_api.py
=========================
CWD-INDEPENDENT launcher for the FastAPI backend.

The bug this fixes: `api/app.py` and `api/README.md` document
    PYTHONPATH=src python3 -m uvicorn api.app:app --reload --port 8000
run "from the repo root". That command does NOT work from the repo
root, and never did -- `api` is a package under `web_prototype/`, not
a top-level package at the repo root, so `import api.app` fails with
`ModuleNotFoundError: No module named 'api'` unless the process's cwd
(and therefore sys.path[0]) happens to already be `web_prototype/`.
This was reproduced and confirmed, not assumed (see
reports/FINAL_VALIDATION_REPORT.md, section 5, which flagged the same
discrepancy but deliberately left it unfixed).

This script removes the cwd dependency entirely: it resolves every
path it needs from its own file location, so it works identically no
matter where it's invoked from.

Usage (from anywhere):
    python3 /path/to/AI_Defence_Sys-main/web_prototype/run_api.py
    python3 web_prototype/run_api.py                 # from repo root
    cd web_prototype && python3 run_api.py            # old style, still fine
    python3 ../web_prototype/run_api.py                # from a sibling dir

Then open http://localhost:8000/docs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent          # .../web_prototype
API_DIR = THIS_DIR / "api"                           # .../web_prototype/api
REPO_ROOT = THIS_DIR.parent                          # repo root
SRC_DIR = REPO_ROOT / "src"

# Put api/ on sys.path so `import app` (api/app.py) and its sibling
# modules (config, pipeline_runner, reports) resolve the same way
# app.py's own sys.path.insert already makes them resolve for each
# other -- and put src/ on sys.path for the pipeline modules the API
# reports on.
for p in (str(API_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn's auto-reload (development only).",
    )
    args = parser.parse_args()

    import uvicorn  # deferred: fail with a clear message below if missing

    # Import the app object directly (not by dotted module string) so
    # this works regardless of what sys.path[0] happens to be -- no
    # reliance on `api.app:app` or `app:app` resolving correctly.
    import app as api_app_module  # api/app.py, found via API_DIR on sys.path

    uvicorn.run(
        api_app_module.app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as exc:  # pragma: no cover - operator feedback path
        missing = getattr(exc, "name", str(exc))
        print(
            f"[run_api.py] Missing dependency: {missing!r}. "
            f"Run `pip install -r requirements.txt` from the repo root "
            f"({REPO_ROOT}) first.",
            file=sys.stderr,
        )
        sys.exit(1)
