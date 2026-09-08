"""
api/pipeline_runner.py
========================
Runs the real stage scripts (blue_team_pipeline.py, cascade_with_graph.py,
...) as actual subprocesses, in order, in a background thread -- so the
API request that starts a run returns immediately with a job_id, and the
frontend polls /api/pipeline/status/{job_id} for live progress.

This intentionally shells out to `python3 <script>` from REPO_ROOT with
PYTHONPATH=src set, exactly as every script's own docstring says to run
it, rather than importing each module. These scripts were written as
standalone CLI programs (argparse-free but full of module-level side
effects, __main__ guards, and relative-path assumptions) -- subprocess
isolation is what actually matches how you already verified them by hand.
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
import uuid
from typing import Optional

import job_store
from config import REPO_ROOT, SRC_DIR, STAGES

# Job state is persisted via job_store (SQLite) rather than kept only in
# this process's memory -- see job_store.py's module docstring for why
# (in short: the previous in-memory-only dict lost all job history on
# every process restart, which happens routinely on a real deployment).
# _LOCK still guards the read-modify-write sequence in _update_stage
# against concurrent stage-completion callbacks; job_store has its own
# internal lock around the actual DB writes.
_LOCK = threading.Lock()

LOG_TAIL_LINES = 40


def _new_stage_state(stage: dict) -> dict:
    return {
        "id": stage["id"],
        "label": stage["label"],
        "script": stage["script"],
        "status": "pending",  # pending | running | success | failed | skipped
        "started_at": None,
        "finished_at": None,
        "returncode": None,
        "log_tail": [],
    }


def get_job(job_id: str) -> Optional[dict]:
    return job_store.load_job(job_id)


def list_jobs() -> list[dict]:
    return job_store.list_jobs()


def start_run(stage_ids: Optional[list[str]] = None) -> str:
    """Kick off a background run of the given stage ids (default: all).
    Returns a job_id immediately; the run happens in a daemon thread."""
    selected = [s for s in STAGES if stage_ids is None or s["id"] in stage_ids]
    if not selected:
        raise ValueError("No matching stage ids")

    job_id = uuid.uuid4().hex[:12]
    job = {
        "job_id": job_id,
        "status": "running",
        "started_at": time.time(),
        "finished_at": None,
        "stages": [_new_stage_state(s) for s in selected],
    }
    job_store.save_job(job)

    thread = threading.Thread(target=_run_job, args=(job_id, selected), daemon=True)
    thread.start()
    return job_id


def _update_stage(job_id: str, idx: int, **kwargs) -> None:
    with _LOCK:
        job = job_store.load_job(job_id)
        job["stages"][idx].update(kwargs)
        job_store.save_job(job)


def _run_job(job_id: str, selected: list[dict]) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC_DIR) + os.pathsep + env.get("PYTHONPATH", "")

    overall_ok = True
    for idx, stage in enumerate(selected):
        if not overall_ok:
            _update_stage(job_id, idx, status="skipped")
            continue

        _update_stage(job_id, idx, status="running", started_at=time.time())
        try:
            proc = subprocess.run(
                ["python3", stage["script"]],
                cwd=str(REPO_ROOT),
                env=env,
                capture_output=True,
                text=True,
                timeout=stage.get("est_seconds", 120) * 6 + 60,  # generous ceiling
            )
            tail = (proc.stdout.strip().splitlines() or [])[-LOG_TAIL_LINES:]
            err_tail = (proc.stderr.strip().splitlines() or [])[-LOG_TAIL_LINES:]
            ok = proc.returncode == 0
            _update_stage(
                job_id, idx,
                status="success" if ok else "failed",
                finished_at=time.time(),
                returncode=proc.returncode,
                log_tail=tail + (["--- stderr ---"] + err_tail if err_tail else []),
            )
            overall_ok = overall_ok and ok
        except subprocess.TimeoutExpired:
            _update_stage(
                job_id, idx,
                status="failed",
                finished_at=time.time(),
                returncode=None,
                log_tail=["TIMEOUT: stage exceeded its time budget"],
            )
            overall_ok = False
        except Exception as exc:  # noqa: BLE001 -- surface any error to the dashboard
            _update_stage(
                job_id, idx,
                status="failed",
                finished_at=time.time(),
                returncode=None,
                log_tail=[f"EXCEPTION: {exc!r}"],
            )
            overall_ok = False

    with _LOCK:
        job = job_store.load_job(job_id)
        job["status"] = "success" if overall_ok else "failed"
        job["finished_at"] = time.time()
        job_store.save_job(job)
