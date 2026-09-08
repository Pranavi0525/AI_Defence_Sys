"""
api/job_store.py
==================
Persists pipeline job state (started by POST /api/pipeline/run, polled by
GET /api/pipeline/status/{job_id}) to a SQLite file instead of keeping it
only in a Python dict in process memory.

WHY THIS EXISTS
---------------
The previous implementation (see git history of pipeline_runner.py) kept
`_JOBS` as an in-memory dict with a comment saying "a hackathon demo
prototype does not need Redis -- restart clears it, good enough." That's
true for a laptop demo, but on a real deployment (Render free tier
specifically) the process restarts on every redeploy, and can also be
killed and respawned by the platform. Every one of those events silently
erased all job history and broke any in-flight status polling from the
frontend (a running job's /api/pipeline/status/{job_id} would 404 after a
restart, with no explanation).

SQLite (stdlib, zero new dependencies) fixes the "process restarts, job
history disappears" problem cheaply: state now survives a crash/restart
on the SAME instance and disk. It does NOT survive a full Render
redeploy that wipes the filesystem (Render's free tier has no persistent
disk) -- that would require an external store (S3, hosted Postgres,
Render's paid persistent disk add-on), which is a real infra decision
for whoever runs this in production, not something to fake here. This
is an honest incremental improvement, not a claim of full durability.

DESIGN
------
One row per job_id, storing the full job dict as a JSON blob. This
intentionally does not normalize into a jobs/stages schema -- the job
dict is small, always read/written as a whole, and a single blob keeps
this module trivial to reason about. If this ever needs querying by
stage status across jobs, that's the point to add real columns.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from config import REPO_ROOT

DB_PATH = REPO_ROOT / "web_prototype" / "api" / "pipeline_jobs.sqlite3"

_LOCK = threading.Lock()
_conn: Optional[sqlite3.Connection] = None


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                started_at REAL NOT NULL,
                finished_at REAL,
                status TEXT NOT NULL,
                data TEXT NOT NULL
            )
            """
        )
        _conn.commit()
    return _conn


def save_job(job: dict) -> None:
    """Insert or fully overwrite the row for job['job_id']. Called after
    every stage-status change so a crash mid-run loses at most the one
    in-flight update, not the whole job."""
    conn = _get_conn()
    with _LOCK:
        conn.execute(
            """
            INSERT INTO jobs (job_id, started_at, finished_at, status, data)
            VALUES (:job_id, :started_at, :finished_at, :status, :data)
            ON CONFLICT(job_id) DO UPDATE SET
                finished_at=excluded.finished_at,
                status=excluded.status,
                data=excluded.data
            """,
            {
                "job_id": job["job_id"],
                "started_at": job["started_at"],
                "finished_at": job["finished_at"],
                "status": job["status"],
                "data": json.dumps(job),
            },
        )
        conn.commit()


def load_job(job_id: str) -> Optional[dict]:
    conn = _get_conn()
    with _LOCK:
        row = conn.execute("SELECT data FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    if row is None:
        return None
    return json.loads(row[0])


def list_jobs(limit: int = 50) -> list[dict]:
    conn = _get_conn()
    with _LOCK:
        rows = conn.execute(
            "SELECT job_id, status, started_at, finished_at FROM jobs "
            "ORDER BY started_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        {"job_id": r[0], "status": r[1], "started_at": r[2], "finished_at": r[3]}
        for r in rows
    ]
