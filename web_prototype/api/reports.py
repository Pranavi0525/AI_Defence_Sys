"""
api/reports.py
================
Reads the real pipeline output files off disk, live, on every request.
Nothing here is cached or baked in -- if you re-run a stage, the very
next GET reflects it. Missing files (a stage that hasn't been run yet)
are reported as {"available": False}, not faked with zeros, so the
frontend can honestly show "not run yet" instead of a misleading 0%.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import REPO_ROOT, REPORTS


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def read_json(key: str) -> dict[str, Any]:
    path = REPORTS[key]
    if not path.exists():
        return {"available": False, "source_file": _rel(path)}
    with open(path) as f:
        data = json.load(f)
    return {
        "available": True,
        "source_file": _rel(path),
        "last_modified": _mtime(path),
        "data": data,
    }


def read_jsonl(key: str) -> dict[str, Any]:
    path = REPORTS[key]
    if not path.exists():
        return {"available": False, "source_file": _rel(path), "rows": []}
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return {
        "available": True,
        "source_file": _rel(path),
        "last_modified": _mtime(path),
        "count": len(rows),
        "rows": rows,
    }


def dashboard_summary() -> dict[str, Any]:
    """One aggregated payload the frontend hits on load. Every field is a
    live read of a report key defined in config.REPORTS -- nothing here
    is hardcoded or pre-computed at import time."""
    stage1_2 = read_json("stage1_2_results")
    stage3 = read_json("stage3_graph_results")
    stage4 = read_json("stage4_autoencoder_results")
    fusion = read_json("risk_fusion_results")
    policy = read_json("decision_policy_results")
    sensitivity = read_json("decision_policy_sensitivity_results")
    misses = read_jsonl("misses")
    adaptive = read_json("adaptive_round2_report")
    retrain = read_json("round1_vs_round2_report")
    importance = read_json("global_feature_importance")
    cases = read_json("case_reports")

    return {
        "stage1_2": stage1_2,
        "stage3_graph": stage3,
        "stage4_autoencoder": stage4,
        "risk_fusion": fusion,
        "decision_policy": policy,
        "decision_policy_sensitivity": sensitivity,
        "misses": misses,
        "adaptive_round2": adaptive,
        "round1_vs_round2": retrain,
        "global_feature_importance": importance,
        "case_reports": cases,
    }
