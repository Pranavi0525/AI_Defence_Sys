"""Shared artifact-metadata stamping helper.

Every JSON artifact this project writes (results.json, risk_fusion_results.json,
decision_policy_results.json, round1_vs_round2_report.json, ...) is a claim
about a specific model trained on a specific dataset with a specific set of
package versions. Without a stamp, "which run produced this file?" can only
be answered by memory or by file-modified-time, neither of which survives a
repo handoff, a re-run six months later, or a judge asking "prove this number
is reproducible."

stamp_artifact() adds an "_artifact_metadata" key to an existing output dict.
It is purely additive: no existing key or value in the dict is touched, and
callers that don't opt in are completely unaffected. This directly answers
outstanding item 5 in reports/FINAL_VALIDATION_REPORT.md ("no artifact
metadata... on any output file").

What gets stamped, and why each field earns its place:
    - generated_at_utc:      when this specific run happened
    - git_commit:            exact code version (falls back to "unknown"
                              outside a git checkout, e.g. after unzipping)
    - git_dirty:             whether the working tree had uncommitted changes
                              when the artifact was generated -- an artifact
                              from a dirty tree is not exactly reproducible
                              from the commit hash alone
    - python_version:        interpreter version
    - package_versions:      versions of the libraries whose behavior could
                              silently change a model's numbers between runs
                              (numpy, pandas, scikit-learn/xgboost, pydantic)
    - dataset_content_hash:  SHA-256 over the actual bytes of the corpus
                              file(s) that fed this run, NOT just a filename
                              -- catches "same filename, different content"
    - feature_schema_hash:   SHA-256 over the ordered FEATURE_COLS list --
                              catches "same model file, different feature
                              contract" silently breaking downstream
                              consumers (risk_fusion, decision_policy, etc.)
    - seeds:                 the RNG seed(s) that determined this run

This module has zero project-internal imports (only stdlib) so it can be
imported from any script -- red team, blue team, retrain -- without risk of
circular imports.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _git_commit(repo_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root, capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown (not a git checkout, or git unavailable)"


def _git_dirty(repo_root: Path) -> bool | str:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root, capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return bool(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for pkg in ("numpy", "pandas", "sklearn", "xgboost", "pydantic"):
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not installed"
    return versions


def hash_file(path: Path) -> str | None:
    """SHA-256 of a file's raw bytes. Returns None if the file doesn't exist,
    so a missing input is visible in the metadata rather than silently
    absent from it."""
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_dataset_files(paths: Iterable[Path]) -> dict[str, str | None]:
    """Content hash per input dataset file, keyed by filename."""
    return {p.name: hash_file(p) for p in paths}


def hash_feature_schema(feature_cols: list[str]) -> str:
    """SHA-256 over the ORDERED feature list, so a reordering (which can
    change column-index-dependent bugs elsewhere) is also caught, not just
    a set-membership change."""
    payload = json.dumps(list(feature_cols), sort_keys=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_metadata(
    repo_root: Path,
    *,
    seeds: dict[str, Any] | None = None,
    dataset_files: Iterable[Path] | None = None,
    feature_cols: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the metadata block. Every argument is optional so callers
    only pay for what's relevant to them (e.g. decision_policy.py has no
    feature_cols of its own to hash)."""
    meta: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(repo_root),
        "git_dirty": _git_dirty(repo_root),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "package_versions": _package_versions(),
    }
    if seeds is not None:
        meta["seeds"] = seeds
    if dataset_files is not None:
        meta["dataset_content_hash"] = hash_dataset_files(dataset_files)
    if feature_cols is not None:
        meta["feature_schema_hash"] = hash_feature_schema(feature_cols)
        meta["feature_schema_n_features"] = len(feature_cols)
    if extra:
        meta.update(extra)
    return meta


def stamp_artifact(
    output: dict[str, Any],
    repo_root: Path,
    *,
    seeds: dict[str, Any] | None = None,
    dataset_files: Iterable[Path] | None = None,
    feature_cols: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return `output` with an added "_artifact_metadata" key. Purely
    additive -- every existing key/value in `output` is untouched.

    Usage (drop-in, right before json.dump):
        output = stamp_artifact(output, REPO_ROOT, feature_cols=FEATURE_COLS)
        json.dump(output, f, indent=2, default=str)
    """
    if "_artifact_metadata" in output:
        # Don't silently clobber an existing stamp from a nested/composed
        # artifact -- keep both, newest under a versioned key.
        output["_artifact_metadata_prior"] = output["_artifact_metadata"]
    output["_artifact_metadata"] = build_metadata(
        repo_root,
        seeds=seeds,
        dataset_files=dataset_files,
        feature_cols=feature_cols,
        extra=extra,
    )
    return output
