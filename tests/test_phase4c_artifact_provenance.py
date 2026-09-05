"""
Regression tests for Phase 4C: artifact provenance, cache-variant safety,
and consistency-check correctness.

FINDINGS COVERED (see reports/ Phase 4C audit):

  B-2: decision_policy_validation_cache.npz was shared, unversioned, and
       silently overwritten between get_validation_data() ("cascade"
       score) and get_validation_data_fused() ("fused" score) writes.
       A reader wanting one variant could silently receive the other,
       since the only prior check (y-array alignment) is identical for
       both variants -- same labels, different scores.

  B-3: misses.jsonl and case_reports.json carried no artifact metadata
       (git commit / dirty state / etc.), unlike decision_policy_results.json,
       risk_fusion_results.json, and results.json, which all call
       artifact_metadata.stamp_artifact().

  B-1/B-4 (consistency_check.py bugs found during Phase 4C Step 1
  inspection): CASE_REPORTS_PATH pointed at a stale root-level
  case_reports.json (frozen at commit 52c8101, pre-dating
  explainability.py's current OUT_DIR) instead of
  blue_team_output/explainability/case_reports.json -- guaranteeing a
  mismatch independent of the cache bug.

These tests exercise the REAL production functions in decision_policy.py,
miss_collector.py, explainability.py, and consistency_check.py -- not
reimplementations of their logic.

NOTE: importing decision_policy.py (and therefore this test module)
requires the project's full pinned stack (xgboost, torch, shap, etc. --
see requirements.txt) because decision_policy.py imports
blue_team_pipeline / cascade_with_graph at module load time. This is the
same requirement every existing decision_policy test already has (e.g.
tests/test_decision_policy_nested_threshold.py); nothing new is
introduced here.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

import decision_policy as dp
import consistency_check as cc


# ---------------------------------------------------------------------------
# 1. Cache variant safety (B-2)
# ---------------------------------------------------------------------------
def _write_cache(path, *, y, proba, dollars, variant):
    """Writes a cache file the same way get_validation_data[_fused] does,
    without pulling in the heavy cascade to produce y/proba/dollars."""
    if variant is None:
        # Simulates a pre-Phase-4C cache with no variant tag at all.
        np.savez(path, y=y, proba=proba, dollars=dollars)
    else:
        np.savez(path, y=y, proba=proba, dollars=dollars, validation_variant=variant)


@pytest.fixture
def fake_cache(tmp_path, monkeypatch):
    """Points decision_policy.CACHE_PATH at a throwaway file for the
    duration of the test, so these tests never touch the real
    decision_policy_validation_cache.npz."""
    cache_path = tmp_path / "decision_policy_validation_cache.npz"
    monkeypatch.setattr(dp, "CACHE_PATH", cache_path)
    return cache_path


def test_cascade_cache_not_returned_as_fused(fake_cache):
    """A cache written with validation_variant="cascade" must never be
    silently handed back to a caller that asked for "fused"."""
    y = np.array([0, 1, 0, 1])
    _write_cache(fake_cache, y=y, proba=np.array([0.1, 0.9, 0.2, 0.8]),
                 dollars=np.array([10.0, 20.0, 30.0, 40.0]), variant="cascade")

    with pytest.raises(dp.ValidationCacheMismatch, match="cascade"):
        dp.load_cached_validation_data("fused")


def test_fused_cache_not_returned_as_cascade(fake_cache):
    """And the reverse: a "fused" cache must never satisfy a "cascade"
    request."""
    y = np.array([0, 1, 0, 1])
    _write_cache(fake_cache, y=y, proba=np.array([0.15, 0.95, 0.25, 0.85]),
                 dollars=np.array([10.0, 20.0, 30.0, 40.0]), variant="fused")

    with pytest.raises(dp.ValidationCacheMismatch, match="fused"):
        dp.load_cached_validation_data("cascade")


def test_matching_variant_is_returned(fake_cache):
    """The non-adversarial path still works: requesting the variant that
    is actually cached succeeds and returns the exact arrays written."""
    y = np.array([0, 1, 1, 0])
    proba = np.array([0.11, 0.91, 0.71, 0.05])
    dollars = np.array([1.0, 2.0, 3.0, 4.0])
    _write_cache(fake_cache, y=y, proba=proba, dollars=dollars, variant="fused")

    y_out, proba_out, dollars_out = dp.load_cached_validation_data("fused")
    assert np.array_equal(y_out, y)
    assert np.array_equal(proba_out, proba)
    assert np.array_equal(dollars_out, dollars)


def test_untagged_legacy_cache_is_rejected(fake_cache):
    """A pre-Phase-4C cache with no validation_variant field at all must
    be treated as unusable, not silently assumed to be either variant."""
    y = np.array([0, 1])
    _write_cache(fake_cache, y=y, proba=np.array([0.3, 0.7]),
                 dollars=np.array([5.0, 6.0]), variant=None)

    with pytest.raises(dp.ValidationCacheMismatch, match="no 'validation_variant'"):
        dp.load_cached_validation_data("fused")
    with pytest.raises(dp.ValidationCacheMismatch, match="no 'validation_variant'"):
        dp.load_cached_validation_data("cascade")


def test_missing_cache_file_is_rejected(fake_cache):
    """fake_cache fixture only sets the path; it does not create the
    file, so this exercises the "cache doesn't exist yet" path."""
    assert not fake_cache.exists()
    with pytest.raises(dp.ValidationCacheMismatch, match="does not exist"):
        dp.load_cached_validation_data("fused")


def test_stale_labels_are_rejected_even_with_matching_variant(fake_cache):
    """Right variant, wrong dataset: y_check catches a cache left over
    from a different df (e.g. different row count) even when the variant
    tag matches what's requested."""
    cached_y = np.array([0, 1, 0])
    _write_cache(fake_cache, y=cached_y, proba=np.array([0.1, 0.9, 0.2]),
                 dollars=np.array([1.0, 2.0, 3.0]), variant="fused")

    current_y = np.array([0, 1, 0, 1])  # different length -> stale
    with pytest.raises(dp.ValidationCacheMismatch, match="don't match the current dataset"):
        dp.load_cached_validation_data("fused", y_check=current_y)


def test_invalid_variant_argument_rejected(fake_cache):
    """load_cached_validation_data only accepts the two known variants --
    a typo'd caller should fail loudly, not silently pass through."""
    with pytest.raises(ValueError):
        dp.load_cached_validation_data("fuzed")


# ---------------------------------------------------------------------------
# 2. Artifact metadata presence (B-3)
# ---------------------------------------------------------------------------
def test_misses_jsonl_metadata_helper_produces_valid_stamp(tmp_path, monkeypatch):
    """miss_collector._stamp_misses_metadata() must produce a dict with
    the same stamp_artifact() shape used elsewhere in the repo (git
    commit / dirty / package versions), and it must round-trip through
    consistency_check's loaders as a header line, not a miss record."""
    import miss_collector as mc

    header = mc._stamp_misses_metadata()
    assert "_artifact_metadata" in header
    meta = header["_artifact_metadata"]
    assert "git_commit" in meta
    assert "git_dirty" in meta

    out_path = tmp_path / "misses.jsonl"
    fake_misses = [
        {"trace_id": "t1", "t_review": 0.1, "t_block": 0.9, "final_decision": "ALLOW", "final_score": 0.05},
        {"trace_id": "t2", "t_review": 0.1, "t_block": 0.9, "final_decision": "ALLOW", "final_score": 0.07},
    ]
    mc.write_misses_jsonl(fake_misses, out_path)

    # consistency_check's data loader must return exactly the 2 real
    # records -- the header line must not be mistaken for a miss.
    records = cc._load_jsonl(out_path)
    assert len(records) == 2
    assert {r["trace_id"] for r in records} == {"t1", "t2"}

    # And the metadata loader must recover the same stamp.
    loaded_meta = cc._load_jsonl_metadata(out_path)
    assert loaded_meta is not None
    assert loaded_meta["git_commit"] == meta["git_commit"]


def test_case_reports_stamping_is_additive(tmp_path):
    """Stamping case_reports.json must not remove/alter any existing case
    entry, and consistency_check's _case_items() must skip the metadata
    key when iterating cases."""
    from artifact_metadata import stamp_artifact

    original = {
        "easy_ato_correctly_blocked": {"trace_id": "abc", "stage4": {"decision": "BLOCK"}},
        "fraud_case_that_slipped_through_as_allow": {"trace_id": "xyz", "stage4": {"decision": "ALLOW"}},
    }
    stamped = stamp_artifact(dict(original), tmp_path)

    assert stamped["easy_ato_correctly_blocked"] == original["easy_ato_correctly_blocked"]
    assert stamped["fraud_case_that_slipped_through_as_allow"] == original["fraud_case_that_slipped_through_as_allow"]
    assert "_artifact_metadata" in stamped

    names = {name for name, _ in cc._case_items(stamped)}
    assert names == set(original.keys())
    assert "_artifact_metadata" not in names


# ---------------------------------------------------------------------------
# 3 & 4. consistency_check.py path fix + mismatch detection
# ---------------------------------------------------------------------------
def test_case_reports_path_points_at_real_explainability_output():
    """B-1 root-cause fix: CASE_REPORTS_PATH must point at
    explainability.py's actual OUT_DIR, not the stale repo-root file."""
    assert cc.CASE_REPORTS_PATH == cc.REPO_ROOT / "blue_team_output" / "explainability" / "case_reports.json"
    assert cc.CASE_REPORTS_PATH != cc.REPO_ROOT / "case_reports.json"


def test_provenance_agreement_passes_on_matching_commits():
    decision_policy = {"_artifact_metadata": {"git_commit": "62fdb9a", "git_dirty": False}}
    misses_meta = {"git_commit": "62fdb9a", "git_dirty": False}
    case_reports = {"_artifact_metadata": {"git_commit": "62fdb9a", "git_dirty": False}}

    result = cc.check_provenance_agreement(decision_policy, misses_meta, case_reports)
    assert result["passed"] is True


def test_provenance_agreement_fails_on_mismatched_commits():
    """Deliberately incompatible provenance (Step 5, test 4): artifacts
    stamped with different commits must be caught, not silently accepted."""
    decision_policy = {"_artifact_metadata": {"git_commit": "62fdb9a", "git_dirty": False}}
    misses_meta = {"git_commit": "a5a5403", "git_dirty": False}  # stale on purpose
    case_reports = {"_artifact_metadata": {"git_commit": "62fdb9a", "git_dirty": False}}

    result = cc.check_provenance_agreement(decision_policy, misses_meta, case_reports)
    assert result["passed"] is False
    assert any("different commits" in d for d in result["details"])


def test_provenance_agreement_skips_when_insufficient_data():
    result = cc.check_provenance_agreement(None, None, None)
    assert result["passed"] is None
