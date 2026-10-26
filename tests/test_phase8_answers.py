"""Phase 8 answers generator: comparisons are computed from the JSON, so branches must follow the numbers."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location("a", ROOT / "scripts" / "build_phase8_answers.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write(tmp, brier_iso=0.00050, ctrl_ap=0.0018, lr_ap=0.684, ci=(0.674, 0.848)):
    mm = lambda ap, br: {"average_precision": ap, "roc_auc": 0.97, "brier": br, "ece_10_quantile_bins": 0.0002, "recall_at_0.1pct_alerts": 0.6,
                         "recall_at_0.5pct_alerts": 0.8, "precision_at_0.5pct_alerts": 0.2, "recall_at_1pct_alerts": 0.85}
    def split(ap, nfr, ci_):
        return {"sizes": {"test": 56962}, "fraud": {"train": 300, "validation": 60, "test": nfr},
                "models": {"xgboost": {"raw": {**mm(ap, 0.00047), "average_precision_ci95": list(ci_)}, "sigmoid": mm(ap, 0.00047), "isotonic": mm(ap - 0.06, brier_iso)},
                           "logreg": {"raw": mm(lr_ap, 0.00067), "sigmoid": mm(lr_ap, 0.00069), "isotonic": mm(lr_ap - 0.02, 0.00062)}}}
    pub = tmp / "reports" / "public"
    pub.mkdir(parents=True)
    (pub / "phase8_main.json").write_text(json.dumps({"data": {"rows": 284807, "fraud": 492, "prevalence": 492 / 284807, "time_range_s": [0, 172792], "notes": []},
                                                      "time_split": split(0.778, 75, ci), "random_split": split(0.881, 99, (0.821, 0.934))}), encoding="utf-8")
    (pub / "phase8_control.json").write_text(json.dumps({"label_shuffle_time_split": {"mean_average_precision": ctrl_ap, "per_repeat": [ctrl_ap] * 3, "test_prevalence": 0.0013}}), encoding="utf-8")
    (pub / "phase8_rolling.json").write_text(json.dumps({"rolling_origin": [{"fold": k, "average_precision": v} for k, v in zip((2, 3, 4, 5), (0.74, 0.82, 0.79, 0.76))]}), encoding="utf-8")


def test_answers_follow_the_numbers(tmp_path):
    _write(tmp_path)
    t = _mod().build(tmp_path)
    assert "no sign of gross label leakage" in t and "did not help XGBoost" in t and "not a clear win" in t
    assert "about the same size as" in t and "suggestive, not shown" in t
    assert "0.263" in t          # 75 / ceil(0.005 * 56962) = 75 / 285


def test_leakage_branch_warns_when_control_is_far_above_base_rate(tmp_path):
    _write(tmp_path, ctrl_ap=0.02)
    assert "FAR ABOVE THE BASE RATE" in _mod().build(tmp_path)


def test_missing_reports_exit_cleanly(tmp_path):
    with pytest.raises(SystemExit):
        _mod().build(tmp_path)
