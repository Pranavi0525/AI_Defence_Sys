"""Phase 9 tests: the generated docs are fully filled, match the report JSONs and avoid forbidden claims."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_phase9_docs", ROOT / "scripts" / "build_phase9_docs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path_factory.mktemp("docs")
    N = mod.build(out=out, root=ROOT)
    return mod, out, N


def test_files_written_and_fully_filled(built):
    _, out, _ = built
    for name in ("README_V2.md", "INTERVIEW_KIT.md", "reports/v2/results_table.md"):
        text = (out / name).read_text(encoding="utf-8")
        assert text and "@@" not in text


def test_fill_refuses_unfilled_placeholders(built):
    mod, _, _ = built
    with pytest.raises(SystemExit):
        mod.fill("value @@NOT_A_KEY@@", {})


def test_headline_numbers_match_the_report_json(built):
    _, out, _ = built
    R = json.loads((ROOT / "reports" / "v2" / "v2_results.json").read_text(encoding="utf-8"))
    readme = (out / "README_V2.md").read_text(encoding="utf-8")
    assert f"{R['phase1']['world_B']['metrics']['f1']['mean']:.3f}" in readme
    assert f"{R['phase1']['world_A']['metrics']['f1']['mean']:.3f}" in readme
    claim = json.loads((ROOT / "phase5a_evaluation_report.json").read_text(encoding="utf-8"))
    assert f"{claim['classification']['f1']:.3f}" in readme


def test_readme_states_synthetic_and_avoids_overclaims(built):
    _, out, _ = built
    readme = (out / "README_V2.md").read_text(encoding="utf-8").lower()
    assert "synthetic" in readme and "limitations" in readme and "real vs offline vs served" in readme
    for banned in ("production-ready", "state-of-the-art", "state of the art", "validated on real data"):
        assert banned not in readme


def test_kit_has_do_not_say_and_claims_ledger(built):
    _, out, _ = built
    kit = (out / "INTERVIEW_KIT.md").read_text(encoding="utf-8")
    assert "## 6. Do not say" in kit and "## 5. Claims ledger" in kit and "Cold-email templates" in kit
    assert "github.com/Pranavi0525" in kit


def test_phase8_section_absent_without_reports_and_present_with_them(built, tmp_path):
    mod, _, _ = built
    sec, note = mod.phase8_section(tmp_path)
    assert sec == "" and "has not been run" in note
    # minimal fake reports: wording must be chosen from the numbers, and a near-tie must be called a tie
    pub = tmp_path / "reports" / "public"
    pub.mkdir(parents=True)
    m = lambda ap, brier: {"average_precision": ap, "brier": brier, "roc_auc": 0.9, "ece_10_quantile_bins": 0.0,
                           "recall_at_0.5pct_alerts": 0.5, "precision_at_0.5pct_alerts": 0.2}
    models = lambda ap: {"xgboost": {"raw": {**m(ap, 0.0016), "average_precision_ci95": [ap - 0.05, ap + 0.05]}, "sigmoid": m(ap, 0.00158), "isotonic": m(ap, 0.00161)},
                         "logreg": {"raw": m(0.2, 0.0017), "sigmoid": m(0.2, 0.0017), "isotonic": m(0.2, 0.0017)}}
    doc = {"data": {"rows": 1000, "fraud": 10, "prevalence": 0.01, "time_range_s": [0, 7200], "notes": []}, "seed": 1, "split_fractions": [0.6, 0.2, 0.2],
           "time_split": {"sizes": {"test": 200}, "fraud": {"train": 6, "validation": 2, "test": 2}, "models": models(0.4)},
           "random_split": {"sizes": {"test": 200}, "fraud": {"train": 6, "validation": 2, "test": 2}, "models": models(0.41)}}
    (pub / "phase8_main.json").write_text(json.dumps(doc), encoding="utf-8")
    sec, note = mod.phase8_section(tmp_path)
    assert "method check" in note.lower() and "not the layered system" in sec
    assert "did not improve the Brier score" in sec and "XGBoost had the higher" in sec and "suggestive of an ordering effect at most" in sec
