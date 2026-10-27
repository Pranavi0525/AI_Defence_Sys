"""The credit-card module and the digital-payment module must stay independent.

Static checks (imports), runtime checks (fresh interpreters), artifact checks and a guard that pins the 27 card features.
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from v2.card.features import FEATURES

REPO = Path(__file__).resolve().parents[1]
CARD_DIR = REPO / "v2" / "card"
CARD_FILES = sorted(CARD_DIR.glob("*.py"))
# modules the card package may import from this repo
ALLOWED_INTERNAL = ("v2.card",)
# payment-side modules the card code must never touch (top-level package or module names)
PAYMENT_ROOTS = ("red_team", "v2.serving", "v2.fusion", "v2.evalkit", "v2.features", "v2.gen", "v2.dossier", "v2.feedback", "v2.phases",
                 "v2.audit", "v2.public_check", "inference", "v2_routes", "decision_policy", "risk_fusion", "blue_team_pipeline", "evaluation_harness",
                 "autoencoder", "gcn")

PINNED_FEATURES = ["amount", "log_amount", "is_pos", "is_ecom", "is_atm", "hour_sin", "hour_cos", "hist_len", "gap_s", "dist_prev_km", "speed_prev_kmh",
                   "dist_cp_km", "gap_cp_s", "speed_cp_kmh", "cnt_10m", "cnt_1h", "cnt_24h", "sum_24h", "dec_10m", "dec_1h", "atm_1h", "same_merch_24h",
                   "amt_z", "amt_ratio_max", "new_mcc", "new_merchant", "dist_home_km"]


def _imports(path: Path, package: str) -> set[str]:
    """Absolute dotted names imported by the file (relative imports resolved against `package`)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")
                base = base[: len(base) - (node.level - 1)]
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = node.module or ""
            names.add(mod)
            names |= {f"{mod}.{a.name}" for a in node.names}
    return names


def _under(name: str, roots) -> bool:
    return any(name == r or name.startswith(r + ".") for r in roots)


class TestCardDoesNotDependOnPayment:
    @pytest.mark.parametrize("path", CARD_FILES, ids=lambda p: p.name)
    def test_card_modules_import_no_payment_code(self, path):
        bad = [n for n in _imports(path, "v2.card") if _under(n, PAYMENT_ROOTS)]
        assert not bad, f"{path.name} imports payment-side modules: {bad}"

    @pytest.mark.parametrize("path", CARD_FILES, ids=lambda p: p.name)
    def test_card_modules_only_import_repo_code_from_v2_card(self, path):
        internal = [n for n in _imports(path, "v2.card") if n.split(".")[0] in {"v2", "src", "red_team", "web_prototype", "api", "scripts"}]
        assert all(_under(n, ALLOWED_INTERNAL) or n == "v2" for n in internal), internal

    def test_card_routes_import_only_card_code(self):
        names = _imports(REPO / "web_prototype" / "api" / "card_routes.py", "")
        repo_level = {n for n in names if n.split(".")[0] in {"v2", "red_team", "inference", "v2_routes", "pipeline_runner", "reports", "config", "schemas", "job_store"}}
        assert all(_under(n, ("v2.card",)) or n == "v2" for n in repo_level), repo_level

    def test_card_training_data_is_the_simulator_only(self):
        names = _imports(CARD_DIR / "train.py", "v2.card")
        assert "v2.card.sim" in names or "v2.card.sim.simulate" in names
        assert not [n for n in names if "paysim" in n or "pooled" in n or "sparkov" in n.lower()]


class TestPaymentDoesNotDependOnCard:
    def _payment_files(self):
        files = [p for p in (REPO / "v2").rglob("*.py") if "card" not in p.relative_to(REPO / "v2").parts[:1]]
        files += list((REPO / "src").rglob("*.py"))
        files += [p for p in (REPO / "web_prototype" / "api").glob("*.py") if p.name != "card_routes.py"]
        files += [p for p in REPO.glob("*.py")]
        return files

    def test_no_payment_module_imports_the_card_package(self):
        offenders = {}
        for p in self._payment_files():
            try:
                names = _imports(p, "v2." + p.parent.name if p.parent.name != "v2" else "v2")
            except SyntaxError:
                continue
            hit = [n for n in names if _under(n, ("v2.card", "card_routes"))]
            if hit:
                offenders[p.relative_to(REPO).as_posix()] = hit
        # the ONLY allowed link is the guarded mount in the API entry point
        assert set(offenders) <= {"web_prototype/api/app.py"}, offenders

    def test_app_mounts_card_defensively(self):
        src = (REPO / "web_prototype" / "api" / "app.py").read_text(encoding="utf-8")
        assert "import card_routes" in src and "except Exception" in src.split("import card_routes")[1][:200]
        assert "if card_routes is not None" in src

    def test_payment_dashboard_default_is_payment_and_card_is_a_separate_file(self):
        html = (REPO / "web_prototype" / "dashboard" / "index.html").read_text(encoding="utf-8")
        assert 'id="paymentModule"' in html and 'id="cardModule" hidden' in html
        assert 'aria-selected="true" type="button">Digital Payment Fraud Detection' in html     # payment is the default view
        assert (REPO / "web_prototype" / "dashboard" / "card.js").is_file() and (REPO / "web_prototype" / "dashboard" / "card.css").is_file()
        js = (REPO / "web_prototype" / "dashboard" / "card.js").read_text(encoding="utf-8")
        assert "/api/score" not in js and "/api/v2" not in js and "/api/pipeline" not in js and "/api/reports" not in js


class TestRuntimeIndependence:
    def _run(self, code: str) -> str:
        r = subprocess.run([sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, {str(REPO)!r})\n" + code], capture_output=True, text=True, timeout=180)
        assert r.returncode == 0, r.stderr[-1500:]
        return r.stdout

    def test_importing_card_serving_loads_no_payment_modules(self):
        out = self._run("import v2.card.serving, json\n"
                        "bad=[m for m in sys.modules if m.split('.')[0]=='red_team' or m in ('v2.serving','v2.fusion','v2.evalkit','v2.gen','v2.dossier','v2.features')]\n"
                        "print(json.dumps(bad))")
        assert json.loads(out) == []

    def test_importing_payment_scorer_loads_no_card_modules(self):
        out = self._run("import v2.serving.scorer, json\nprint(json.dumps([m for m in sys.modules if m.startswith('v2.card')]))")
        assert json.loads(out) == []

    def test_card_serving_does_not_need_scikit_learn(self):
        out = self._run("import v2.card.serving, json\nprint(json.dumps('sklearn' in sys.modules))")
        assert json.loads(out) is False


class TestArtifactsAndFeatures:
    def test_card_bundle_lives_in_its_own_directory_with_only_card_files(self):
        card = REPO / "models" / "card"
        assert sorted(p.name for p in card.iterdir()) == ["policy.json", "xgb.json"]
        policy = json.loads((card / "policy.json").read_text())
        assert policy["format"] == "card-bundle-1" and policy["synthetic"] is True

    def test_payment_model_directories_hold_no_card_artifacts(self):
        for d in ("models", "models/v2", "blue_team_output"):
            for p in (REPO / d).iterdir():
                if (d, p.name) != ("models", "card"):
                    assert "card" not in p.name.lower(), (d, p.name)

    def test_card_policy_has_its_own_threshold_and_feature_list(self):
        card = json.loads((REPO / "models" / "card" / "policy.json").read_text())
        pay = json.loads((REPO / "models" / "v2" / "policy.json").read_text())
        assert card["features"] == PINNED_FEATURES
        assert card["features"] != pay["features"]
        assert card["model_version"] != pay["model_version"]

    def test_the_27_card_features_are_unchanged(self):
        assert FEATURES == PINNED_FEATURES and len(FEATURES) == 27

    def test_training_script_refuses_to_overwrite_an_existing_bundle(self, tmp_path):
        (tmp_path / "xgb.json").write_text("existing")
        r = subprocess.run([sys.executable, str(REPO / "scripts" / "train_card_model.py"), "--out", str(tmp_path)], capture_output=True, text=True, timeout=120)
        assert r.returncode != 0 and "--force" in (r.stdout + r.stderr)
        assert (tmp_path / "xgb.json").read_text() == "existing" and not (tmp_path / "policy.json").exists()
