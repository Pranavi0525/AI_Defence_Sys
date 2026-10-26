"""Phase 7a tests: case dossiers (content, read-only behaviour, liability mapping)."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def scorer():
    from v2.serving.scorer import V2Scorer
    return V2Scorer(ROOT / "models" / "v2")


def _example(name):
    j = json.loads((ROOT / "web_prototype" / "examples" / name).read_text(encoding="utf-8"))
    return j.get("trace", j)


def test_v1_liability_mapping_matches_decision_policy():
    """The three v1 families must keep the exact mapping used by the v1 decision policy (no silent drift)."""
    import decision_policy as dp
    from v2.dossier import LIABILITY_V2
    for fam in ("ACCOUNT_TAKEOVER", "AUTHORIZED_PUSH_PAYMENT", "MULE_NETWORK"):
        liable, acting, basis, _ = LIABILITY_V2[fam]
        assert (liable, acting) == (dp.liable_side(fam), dp.acting_side(fam)) and basis == "v1_policy"


def test_new_family_liability_is_labelled_as_assumption():
    from v2.dossier import LIABILITY_V2
    for fam in ("SYNTHETIC_IDENTITY", "FAKE_MERCHANT", "DEEPFAKE_KYC", "AI_SCRIPTED_APP"):
        assert LIABILITY_V2[fam][2] == "ASSUMED"


def test_block_case_has_all_sections_and_caveats(scorer):
    from v2.dossier import build_dossier, render_markdown
    d = build_dossier(scorer, _example("v2_sample_deepfake_kyc.json"))
    assert d["decision"] in ("REVIEW", "BLOCK")
    assert d["model_drivers"] and d["graph_evidence"] and d["caveats"]
    assert d["acting_party"]["family"] is None and "unknown" in d["acting_party"]["note"].lower()
    md = render_markdown(d)
    for part in ("### 1. Decision", "### 3. What pushed the model", "### 4. Evidence", "### 6. Acting party", "Synthetic-data prototype"):
        assert part in md


def test_family_fills_acting_party(scorer):
    from v2.dossier import build_dossier
    d = build_dossier(scorer, _example("v2_sample_deepfake_kyc.json"), family="DEEPFAKE_KYC")
    assert d["acting_party"]["liable_side"] == "ONBOARDING_INSTITUTION" and d["acting_party"]["basis"] == "ASSUMED"


def test_allow_returns_none_unless_forced(scorer):
    from v2.dossier import build_dossier
    rec = _example("v2_sample_routine_legit.json")
    assert build_dossier(scorer, rec) is None
    d = build_dossier(scorer, rec, include_allow=True)
    assert d["decision"] == "ALLOW"


def test_dossier_is_read_only(scorer):
    from v2.dossier import build_dossier
    before = json.dumps(scorer.store.stats(), sort_keys=True)
    build_dossier(scorer, _example("v2_sample_deepfake_kyc.json"))
    assert json.dumps(scorer.store.stats(), sort_keys=True) == before


def test_dossier_decision_matches_scorer(scorer):
    from v2.dossier import build_dossier
    rec = _example("v2_sample_deepfake_kyc.json")
    assert build_dossier(scorer, rec)["decision"] == scorer.score(rec)["decision"]


def test_evidence_flags_fire_on_known_values():
    from v2.dossier import evidence_flags
    nan = float("nan")
    f = {"beneficiary_added_before_transaction": 1, "time_from_beneficiary_add_to_transaction": 600.0, "new_device_present": 1,
         "kyc_virtual_camera": 1.0, "kyc_liveness": 0.4, "kyc_retries": 4.0, "account_age_days": 3.0, "merchant_age_min": nan,
         "failed_then_completed": 0, "count_transaction": 2, "min_time_between_transactions": 500.0}
    flags = {e["flag"] for e in evidence_flags(f)}
    assert flags == {"new_payee_paid_quickly", "new_device", "kyc_virtual_camera", "kyc_low_liveness", "kyc_many_retries", "young_account"}
    assert evidence_flags({"beneficiary_added_before_transaction": 0, "time_from_beneficiary_add_to_transaction": -1.0, "new_device_present": 0,
                           "kyc_virtual_camera": nan, "kyc_liveness": nan, "kyc_retries": nan, "account_age_days": nan, "merchant_age_min": nan,
                           "failed_then_completed": 0, "count_transaction": 1, "min_time_between_transactions": 10.0}) == []


def test_graph_evidence_sentences():
    from v2.dossier import graph_evidence
    assert "adds nothing" in graph_evidence({})[0]
    s = graph_evidence({"g_payee_shared_customers": 3, "g_component_size": 5})
    assert any("payee" in x and "3" in x for x in s) and any("group of 5" in x for x in s)
