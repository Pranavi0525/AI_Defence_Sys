import numpy as np
from v2.gen import build_v2_records
from v2.history import add_history
from v2.hist_features import hist_features, HIST, _parse

KW = dict(seed=5, n_legit=120, n_per_family=15, stealth=0.5, onboarding_share=0.2)

def test_history_is_strictly_before_trace_and_originals_untouched():
    base = build_v2_records(**KW); snap = [r["events"][0]["timestamp"] for r in base]
    recs = add_history(base, seed=1, strength=1.0)
    assert [r["events"][0]["timestamp"] for r in base] == snap            # input not mutated
    n_with = 0
    for r in recs:
        t0 = min(_parse(e["timestamp"]) for e in r["events"])
        assert all(_parse(e["timestamp"]) < t0 for e in r.get("history", []))
        n_with += bool(r.get("history"))
    assert n_with > 0

def test_cold_start_families_have_no_or_thin_history():
    recs = add_history(build_v2_records(**KW), seed=1)
    dk = [r for r in recs if r["attack_family"] == "DEEPFAKE_KYC"]
    assert all(len(r["history"]) == 0 for r in dk)

def test_strength_zero_makes_history_uninformative():
    recs = add_history(build_v2_records(**KW), seed=1, strength=0.0)
    f = [hist_features(r) for r in recs]
    leg = [x["hist_merchant_known_share"] for x, r in zip(f, recs) if r["fraud"] == 0 and not np.isnan(x["hist_merchant_known_share"])]
    fr = [x["hist_merchant_known_share"] for x, r in zip(f, recs) if r["fraud"] == 1 and not np.isnan(x["hist_merchant_known_share"])]
    assert np.mean(leg) < 0.2 and (not fr or np.mean(fr) < 0.2)           # chance-level repeat only (merchant pool overlap)

def test_features_ignore_future_events():
    recs = add_history(build_v2_records(**KW), seed=1)
    r = next(x for x in recs if x["history"])
    base = hist_features(r)
    r2 = dict(r); r2["history"] = r["history"] + [dict(r["history"][0], timestamp="2099-01-01T00:00:00")]
    a, b = base, hist_features(r2)
    assert all((np.isnan(a[k]) and np.isnan(b[k])) or a[k] == b[k] for k in HIST)
