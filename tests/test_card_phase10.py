import numpy as np
from v2.card.sim import simulate
from v2.card.features import build_features, FEATURES

def test_features_are_causal_and_label_free():
    df = simulate(n_cards=60, days=30, seed=1, events_per_pattern=6)
    full = build_features(df)
    cut = df[df.ts < 15 * 86400]
    part = build_features(cut)
    a = full[full.ts < 15 * 86400].sort_values(["card", "ts"]).reset_index(drop=True)
    b = part.sort_values(["card", "ts"]).reset_index(drop=True)
    assert len(a) == len(b)
    np.testing.assert_allclose(a[FEATURES].to_numpy(float), b[FEATURES].to_numpy(float), equal_nan=True)   # future rows change nothing
    for bad in ("pattern", "event", "is_fraud", "declined"):
        assert bad not in FEATURES

def test_simulator_has_all_patterns():
    df = simulate(n_cards=200, days=30, seed=2, events_per_pattern=5)
    assert set(df.pattern.unique()) >= {"LEGIT", "CLONED_CARD_POS", "CARD_TESTING", "CNP_SPREE", "ATM_CASHOUT", "SLOW_DRIP", "LIMIT_SPLIT"}
