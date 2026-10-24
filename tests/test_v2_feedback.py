"""Phase 7b tests: the analyst feedback loop is leak-free, consistent with the served bundle, and its statistics are right."""
import numpy as np
import pytest

from v2.features import WORLD_B_HARD, build_world_b


@pytest.fixture(scope="module")
def world():
    df, recs = build_world_b(seed=5, n_legit=300, n_per_family=30, return_records=True, **WORLD_B_HARD)
    return df, recs


def test_pipeline_thresholds_equal_served_bundle(world, tmp_path):
    """The loop retrains with the SAME recipe as scripts/build_v2_bundle.py (otherwise before/after would not describe the served model)."""
    from v2.feedback import Pipeline
    from v2.phases import FEATS_B, NOV_B
    from v2.serving.bundle import fit_bundle
    df, recs = world
    b = fit_bundle(df, recs, FEATS_B, NOV_B, tmp_path)
    p = Pipeline(seed=11).fit(df)
    assert p.t_review == pytest.approx(b["t_review"]) and p.t_block == pytest.approx(b["t_block"])


def test_queue_contains_only_real_mistakes(world):
    from v2.feedback import Pipeline, analyst_queue
    df, _ = world
    p = Pipeline(seed=11).fit(df)
    q = analyst_queue(df, p.oof_decision, found_share=1.0)
    assert (q["misses"].fraud == 1).all() and (q["false_alarms"].fraud == 0).all()
    assert len(q["misses"]) == q["n_missed_total"] == int(((df.fraud == 1) & (p.oof_decision == 0)).sum())
    assert analyst_queue(df, p.oof_decision, found_share=0.0)["n_misses_found"] == 0


def test_hard_examples_match_mistakes_and_are_new_customers(world):
    from v2.feedback import Pipeline, analyst_queue, frame_from_records, generate_hard_examples, generate_control
    df, _ = world
    p = Pipeline(seed=11).fit(df)
    q = analyst_queue(df, p.oof_decision)
    recs = generate_hard_examples(q, seed=123, per_case=2)
    fresh = frame_from_records(recs)
    for fam, n in q["misses"].family.value_counts().items():
        assert (fresh.family == fam).sum() == 2 * n
    assert fresh.fraud.sum() == 2 * len(q["misses"]) and (1 - fresh.fraud).sum() == 2 * len(q["false_alarms"])
    assert not set(fresh.customer_id) & set(df.customer_id)
    ctrl = frame_from_records(generate_control(int(fresh.fraud.sum()), int((1 - fresh.fraud).sum()), seed=7))
    assert ctrl.fraud.sum() == fresh.fraud.sum() and (1 - ctrl.fraud).sum() == (1 - fresh.fraud).sum()


def test_summarise_paired_differences():
    from v2.feedback import summarise
    def rep(b, t, c):
        return {"arms": {"baseline": {"m": b}, "targeted": {"m": t}, "control_same_size": {"m": c}}}
    S = summarise([rep(1.0, 2.0, 1.5), rep(1.0, 3.0, 4.0), rep(2.0, 2.0, 2.0)])
    a = S["paired"]["targeted_minus_baseline"]["m"]
    assert a["mean"] == pytest.approx(np.mean([1.0, 2.0, 0.0])) and (a["n_up"], a["n_down"]) == (2, 0)
    c = S["paired"]["targeted_minus_control_same_size"]["m"]
    assert (c["n_up"], c["n_down"]) == (1, 1)
