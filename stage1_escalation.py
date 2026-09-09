"""
stage1_escalation.py
=====================
Stage-1 addendum: two new cheap, deterministic escalation signals, added
ON TOP OF blue_team_pipeline.stage1_rule_filter (v1) -- never replacing or
weakening it.

WHY THESE TWO SIGNALS
----------------------
miss_collector.py's real, full-cascade misses.jsonl (7 real misses, not the
narrower 2-miss blue_team_output/misses.jsonl) shows 5 stage1_miss cases --
all AUTHORIZED_PUSH_PAYMENT -- that none of the four existing Stage-1 rules
catch. Inspecting their behavioral_features shows they split cleanly into
exactly two shapes neither existing rule looks at:

  1. One unusually large transaction, otherwise unremarkable behavior
     (atk-1ebff3d8: $15,862.36; atk-3f48c9b5: $35,302.40;
      atk-7cad7aab: $10,031.44) -- "victim authorizes one big push payment."
  2. A BENEFICIARY_ADDITION event with ZERO transactions recorded in the
     observation window (atk-ee11d67a, atk-a127c99a) -- the existing
     beneficiary_added_before_transaction feature requires BOTH events in
     the SAME window, so it structurally cannot fire when the payment
     lands just outside the window boundary.

CALIBRATION -- AGAINST THE REAL LEGIT POPULATION, NOT THE MISSES
------------------------------------------------------------------
Thresholds below are set from feature_table.csv's 1,205 real legitimate
traces (see calibrate_against_legit_population()), NOT fitted to the 5
miss values themselves -- fitting a threshold to the exact cases you're
trying to catch is the same "solve for the answer" error as retraining on
data Stage 1 will never show the model. `stage1_high_value_signal` uses
$8,000, comfortably above the legit population's observed p99.9 (~$6,972)
and even its outright max ($7,294.71).

Run this file directly to print the calibration cost measured against the
real feature_table.csv legit rows.
"""
from __future__ import annotations

import pandas as pd

HIGH_VALUE_THRESHOLD = 8_000.0


def stage1_high_value_signal(row) -> bool:
    """New signal #1: one transaction well above anything seen in the
    real legitimate population (see calibration below)."""
    return bool(row["amount_max"] > HIGH_VALUE_THRESHOLD)


def stage1_beneficiary_no_transaction_signal(row) -> bool:
    """New signal #2: a beneficiary was added but zero transactions have
    happened yet in this observation window. Complements (does not
    replace) beneficiary_added_before_transaction, which requires both
    events to land inside the same window to fire."""
    return bool(row["count_beneficiary_addition"] >= 1 and row["count_transaction"] == 0)


def stage1_rule_filter_v2(row, base_filter) -> bool:
    """Pure OR-addendum: v2 = v1 OR high_value OR beneficiary_no_txn.
    Guaranteed to escalate a superset of whatever v1 already escalates --
    it can only ever widen the funnel, never narrow it. `base_filter` is
    passed in explicitly (rather than imported and hardcoded) so this
    stays a wrapper around whatever the live blue_team_pipeline.stage1_rule_filter
    is, never a fork of it.
    """
    return bool(
        base_filter(row)
        or stage1_high_value_signal(row)
        or stage1_beneficiary_no_transaction_signal(row)
    )


def calibrate_against_legit_population(feature_table_path: str, base_filter) -> dict:
    """Measures the COST of the addendum: how many additional legitimate
    traces get escalated to Stage 2 that v1 didn't already escalate.
    This is the number that matters for "did we just reintroduce a
    review/cost-savings problem" -- not accuracy on the misses, which by
    construction will always look good since the signals were designed
    around them.
    """
    df = pd.read_csv(feature_table_path)
    legit = df[df["fraud"] == 0]

    v1_escalated = legit.apply(base_filter, axis=1)
    v2_escalated = legit.apply(lambda r: stage1_rule_filter_v2(r, base_filter), axis=1)

    newly_escalated = v2_escalated & (~v1_escalated)
    high_value_escalated = legit.apply(stage1_high_value_signal, axis=1)
    beneficiary_escalated = legit.apply(stage1_beneficiary_no_transaction_signal, axis=1)

    return {
        "n_legit_traces": int(len(legit)),
        "v1_legit_escalation_rate": float(v1_escalated.mean()),
        "v2_legit_escalation_rate": float(v2_escalated.mean()),
        "additional_legit_traces_escalated_by_v2": int(newly_escalated.sum()),
        "additional_legit_escalation_rate": float(newly_escalated.mean()),
        "high_value_signal_alone_escalates_n_legit": int(high_value_escalated.sum()),
        "beneficiary_no_txn_signal_alone_escalates_n_legit": int(beneficiary_escalated.sum()),
        "legit_amount_max_p99": float(legit["amount_max"].quantile(0.99)),
        "legit_amount_max_p999": float(legit["amount_max"].quantile(0.999)),
        "legit_amount_max_max": float(legit["amount_max"].max()),
        "high_value_threshold_used": HIGH_VALUE_THRESHOLD,
    }


if __name__ == "__main__":
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from blue_team_pipeline import stage1_rule_filter as v1
    except ModuleNotFoundError:
        sys.path.insert(0, str(Path(__file__).parent / "xgb_stub"))
        from blue_team_pipeline import stage1_rule_filter as v1

    cost = calibrate_against_legit_population(
        str(Path(__file__).parent / "blue_team_output" / "feature_table.csv"), v1
    )
    import json
    print(json.dumps(cost, indent=2))
