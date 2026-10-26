"""
decision_policy_sensitivity.py
================================
Stage 4, Phase 2 -- Sensitivity analysis on the cost-optimal decision policy.

decision_policy.py picks ONE (t_review, t_block) pair by minimizing expected
dollar cost under a single set of placeholder business assumptions:

    assumed_production_fraud_rate = 0.6%
    review_ops_cost                = $12
    review_catch_rate              = 85%
    legit_block_friction_cost      = $150

Those are explicitly flagged in decision_policy.py as business assumptions,
not measured constants. This script does not re-run the cascade -- it reuses
the exact cached out-of-fold (y, proba, dollars) from
decision_policy_validation_cache.npz (produced by decision_policy.py's
get_validation_data_fused(), the Stage 5 fused score) and re-optimizes the two thresholds under a grid of
alternative assumptions, so we can show a judge/reviewer HOW MUCH the policy
moves if a given assumption is wrong, instead of asserting it doesn't.

Three sweeps are produced:
  1. Production fraud rate:  0.2% / 0.6% / 1.0% / 2.0%   (review cost fixed at $12)
  2. Review ops cost:        $5 / $12 / $25 / $50         (fraud rate fixed at 0.6%)
  3. Joint grid of both (4x4), to check for interaction effects between them.

Note: the optimize_thresholds/expected_cost/sample_weights logic here is
copied verbatim from decision_policy.py rather than imported from it, so this
script only needs numpy/pandas -- it does NOT import blue_team_pipeline /
cascade_with_graph (which pull in xgboost/torch and would force a full
cascade re-run just to read a cache file). If decision_policy.py's cost
logic changes, mirror the change here.

Run:
    python3 decision_policy_sensitivity.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Cost model + optimizer (mirrors decision_policy.py -- see note above)
# ---------------------------------------------------------------------------
@dataclass
class CostModel:
    review_ops_cost: float = 12.0
    review_catch_rate: float = 0.85
    legit_block_friction_cost: float = 150.0
    assumed_production_fraud_rate: float | None = 0.006


def sample_weights(y: np.ndarray, target_fraud_rate: float | None) -> np.ndarray:
    if target_fraud_rate is None:
        return np.ones(len(y))
    n_legit = int((y == 0).sum())
    n_fraud = int((y == 1).sum())
    if n_fraud == 0:
        return np.ones(len(y))
    w_fraud = (target_fraud_rate / (1 - target_fraud_rate)) * (n_legit / n_fraud)
    return np.where(y == 1, w_fraud, 1.0)


def expected_cost(y, proba, dollars, weights, t_review, t_block, cost: CostModel) -> float:
    assert t_review <= t_block
    is_block = proba >= t_block
    is_review = (proba >= t_review) & ~is_block
    is_allow = ~is_block & ~is_review
    fraud = y == 1

    c = np.zeros(len(y))
    c[is_allow & fraud] = dollars[is_allow & fraud]
    c[is_review] = cost.review_ops_cost
    c[is_review & fraud] += dollars[is_review & fraud] * (1 - cost.review_catch_rate)
    c[is_block & ~fraud] = cost.legit_block_friction_cost
    return float(np.sum(c * weights))


def policy_stats(y, proba, dollars, t_review, t_block, cost: CostModel) -> dict:
    is_block = proba >= t_block
    is_review = (proba >= t_review) & ~is_block
    is_allow = ~is_block & ~is_review
    fraud = y == 1
    legit = ~fraud

    fraud_caught = int((is_block & fraud).sum()) + int((is_review & fraud).sum())
    weights = sample_weights(y, cost.assumed_production_fraud_rate)

    return {
        "t_review": round(float(t_review), 4),
        "t_block": round(float(t_block), 4),
        "allow_rate": round(float(is_allow.mean()), 4),
        "review_rate": round(float(is_review.mean()), 4),
        "block_rate": round(float(is_block.mean()), 4),
        "legit_blocked": int((is_block & legit).sum()),
        "legit_blocked_rate": round(float((is_block & legit).sum() / max(legit.sum(), 1)), 4),
        "fraud_blocked": int((is_block & fraud).sum()),
        "fraud_reviewed": int((is_review & fraud).sum()),
        "fraud_allowed": int((is_allow & fraud).sum()),
        "fraud_recall_blocked_only": round(float((is_block & fraud).sum() / max(fraud.sum(), 1)), 4),
        "fraud_recall_blocked_plus_review": round(float(fraud_caught / max(fraud.sum(), 1)), 4),
        "dollars_fraud_allowed_through": round(float(dollars[is_allow & fraud].sum()), 2),
        "expected_cost_at_assumed_prevalence": round(
            expected_cost(y, proba, dollars, weights, t_review, t_block, cost), 2
        ),
    }


def optimize_thresholds(y, proba, dollars, cost: CostModel, n_candidates: int = 60) -> dict:
    weights = sample_weights(y, cost.assumed_production_fraud_rate)
    quantiles = np.linspace(0, 1, n_candidates)
    candidates = np.unique(np.concatenate([[0.0, 1.0], np.quantile(proba, quantiles)]))

    best = None
    for t_block in candidates:
        for t_review in candidates:
            if t_review > t_block:
                continue
            c = expected_cost(y, proba, dollars, weights, t_review, t_block, cost)
            if best is None or c < best[0]:
                best = (c, t_review, t_block)

    cost_val, t_review, t_block = best
    result = policy_stats(y, proba, dollars, t_review, t_block, cost)
    result["cost_model"] = asdict(cost)
    return result


def load_cached_validation_data():
    """Reads decision_policy_validation_cache.npz, which this script has
    always assumed holds the Stage 5 FUSED score (see module docstring).

    Phase 4C, B-2: the cache is shared with get_validation_data()'s
    "cascade" variant and previously carried no tag saying which one was
    actually in it, so this could silently run its sweeps against the
    wrong score. Now refuses to proceed on anything but a cache
    explicitly tagged "fused" -- this script has no light way to
    regenerate it itself (that requires the heavy blue_team_pipeline /
    cascade_with_graph / risk_fusion stack this file deliberately avoids
    importing), so it fails loudly with instructions instead.
    """
    cache_path = Path(__file__).parent / "decision_policy_validation_cache.npz"
    if not cache_path.exists():
        raise RuntimeError(
            f"{cache_path.name} not found. Run `python3 decision_policy.py` "
            f"first to generate the Stage 5 fused-score validation cache."
        )
    data = np.load(cache_path, allow_pickle=False)
    variant = str(data["validation_variant"]) if "validation_variant" in data.files else None
    if variant != "fused":
        raise RuntimeError(
            f"{cache_path.name} holds validation_variant={variant!r}, not "
            f"'fused'. This script requires the Stage 5 fused-score cache -- "
            f"run `python3 decision_policy.py` (which ends by calling "
            f"get_validation_data_fused()) to regenerate it, then re-run this "
            f"script."
        )
    return data["y"], data["proba"], data["dollars"]


# ---------------------------------------------------------------------------
# Sensitivity sweeps
# ---------------------------------------------------------------------------
FRAUD_RATES = [0.002, 0.006, 0.01, 0.02]
REVIEW_COSTS = [5.0, 12.0, 25.0, 50.0]
BASE_REVIEW_COST = 12.0
BASE_FRAUD_RATE = 0.006


def run_fraud_rate_sweep(y, proba, dollars) -> list[dict]:
    rows = []
    for r in FRAUD_RATES:
        cost = CostModel(review_ops_cost=BASE_REVIEW_COST, assumed_production_fraud_rate=r)
        res = optimize_thresholds(y, proba, dollars, cost)
        res["swept_param"] = "assumed_production_fraud_rate"
        res["swept_value"] = r
        rows.append(res)
    return rows


def run_review_cost_sweep(y, proba, dollars) -> list[dict]:
    rows = []
    for c_val in REVIEW_COSTS:
        cost = CostModel(review_ops_cost=c_val, assumed_production_fraud_rate=BASE_FRAUD_RATE)
        res = optimize_thresholds(y, proba, dollars, cost)
        res["swept_param"] = "review_ops_cost"
        res["swept_value"] = c_val
        rows.append(res)
    return rows


def run_joint_grid(y, proba, dollars) -> list[dict]:
    rows = []
    for r in FRAUD_RATES:
        for c_val in REVIEW_COSTS:
            cost = CostModel(review_ops_cost=c_val, assumed_production_fraud_rate=r)
            res = optimize_thresholds(y, proba, dollars, cost)
            res["fraud_rate"] = r
            res["review_cost"] = c_val
            rows.append(res)
    return rows


def fmt_table(rows: list[dict], label_key: str, label_fmt) -> str:
    header = (f"{'setting':>10} | {'t_review':>8} | {'t_block':>8} | "
              f"{'allow%':>7} | {'review%':>7} | {'block%':>7} | "
              f"{'legit_blk%':>10} | {'recall(B+R)%':>12} | {'exp_cost($)':>11}")
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(
            f"{label_fmt(r[label_key]):>10} | {r['t_review']:>8} | {r['t_block']:>8} | "
            f"{r['allow_rate']*100:>6.1f}% | {r['review_rate']*100:>6.1f}% | "
            f"{r['block_rate']*100:>6.1f}% | {r['legit_blocked_rate']*100:>9.1f}% | "
            f"{r['fraud_recall_blocked_plus_review']*100:>11.1f}% | "
            f"{r['expected_cost_at_assumed_prevalence']:>11,.2f}"
        )
    return "\n".join(lines)


def main():
    y, proba, dollars = load_cached_validation_data()
    n_fraud, n = int((y == 1).sum()), len(y)

    print("=" * 88)
    print("SENSITIVITY ANALYSIS -- Stage 4 cost-optimal decision policy")
    print("=" * 88)
    print(f"Validation set: {n} traces, {n_fraud} fraud ({n_fraud/n:.1%}) -- "
          f"red-team corpus prevalence, reweighted per assumption below.\n")

    fraud_sweep = run_fraud_rate_sweep(y, proba, dollars)
    print(f"-- Sweep 1: production fraud rate (review_ops_cost fixed at "
          f"${BASE_REVIEW_COST:.0f}) --")
    print(fmt_table(fraud_sweep, "swept_value", lambda v: f"{v:.1%}"))
    print()

    review_sweep = run_review_cost_sweep(y, proba, dollars)
    print(f"-- Sweep 2: review ops cost (fraud rate fixed at {BASE_FRAUD_RATE:.1%}) --")
    print(fmt_table(review_sweep, "swept_value", lambda v: f"${v:.0f}"))
    print()

    joint = run_joint_grid(y, proba, dollars)
    print("-- Sweep 3: joint grid (fraud rate x review cost), t_block only --")
    header = f"{'fraud rate':>10} \\ {'review cost':<8}" + "".join(
        f"{'$'+str(int(c)):>10}" for c in REVIEW_COSTS
    )
    print(header)
    for r in FRAUD_RATES:
        row_vals = [x for x in joint if x["fraud_rate"] == r]
        row_vals.sort(key=lambda x: x["review_cost"])
        line = f"{r*100:>9.1f}% " + "".join(f"{x['t_block']:>10.3f}" for x in row_vals)
        print(line)
    print()

    # Range summary -- how much does the threshold actually move?
    t_block_range_fraud = [x["t_block"] for x in fraud_sweep]
    t_block_range_review = [x["t_block"] for x in review_sweep]
    print("-- Range summary --")
    print(f"t_block across fraud-rate sweep (0.2% - 2.0%): "
          f"{min(t_block_range_fraud):.3f} - {max(t_block_range_fraud):.3f} "
          f"(span {max(t_block_range_fraud)-min(t_block_range_fraud):.3f})")
    print(f"t_block across review-cost sweep ($5 - $50):    "
          f"{min(t_block_range_review):.3f} - {max(t_block_range_review):.3f} "
          f"(span {max(t_block_range_review)-min(t_block_range_review):.3f})")

    out = {
        "base_assumptions": {"review_ops_cost": BASE_REVIEW_COST,
                              "assumed_production_fraud_rate": BASE_FRAUD_RATE},
        "fraud_rate_sweep": fraud_sweep,
        "review_cost_sweep": review_sweep,
        "joint_grid": joint,
    }
    out_path = Path(__file__).parent / "decision_policy_sensitivity_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
