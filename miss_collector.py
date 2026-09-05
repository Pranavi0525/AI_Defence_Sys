"""
miss_collector.py
===================
Stage 6 -- Miss Collector: closed-loop handoff back to Red Team

WHAT THIS FILE DOES
--------------------
Consumes the verified, UNMODIFIED pipeline:

    Stage 1 (deterministic rules)   -- blue_team_pipeline.stage1_rule_filter
    Stage 2 (XGBoost)               -- blue_team_pipeline FEATURE_COLS + model
    Stage 3 (graph / GCN)           -- cascade_with_graph.run_three_stage_cascade (base score)
    Stage 4 (autoencoder)           -- cascade_with_autoencoder (base score)
    Stage 5 (Risk Fusion)           -- decision_policy.get_validation_data_fused
                                        (stacked LR over Stages 1+2/3/4)
    Stage 6 (cost-optimal policy)   -- decision_policy.optimize_thresholds,
                                        now applied to the Stage 5 fused score

and asks one specific question:

    Which fraud traces make it all the way through the FULL cascade and
    STILL come out the other end with a final decision of ALLOW?

"ALLOW" means the genuine, decision-grade, out-of-fold Stage 1+2+3 score
falls BELOW the prevalence-corrected t_review threshold from
decision_policy.py -- i.e. the trace was never even routed to a human
reviewer, let alone blocked. These are exactly the cases the whole
4-stage system is designed to route as "let the money move," and every
one of them is worth handing back to the Red Team with enough context
(attack family, difficulty, the actual behavioral feature values, and
why each stage failed to catch it) to reason about.

RELATIONSHIP TO THE EXISTING write_misses() (blue_team_pipeline.py)
----------------------------------------------------------------------
blue_team_pipeline.write_misses() already exists and is untouched by
this file. It answers a narrower question -- fraud rows a single 80/20
holdout split's calibrated Stage-2 model, alone, scores below a flat
0.5 -- and keeps writing to blue_team_output/misses.jsonl exactly as
before. This file answers the broader "ultimately gets ALLOW" question
using the artifact everything downstream (decision_policy.py,
explainability.py) already trusts: the real 5-fold out-of-fold
Stage 1+2+3 cascade score, evaluated against the actual operational
ALLOW/REVIEW/BLOCK thresholds, not a single train/test split or a bare
Stage-2-only cutoff. The two files are complementary, not competing --
this one is written to its own top-level misses.jsonl so neither
overwrites the other.

NOTHING in blue_team_pipeline.py, cascade_with_graph.py, gcn.py,
quiet_ring_overlay.py, or decision_policy.py is modified or
reimplemented differently here. This file only calls their existing,
unmodified functions and reads their outputs.

Run from the repo root, with src/ on PYTHONPATH:

    PYTHONPATH=src python3 miss_collector.py

Output: ./misses.jsonl -- one JSON object per fraud trace that
resolves to ALLOW under the full cascade + decision policy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

import blue_team_pipeline as btp        # noqa: E402  -- unmodified
import cascade_with_graph as cwg        # noqa: E402  -- unmodified
import decision_policy as dp            # noqa: E402  -- unmodified

try:
    # Reuse the same human-readable Stage-1 rule descriptions and
    # feature labels explainability.py already defines, rather than
    # re-authoring a second copy that could drift out of sync.
    from explainability import STAGE1_RULES, FEATURE_LABELS
except Exception:  # pragma: no cover - explainability has heavier deps
    # (shap/matplotlib) that aren't required just to collect misses.
    STAGE1_RULES = [
        ("beneficiary_added_before_transaction", "==", 1,
         "a payee was added and then paid within the same session"),
        ("new_device_present", "==", 1,
         "a new device was registered on this trace"),
        ("min_time_between_transactions", "<", 3600,
         "two transactions happened under an hour apart"),
        ("transactions_per_hour", ">", 2.5,
         "transaction velocity exceeded 2.5/hour"),
    ]
    FEATURE_LABELS = {}

CACHE_PATH = Path(__file__).parent / "decision_policy_validation_cache.npz"
OUT_PATH = Path(__file__).parent / "misses.jsonl"


# ---------------------------------------------------------------------------
# Step 1 -- feature table + the genuine, decision-grade OOF cascade score
# ---------------------------------------------------------------------------
def get_df_and_scores(cfg: dict):
    """Builds the (unmodified) feature table via cascade_with_graph, then
    gets the genuine out-of-fold Stage 1+2+3 score + per-trace dollar
    exposure for every row -- from the cached validation run if it's
    still aligned to this data, otherwise by running the real thing
    (decision_policy.get_validation_data, same as decision_policy.py's
    own main() does).
    """
    print("Loading Red Team corpora + building the (unmodified) feature table...")
    all_records = cwg.load_all_records(cfg)
    ring_membership = cwg.get_graph_connected_trace_ids(all_records)
    df, A, connected_mask = cwg.build_feature_table_and_graph(all_records, ring_membership)
    y = df["fraud"].values.astype(int)

    # Phase 4C, B-2: this file needs the Stage 5 FUSED score (per the
    # module docstring above), so the cache is only usable when it is
    # explicitly tagged validation_variant="fused" AND its labels match
    # this df -- checking y-alignment alone (the old behavior) could not
    # tell a "cascade"-variant cache apart from a "fused" one, since both
    # carry the identical y array. That was the confirmed root cause of
    # misses.jsonl silently disagreeing with decision_policy_results.json.
    if CACHE_PATH.exists():
        try:
            y_cached, proba, dollars = dp.load_cached_validation_data("fused", y_check=y)
            print(f"Using cached out-of-fold Stage 5 fused scores ({CACHE_PATH.name}).")
            return df, all_records, A, connected_mask, y, proba, dollars
        except dp.ValidationCacheMismatch as e:
            print(f"Cache unusable ({e}) -- regenerating.")

    print("No usable cache -- running the real Risk Fusion pipeline via "
          "decision_policy.get_validation_data_fused() "
          "(a couple of minutes, fresh GCN + autoencoder + fusion LR per fold)...")
    df2, y2, proba, dollars, _fusion_result = dp.get_validation_data_fused()
    # dp.get_validation_data_fused() rebuilds its own df internally via the
    # same unmodified cwg functions; use ITS df (and matching A/connected_mask)
    # so everything stays positionally aligned to proba/dollars. Using the
    # fused score here (not the raw Stage 1+2+3 score) keeps this file
    # consistent with decision_policy.py's own frozen thresholds, which are
    # now tuned against the fused score too.
    _, A2, connected_mask2 = cwg.build_feature_table_and_graph(all_records, ring_membership)
    return df2, all_records, A2, connected_mask2, y2, proba, dollars


# ---------------------------------------------------------------------------
# Step 2 -- Stage 1 rule breakdown for one row (reused, not reimplemented)
# ---------------------------------------------------------------------------
def _stage1_rules_checked(row) -> list[dict]:
    checked = []
    for feat, op, thresh, desc in STAGE1_RULES:
        val = row[feat]
        fired = bool((val == thresh) if op == "==" else (val < thresh) if op == "<" else (val > thresh))
        checked.append({
            "rule": desc,
            "feature": feat,
            "value": round(float(val), 3) if isinstance(val, (int, float, np.floating, np.integer)) else val,
            "threshold": thresh,
            "fired": fired,
        })
    return checked


def _reason_for_miss(escalated: bool, connected: bool, is_ring: bool,
                      score: float, t_review: float) -> str:
    if not escalated:
        return (
            "Stage 1 (rule filter) never escalated this trace to the ML "
            "model -- none of the four trigger rules fired, so the cascade "
            "score stayed at 0.0 and the trace defaulted to ALLOW without "
            "ever being scored."
        )
    if connected:
        ring_note = (
            " (it IS a flagged mule-ring member, but the ring signal still "
            "wasn't enough to clear REVIEW)" if is_ring else
            " (it is graph-connected to other customers' traces, but not "
            "part of the flagged ring)"
        )
        return (
            f"Stage 1 escalated this trace and Stage 3's graph step ran on "
            f"it{ring_note}, but the final score ({score:.3f}) still fell "
            f"under the REVIEW threshold ({t_review:.3f})."
        )
    return (
        f"Stage 1 escalated this trace to Stage 2 (XGBoost), and it has no "
        f"cross-customer graph connections for Stage 3 to escalate via, "
        f"but Stage 2's score ({score:.3f}) still fell under the REVIEW "
        f"threshold ({t_review:.3f})."
    )


# ---------------------------------------------------------------------------
# Step 3 -- collect every fraud trace whose FINAL decision is ALLOW
# ---------------------------------------------------------------------------
def collect_misses(
    df: pd.DataFrame, all_records: list[dict], connected_mask: np.ndarray,
    y: np.ndarray, proba: np.ndarray, dollars: np.ndarray,
    t_review: float, t_block: float,
) -> list[dict]:
    escalate = df.apply(btp.stage1_rule_filter, axis=1).values  # unmodified Stage 1
    is_fraud = y == 1
    is_allow = proba < t_review  # Stage 4's own ALLOW definition, unmodified

    miss_idx = np.where(is_fraud & is_allow)[0]
    print(f"\n{len(miss_idx)} fraud trace(s) out of {int(is_fraud.sum())} total "
          f"fraud traces resolve to a final decision of ALLOW "
          f"(score < t_review={t_review:.4f}).")

    misses = []
    for i in miss_idx:
        row = df.iloc[i]
        behavioral_features = {
            col: (round(float(row[col]), 4) if isinstance(row[col], (int, float, np.floating, np.integer)) else row[col])
            for col in btp.FEATURE_COLS
        }
        record = {
            "trace_id": row["trace_id"],
            "customer_id": row["customer_id"],
            "attack_family": row["attack_family"],
            "attack_difficulty": row["attack_difficulty"],
            "final_decision": "ALLOW",
            "final_score": round(float(proba[i]), 4),
            "t_review": round(float(t_review), 4),
            "t_block": round(float(t_block), 4),
            "dollars_in_trace": round(float(dollars[i]), 2),
            "stage1_escalated_to_ml": bool(escalate[i]),
            "graph_connected": bool(connected_mask[i]),
            "is_flagged_mule_ring_member": bool(row["is_ring"]) if "is_ring" in row else False,
            "stage1_rules_checked": _stage1_rules_checked(row),
            "behavioral_features": behavioral_features,
            "reason_for_miss": _reason_for_miss(
                bool(escalate[i]), bool(connected_mask[i]),
                bool(row["is_ring"]) if "is_ring" in row else False,
                float(proba[i]), float(t_review),
            ),
        }
        misses.append(record)

    # Most concerning (closest to slipping into REVIEW, or highest $
    # exposure) first, then group by attack family for readability.
    misses.sort(key=lambda m: (m["attack_family"], -m["final_score"]))
    return misses


def _stamp_misses_metadata() -> dict:
    """Builds the provenance header line for misses.jsonl (Phase 4C, B-3).

    misses.jsonl is a JSONL file of per-trace records, not a single dict,
    so artifact_metadata.stamp_artifact() (which adds a key to a dict)
    can't be applied to the file as a whole the way decision_policy.py /
    risk_fusion.py / blue_team_pipeline.py do. Instead this reuses the
    exact same stamp_artifact() mechanism on a small standalone dict and
    writes it as the FIRST line of the file -- a header record, not a
    miss. Every reader of misses.jsonl (consistency_check.py's
    _load_jsonl) explicitly recognizes and skips this line by its
    "_artifact_metadata" key so it's never mistaken for a trace record.
    """
    from artifact_metadata import stamp_artifact
    return stamp_artifact({}, Path(__file__).parent)


def write_misses_jsonl(misses: list[dict], out_path: Path) -> None:
    with open(out_path, "w") as f:
        f.write(json.dumps(_stamp_misses_metadata()) + "\n")
        for m in misses:
            f.write(json.dumps(m) + "\n")
    print(f"Wrote {len(misses)} missed fraud trace(s) to {out_path} "
          f"(plus a provenance header line).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    cfg = btp.CONFIG
    df, all_records, A, connected_mask, y, proba, dollars = get_df_and_scores(cfg)

    print("\nGetting Stage 4's cost-optimal (prevalence-corrected) thresholds "
          "via decision_policy.optimize_thresholds (unmodified)...")
    cost = dp.CostModel()
    policy = dp.optimize_thresholds(y, proba, dollars, cost)
    t_review, t_block = float(policy["t_review"]), float(policy["t_block"])
    print(f"  REVIEW >= {t_review:.4f}, BLOCK >= {t_block:.4f}")

    misses = collect_misses(df, all_records, connected_mask, y, proba, dollars, t_review, t_block)
    write_misses_jsonl(misses, OUT_PATH)

    if misses:
        by_family = {}
        for m in misses:
            by_family.setdefault(m["attack_family"], []).append(m)
        total_dollars = sum(m["dollars_in_trace"] for m in misses)
        print("\nBreakdown by attack family:")
        for fam, ms in by_family.items():
            fam_dollars = sum(m["dollars_in_trace"] for m in ms)
            by_diff = {}
            for m in ms:
                by_diff[m["attack_difficulty"]] = by_diff.get(m["attack_difficulty"], 0) + 1
            print(f"  {fam}: {len(ms)} missed, ${fam_dollars:,.2f} allowed through, "
                  f"by difficulty: {by_diff}")
        print(f"\nTotal dollars allowed through across all misses: ${total_dollars:,.2f}")
        never_escalated = sum(1 for m in misses if not m["stage1_escalated_to_ml"])
        print(f"Of those, {never_escalated} were never even escalated past Stage 1 "
              f"(the ML model never scored them).")
    else:
        print("\nNo fraud traces resolved to ALLOW under the current cascade + "
              "decision policy.")


if __name__ == "__main__":
    main()