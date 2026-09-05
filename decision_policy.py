"""
decision_policy.py
===================
Stage 4 -- Cost-Optimal Decision Policy (ALLOW / REVIEW / BLOCK)

Turns the verified Stage 1+2+3 cascade's fraud probability into an actual
operational decision, by choosing two thresholds (t_review <= t_block)
that MINIMIZE AN EXPECTED DOLLAR COST over the validation population,
rather than being eyeballed off a probability number.

    score <  t_review                -> ALLOW
    t_review <= score < t_block      -> REVIEW  (human/step-up)
    score >= t_block                 -> BLOCK

WHY A NAIVE VERSION OF THIS BREAKS (block-everyone bug)
--------------------------------------------------------
The validation population built by cascade_with_graph.load_all_records()
is a RED TEAM CORPUS: ~19% of traces are fraud (277/1458). That is not
even close to a real deposit account's fraud prevalence (typically well
under 1%). If you compute expected cost directly on this sample -- i.e.
you let the sample's own class balance stand in for the production
prior -- the optimizer is implicitly being told "1 in 5 customers walking
in the door is a fraudster," and at that prevalence, blocking almost
everyone genuinely does minimize dollar cost: the model is cheap
(false-positive friction) relative to how often it's right. That is a
real, correct optimum for a 19%-fraud population; it is just not *our*
population.

The fix is IMPORTANCE-REWEIGHTING the fraud class down to match an
assumed production prevalence before computing costs (see
`assumed_production_fraud_rate` on CostModel). This file computes and
reports BOTH the naive (unweighted) and reweighted optimum so the
difference -- and the reason for it -- is visible rather than silently
"fixed".

Run:
    PYTHONPATH=src python3 decision_policy.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

import blue_team_pipeline as btp          # noqa: E402
import cascade_with_graph as cwg          # noqa: E402


# ---------------------------------------------------------------------------
# Step 1 -- validation data: out-of-fold cascade scores + true $ exposure
# ---------------------------------------------------------------------------
def get_validation_data(random_state: int = cwg.RANDOM_STATE):
    """
    Re-runs the verified Stage 1+2+3 cascade (5-fold CV -- every trace's
    score comes from a fold that never trained on it) and pairs it with
    the ACTUAL dollars transacted in that trace: the sum of every
    TRANSACTION event's amount, not amount_mean/amount_max (those are
    per-transaction summary features of the trace, not its total $
    exposure -- using them would understate loss for multi-transaction
    traces).

    Returns
    -------
    df      : feature table (also carries fraud / attack_family / etc.)
    y       : (n,) int array, 1 = fraud
    proba   : (n,) float array, out-of-fold Stage 1+2+3 score
    dollars : (n,) float array, total $ moved in that trace
    """
    cfg = btp.CONFIG
    all_records = cwg.load_all_records(cfg)
    graph_connected_ids = cwg.get_graph_connected_trace_ids(all_records)
    df, A, connected_mask = cwg.build_feature_table_and_graph(all_records, graph_connected_ids)

    print(f"\nRunning 3-stage cascade, {cwg.N_SPLITS}-fold CV "
          f"(fresh GCN retrained per fold, takes a couple minutes)...")
    _, proba, y = cwg.run_three_stage_cascade(df, A, connected_mask, n_splits=cwg.N_SPLITS)

    # df rows are built by iterating all_records in order, with no
    # reordering afterwards (see build_feature_table_and_graph) --
    # positional alignment with all_records holds.
    dollars = np.array([
        sum(float(e["amount"]) for e in rec["events"] if e["event_type"] == "TRANSACTION")
        for rec in all_records
    ])
    assert len(dollars) == len(df) == len(proba) == len(y), "misaligned validation arrays"

    # Cache OOF scores so downstream analysis/plots don't have to pay for
    # a full cascade retrain (5-fold CV incl. a fresh GCN each fold) again.
    cache_path = Path(__file__).parent / "decision_policy_validation_cache.npz"
    np.savez(cache_path, y=y, proba=proba, dollars=dollars)

    return df, y, proba, dollars


def load_cached_validation_data():
    """Fast path for re-analysis: loads the (y, proba, dollars) cached by
    the last get_validation_data() run instead of retraining the cascade."""
    cache_path = Path(__file__).parent / "decision_policy_validation_cache.npz"
    data = np.load(cache_path)
    return data["y"], data["proba"], data["dollars"]


def get_validation_data_fused(random_state: int = cwg.RANDOM_STATE):
    """
    Same validation population as get_validation_data() (ring-overlay
    corpus via cwg.load_all_records -- kept identical so Decision Policy
    is scored on the same rows Stage 3/Stage 4 were validated against),
    but the score is Risk Fusion's stacked-LR output
    (risk_fusion.run_risk_fusion) instead of the raw max()-style
    Stage 1+2+3 cascade score. This is the score decision thresholds
    should actually be tuned against per the stated pipeline order
    (... -> GCN -> Autoencoder -> Risk Fusion -> Decision Policy -> ...).

    Also persists the fusion coefficients THIS run actually produced
    into decision_policy_results.json's "score_source" block, so
    downstream consumers (e.g. explainability.py) that need to explain
    *why* the fused score moved don't have to guess which fusion run's
    coefficients apply -- risk_fusion.py's own standalone main() run
    uses a DIFFERENT (no-ring-overlay) corpus and therefore produces
    different coefficients (notably gcn_score ~0 there, since no entity
    clears the fan-out threshold without the overlay).
    """
    import risk_fusion as rf

    cfg = btp.CONFIG
    all_records = cwg.load_all_records(cfg)
    graph_connected_ids = cwg.get_graph_connected_trace_ids(all_records)
    df, A, connected_mask = cwg.build_feature_table_and_graph(all_records, graph_connected_ids)

    print(f"\nRunning Risk Fusion (Stage 1+2 -> GCN -> Autoencoder -> stacked LR), "
          f"{rf.N_SPLITS}-fold CV on the ring-overlay validation population...")
    fusion_result, fused_proba, y = rf.run_risk_fusion(df, A, connected_mask, n_splits=rf.N_SPLITS)

    dollars = np.array([
        sum(float(e["amount"]) for e in rec["events"] if e["event_type"] == "TRANSACTION")
        for rec in all_records
    ])
    assert len(dollars) == len(df) == len(fused_proba) == len(y), "misaligned validation arrays"

    cache_path = Path(__file__).parent / "decision_policy_validation_cache.npz"
    np.savez(cache_path, y=y, proba=fused_proba, dollars=dollars)

    return df, y, fused_proba, dollars, fusion_result


# ---------------------------------------------------------------------------
# Step 2 -- explicit, labeled cost model
# ---------------------------------------------------------------------------
@dataclass
class CostModel:
    """
    Every number below is a PLACEHOLDER business assumption, not a
    measured constant. They are the actual knobs Risk/Finance need to own
    before this policy touches real money -- flagging that explicitly
    rather than burying it in code.
    """
    # Ops cost to manually review one case that lands in REVIEW (fraud or legit).
    review_ops_cost: float = 12.0
    # P(a human reviewer correctly stops a genuinely fraudulent case that
    # was routed to REVIEW). Reviewer sees more context than the model
    # (call the customer, check other signals) but isn't perfect.
    review_catch_rate: float = 0.85
    # Cost of wrongly BLOCKING a legitimate transaction outright: support
    # ticket + churn/reputation risk. Deliberately NOT the transaction's
    # own dollar value -- blocking doesn't destroy that money, it just
    # angers a real customer.
    legit_block_friction_cost: float = 150.0
    # Assumed TRUE production fraud prevalence, used to reweight this
    # ~19%-fraud red-team validation set down to a realistic prior before
    # costs are computed. Set to None to disable reweighting (uses the
    # validation set's own prevalence as-is -- useful for seeing the bug
    # reproduce, not for setting a real policy).
    assumed_production_fraud_rate: float | None = 0.006
    # Fraction of an APP-fraud trace's $ loss THIS institution (the
    # sending/customer-facing side, since that's whose behavioral
    # signals this model scores) is on the hook for if it's let through.
    # Modeled on the UK PSR's Authorised Push Payment reimbursement
    # requirement (in force since Oct 2024), which splits reimbursement
    # 50/50 between the sending PSP and the receiving PSP by default.
    # ATO and MULE_NETWORK are NOT split this way -- see liable_side()
    # and exposure_share() below for why.
    app_sending_liability_share: float = 0.5


# ---------------------------------------------------------------------------
# Role-aware liability model
# ---------------------------------------------------------------------------
# Which side of the payment bears responsibility for a given attack family,
# and therefore who should actually be alerted / act on a Review or Block
# decision. This is a policy/regulatory mapping, not a statistical one --
# it doesn't come from the data, it comes from how each fraud type is
# reimbursed in practice:
#
#   ATO (Account Takeover): the transaction was never authorised by the
#     customer at all. Under standard unauthorised-payment rules (PSD2 /
#     UK PSRs), the customer's OWN bank -- the sending institution -- must
#     refund it. There is no receiving-side liability share.
#
#   APP (Authorised Push Payment fraud): the customer WAS tricked into
#     authorising the payment themselves. The UK Payment Systems
#     Regulator's APP reimbursement requirement splits the cost 50/50
#     between the sending PSP and the receiving PSP by default, on the
#     logic that the receiving side had the chance to catch a mule/scam
#     account on the way in.
#
#   MULE_NETWORK: the fraud IS the receiving account -- a mule account
#     assembled to launder funds. Responsibility for having onboarded
#     and failed to flag that account sits with the RECEIVING
#     institution, not the sender.
LIABILITY_SIDE = {
    "ACCOUNT_TAKEOVER": "SENDING",
    "AUTHORIZED_PUSH_PAYMENT": "SHARED_50_50",
    "MULE_NETWORK": "RECEIVING",
}


def liable_side(attack_family: str) -> str:
    """Which side bears reimbursement liability for this attack family.
    Returns 'N/A' for legitimate traffic or any unrecognized family."""
    return LIABILITY_SIDE.get(attack_family, "N/A")


def acting_side(attack_family: str) -> str:
    """Which side should actually be alerted to act on a Review/Block
    decision for this attack family. For APP this is deliberately BOTH,
    even though liability is a 50/50 dollar split -- both PSPs have an
    independent chance to intervene before the payment settles."""
    side = liable_side(attack_family)
    if side == "SHARED_50_50":
        return "BOTH"
    if side == "N/A":
        return "N/A"
    return side


def exposure_share(attack_family: str, cost: "CostModel") -> float:
    """
    Fraction of a trace's $ loss THIS institution actually bears if that
    fraud is allowed through, given the liability split above. Only
    affects fraud-loss dollars (a fraud trace that gets through) -- the
    legit-block friction cost is always fully this institution's own
    problem regardless of attack family, since it's always this
    institution's own customer being inconvenienced.
    """
    if attack_family == "AUTHORIZED_PUSH_PAYMENT":
        return cost.app_sending_liability_share
    # ATO: full share (sending bank must reimburse in full).
    # MULE_NETWORK: modeled as full share too -- a sending bank isn't
    # shielded from its own liability just because the mule account
    # that received the funds happens to sit at another institution.
    return 1.0


def sample_weights(y: np.ndarray, target_fraud_rate: float | None) -> np.ndarray:
    """
    Importance weights that make the WEIGHTED sample's fraud prevalence
    equal target_fraud_rate, by scaling the fraud class down (this
    validation set over-samples fraud relative to production by
    construction, not the other way around). Legit weights stay at 1.
    """
    if target_fraud_rate is None:
        return np.ones(len(y))
    n_legit = int((y == 0).sum())
    n_fraud = int((y == 1).sum())
    if n_fraud == 0:
        return np.ones(len(y))
    w_fraud = (target_fraud_rate / (1 - target_fraud_rate)) * (n_legit / n_fraud)
    w = np.where(y == 1, w_fraud, 1.0)
    return w


# ---------------------------------------------------------------------------
# Step 3 -- expected cost of a (t_review, t_block) policy, vectorized
# ---------------------------------------------------------------------------
def expected_cost(
    y: np.ndarray, proba: np.ndarray, dollars: np.ndarray, weights: np.ndarray,
    t_review: float, t_block: float, cost: CostModel,
    attack_family: np.ndarray | None = None,
) -> float:
    """
    If attack_family is provided, fraud-loss dollars are scaled by
    exposure_share() before being costed -- i.e. this institution is
    only charged the fraction of the loss it's actually liable for
    under LIABILITY_SIDE (full share for ATO/MULE_NETWORK, the APP
    50/50 split's sending-side share for APP). Pass None to cost every
    fraud dollar at full share (the old, family-blind behavior).
    """
    assert t_review <= t_block
    is_block = proba >= t_block
    is_review = (proba >= t_review) & ~is_block
    is_allow = ~is_block & ~is_review

    fraud = y == 1

    if attack_family is not None:
        share = np.array([exposure_share(fam, cost) for fam in attack_family])
    else:
        share = np.ones(len(y))
    liable_dollars = dollars * share

    c = np.zeros(len(y))
    # ALLOW: legit costs 0; fraud costs its liable $ exposure (no recovery assumed)
    c[is_allow & fraud] = liable_dollars[is_allow & fraud]
    # REVIEW: ops cost for everyone reviewed, plus expected residual loss
    # for fraud the reviewer still misses
    c[is_review] = cost.review_ops_cost
    c[is_review & fraud] += liable_dollars[is_review & fraud] * (1 - cost.review_catch_rate)
    # BLOCK: fraud costs 0 (fully prevented); legit costs the friction cost
    # (friction is never shared -- it's always this institution's own customer)
    c[is_block & ~fraud] = cost.legit_block_friction_cost

    return float(np.sum(c * weights))


def liability_breakdown(
    y: np.ndarray, proba: np.ndarray, dollars: np.ndarray,
    attack_family: np.ndarray, t_review: float, t_block: float, cost: CostModel,
) -> dict:
    """
    Per-attack-family read-out of who is liable and who should act, for
    every case that lands in REVIEW or BLOCK (and for fraud $ that still
    gets through on ALLOW). This is the "AND which side should act"
    output referenced in the architecture: same Allow/Review/Block
    decision, but tagged with WHO -- sending, receiving, or both --
    needs to see the alert.
    """
    is_block = proba >= t_block
    is_review = (proba >= t_review) & ~is_block
    is_allow = ~is_block & ~is_review
    fraud = y == 1

    breakdown = {}
    for fam in sorted(set(attack_family[fraud])):
        fam_mask = (attack_family == fam) & fraud
        n_fam = int(fam_mask.sum())
        if n_fam == 0:
            continue
        side = liable_side(fam)
        share = exposure_share(fam, cost)
        # Derived from liable_side/share, never a second hardcoded family
        # string, so this can't drift out of sync with LIABILITY_SIDE again.
        if side == "SHARED_50_50":
            receiving_share = 1 - share
        elif side == "SENDING":
            receiving_share = 0.0
        elif side == "RECEIVING":
            receiving_share = 1.0
        else:  # N/A / unrecognized family
            receiving_share = 0.0
        breakdown[fam] = {
            "liable_side": side,
            "acting_side": acting_side(fam),
            "sending_liability_share": round(share, 3),
            "receiving_liability_share": round(receiving_share, 3),
            "n_fraud_traces": n_fam,
            "n_blocked": int((fam_mask & is_block).sum()),
            "n_reviewed": int((fam_mask & is_review).sum()),
            "n_allowed_through": int((fam_mask & is_allow).sum()),
            "dollars_allowed_through_total": round(float(dollars[fam_mask & is_allow].sum()), 2),
            "dollars_allowed_through_this_institution_liable_for": round(
                float(dollars[fam_mask & is_allow].sum() * share), 2
            ),
        }
    return breakdown


def policy_stats(
    y: np.ndarray, proba: np.ndarray, dollars: np.ndarray,
    t_review: float, t_block: float, cost: CostModel,
    attack_family: np.ndarray | None = None,
) -> dict:
    """Operational read-out for a given threshold pair (unweighted counts
    -- i.e. as observed on this validation set -- plus the reweighted
    expected cost, which is the number the thresholds were chosen to
    minimize). If attack_family is provided, also includes a
    liability_breakdown: who bears the $ cost and who should act, per
    attack family (see liable_side / acting_side / exposure_share)."""
    is_block = proba >= t_block
    is_review = (proba >= t_review) & ~is_block
    is_allow = ~is_block & ~is_review
    fraud = y == 1
    legit = ~fraud

    n = len(y)
    fraud_caught = int((is_block & fraud).sum()) + int((is_review & fraud).sum())
    # "caught" counts REVIEW fraud as caught at review_catch_rate, reported separately below
    weights = sample_weights(y, cost.assumed_production_fraud_rate)

    result = {
        "t_review": round(float(t_review), 4),
        "t_block": round(float(t_block), 4),
        "allow_rate": round(float(is_allow.mean()), 4),
        "review_rate": round(float(is_review.mean()), 4),
        "block_rate": round(float(is_block.mean()), 4),
        "legit_blocked": int((is_block & legit).sum()),
        "legit_blocked_rate": round(float((is_block & legit).sum() / max(legit.sum(), 1)), 4),
        "legit_reviewed_rate": round(float((is_review & legit).sum() / max(legit.sum(), 1)), 4),
        "fraud_blocked": int((is_block & fraud).sum()),
        "fraud_reviewed": int((is_review & fraud).sum()),
        "fraud_allowed": int((is_allow & fraud).sum()),
        "fraud_recall_blocked_only": round(float((is_block & fraud).sum() / max(fraud.sum(), 1)), 4),
        "fraud_recall_blocked_plus_review": round(float(fraud_caught / max(fraud.sum(), 1)), 4),
        "dollars_fraud_allowed_through": round(float(dollars[is_allow & fraud].sum()), 2),
        "dollars_fraud_total": round(float(dollars[fraud].sum()), 2),
        "expected_cost_at_assumed_prevalence": round(
            expected_cost(y, proba, dollars, weights, t_review, t_block, cost, attack_family), 2
        ),
    }
    if attack_family is not None:
        result["liability_breakdown"] = liability_breakdown(
            y, proba, dollars, attack_family, t_review, t_block, cost
        )
    return result


# ---------------------------------------------------------------------------
# Step 4 -- grid search over the two thresholds
# ---------------------------------------------------------------------------
def optimize_thresholds(
    y: np.ndarray, proba: np.ndarray, dollars: np.ndarray, cost: CostModel,
    n_candidates: int = 60, attack_family: np.ndarray | None = None,
) -> dict:
    """
    Two-threshold grid search minimizing reweighted expected cost.
    Candidates are drawn from the empirical score distribution (plus 0
    and 1) rather than an arbitrary linspace, so the grid actually has
    resolution where the scores live.

    If attack_family is provided, the cost being minimized is
    role-aware: fraud-loss dollars are pre-scaled by exposure_share()
    (full share for ATO/MULE_NETWORK, the APP 50/50 sending-side share
    for APP) BEFORE the threshold search runs, so the chosen thresholds
    already reflect that this institution isn't on the hook for the
    receiving side's half of an APP loss. The returned policy_stats
    still separately reports liability_breakdown for transparency.
    """
    weights = sample_weights(y, cost.assumed_production_fraud_rate)

    quantiles = np.linspace(0, 1, n_candidates)
    candidates = np.unique(np.concatenate([[0.0, 1.0], np.quantile(proba, quantiles)]))

    best = None
    for t_block in candidates:
        for t_review in candidates:
            if t_review > t_block:
                continue
            c = expected_cost(y, proba, dollars, weights, t_review, t_block, cost, attack_family)
            if best is None or c < best[0]:
                best = (c, t_review, t_block)

    cost_val, t_review, t_block = best
    result = policy_stats(y, proba, dollars, t_review, t_block, cost, attack_family)
    result["cost_model"] = asdict(cost)
    return result


# ---------------------------------------------------------------------------
# Step 5 -- diagnostic: reproduce and explain the block-everyone failure
# ---------------------------------------------------------------------------
def diagnose_prevalence_bug(
    y: np.ndarray, proba: np.ndarray, dollars: np.ndarray, cost: CostModel,
    attack_family: np.ndarray | None = None,
):
    naive_cost = CostModel(**{**asdict(cost), "assumed_production_fraud_rate": None})
    naive = optimize_thresholds(y, proba, dollars, naive_cost, attack_family=attack_family)
    corrected = optimize_thresholds(y, proba, dollars, cost, attack_family=attack_family)

    print("\n" + "=" * 72)
    print("DIAGNOSIS: naive (unweighted) vs prevalence-corrected optimum")
    print("=" * 72)
    n_fraud, n = int((y == 1).sum()), len(y)
    print(f"Validation set fraud prevalence: {n_fraud}/{n} = {n_fraud/n:.1%} "
          f"(red-team corpus, NOT the production prior)")
    print(f"Assumed production fraud prevalence used for correction: "
          f"{cost.assumed_production_fraud_rate:.2%}\n")

    for label, r in [("NAIVE (validation-set prevalence)", naive),
                     ("PREVALENCE-CORRECTED", corrected)]:
        print(f"-- {label} --")
        print(f"   thresholds: REVIEW >= {r['t_review']}, BLOCK >= {r['t_block']}")
        print(f"   allow/review/block rate: "
              f"{r['allow_rate']:.1%} / {r['review_rate']:.1%} / {r['block_rate']:.1%}")
        print(f"   legit blocked: {r['legit_blocked']} "
              f"({r['legit_blocked_rate']:.1%} of legit)")
        print(f"   fraud recall (block+review): {r['fraud_recall_blocked_plus_review']:.1%}")
        print()
    return naive, corrected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df, y, proba, dollars, fusion_result = get_validation_data_fused()
    cost = CostModel()
    attack_family = df["attack_family"].values

    naive, corrected = diagnose_prevalence_bug(y, proba, dollars, cost, attack_family)

    print("=" * 72)
    print("FINAL POLICY (prevalence-corrected, role-aware, scored on Risk Fusion output)")
    print("=" * 72)
    print(json.dumps(corrected, indent=2))

    print("\n" + "=" * 72)
    print("WHO SHOULD ACT (liable_side / acting_side by attack family)")
    print("=" * 72)
    for fam, row in corrected.get("liability_breakdown", {}).items():
        print(f"  {fam}: liable={row['liable_side']}  acting={row['acting_side']}  "
              f"sending_share={row['sending_liability_share']}  "
              f"$ this institution is liable for on ALLOW-through fraud: "
              f"{row['dollars_allowed_through_this_institution_liable_for']}")

    score_source = {
        "score": "risk_fusion_stacked_lr",
        "validation_population": "ring_overlay_corpus (cwg.load_all_records, "
                                  "same rows Stage 3/Stage 4 were validated on)",
        "fusion_avg_coefficients": fusion_result["fusion_avg_coefficients"],
        "n_graph_connected_nodes": fusion_result["n_graph_connected_nodes"],
        "note": "These coefficients come from THIS run (ring-overlay corpus), "
                "not from risk_fusion_results.json's real_corpus_risk_fusion "
                "block, which is a separate no-overlay run with a structurally "
                "different (near-zero) gcn_score weight. Consumers needing to "
                "explain why the fused score moved (e.g. explainability.py) "
                "should read fusion_avg_coefficients from here, not from "
                "risk_fusion_results.json's real-corpus block.",
    }

    output = {"naive": naive, "corrected": corrected, "score_source": score_source}
    from artifact_metadata import stamp_artifact
    output = stamp_artifact(
        output,
        Path(__file__).parent,
        seeds={"RANDOM_STATE": btp.CONFIG.get("RANDOM_STATE")},
    )
    out_path = Path(__file__).parent / "decision_policy_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
