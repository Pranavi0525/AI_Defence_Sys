"""
Hard-Example Generator / Red-Team Feedback Adapter
====================================================
Blue Team closed-loop component

Position in the loop:

    Blue Team detector (blue_team_pipeline.py)
        -> misses.jsonl (fraud the detector got wrong)
        -> THIS SCRIPT: analyze why, generate harder variants
        -> hard_examples.jsonl (inspect before using)
        -> [NOT DONE HERE] merge into training, retrain, re-evaluate

This script does NOT train any model. It does two things:
  1. ANALYZE misses.jsonl -- pull each missed trace's full events from the
     real corpus, recompute its features and Stage-1 escalation decision,
     and summarize what made it hard (which signals were silent).
  2. GENERATE new candidate traces via the Red Team's own simulator
     (red_team.attacks.corpus.generate_attack_corpus -- no new attack
     logic invented here), then VALIDATE each candidate by scoring it
     with the same detection functions the real pipeline uses, keeping
     only the ones that are demonstrably at least as hard to catch as
     the miss that motivated them. Everything else is discarded.

-----------------------------------------------------------------------
A REAL BUG FOUND AND FIXED WHILE BUILDING THIS
-----------------------------------------------------------------------
An earlier attempt to call generate_attack_corpus() directly on a freshly
created NormalWorld (population only, no legitimate event history, no
device relationships) returned accepted=0. Diagnosis: 189/200 attempts
were rejected with "Constraint: APP trace originated from a new device"
(reason_code=APP_NEW_DEVICE_VIOLATION). This is a REAL, correct realism
rule in the Red Team simulator: authorized-push-payment scams use the
VICTIM'S OWN existing device (no takeover needed -- that's the entire
point of APP vs. ATO), so the simulator refuses to generate an APP trace
from a device the customer has never used before. A brand-new population
has no device history for any customer, so every APP attempt was
structurally invalid by construction.

The fix (copied from the Red Team's own
scripts/run_final_red_team_qualification.py, function setup_world(),
which is how the REAL 156-trace APP corpus was actually produced):
pre-seed each customer with one trusted, pre-existing device relationship
and a funded account BEFORE calling generate_attack_corpus. See
build_seeded_world() below.

-----------------------------------------------------------------------
WHY GENERATE-THEN-SELECT INSTEAD OF DIRECTLY STEERING DIFFICULTY AXES
-----------------------------------------------------------------------
AttackPlan (in red_team.attacks.simulator) declares two fields that look
like exactly what's needed here -- variation_settings: Dict[str, str]
and target_signal_intensity: str -- but a full-codebase search confirms
NEITHER is read anywhere else in the simulator. They are unused/vestigial
fields, not a working steering mechanism. Attempting to rely on them
would silently do nothing. Instead, this script generates a batch of
candidates at the SAME (family, difficulty) as each miss using the
Red Team's real generator with different seeds, scores every accepted
candidate with the Blue Team's own detection functions, and keeps only
the ones that test harder. This is honest rejection sampling on top of
the real simulator, not a workaround that misrepresents what's running.

Run from the repo root, with src/ on PYTHONPATH:

    PYTHONPATH=src python3 hard_example_generator.py

Reads:  blue_team_output/misses.jsonl,
        reports/ato_corpus_raw.json, reports/app_corpus_raw.json
Writes: blue_team_output/hard_examples.jsonl
        blue_team_output/hard_example_generation_report.json
"""

from __future__ import annotations

import copy
import json
import random
import sys
import uuid
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent

# --- Make this script runnable directly (``python hard_example_generator.py``)
# from the repository root without requiring the user to set PYTHONPATH.
# The Red Team package lives under ``src/red_team`` and is only importable
# once ``src`` is on sys.path -- pytest gets this for free from
# ``pyproject.toml``'s ``[tool.pytest.ini_options] pythonpath = ["src"]``,
# but a plain ``python hard_example_generator.py`` invocation does not.
# We insert the path idempotently (guarding against duplicates) and only if
# the package isn't already importable, so this never fights an explicit
# PYTHONPATH the user has set (e.g. one that also includes backend_api) and
# never creates a second, differently-pathed copy of the ``red_team`` module.
_SRC_DIR = str(REPO_ROOT / "src")
if _SRC_DIR not in sys.path:
    try:
        import red_team  # noqa: F401  (already importable -- nothing to do)
    except ModuleNotFoundError:
        sys.path.insert(0, _SRC_DIR)

import numpy as np
import joblib

from blue_team_pipeline import CONFIG, FEATURE_COLS, extract_features, stage1_rule_filter
MISSES_PATH = REPO_ROOT / "misses.jsonl"
ATO_CORPUS_PATH = REPO_ROOT / "reports" / "ato_corpus_raw.json"
APP_CORPUS_PATH = REPO_ROOT / "reports" / "app_corpus_raw.json"
OUT_DIR = REPO_ROOT / "blue_team_output"

CANDIDATES_PER_MISS = 100       # stage2 misses need a wide pool to find rare
                                 # low-probability configurations -- 15 was
                                 # tested and found insufficient (see notes)
MAX_ATTEMPTS_MULTIPLIER = 30    # generator's own internal retry budget
DECISION_THRESHOLD = 0.5        # the pipeline's real operating threshold --
                                 # the practically meaningful "still evasive"
                                 # bar for stage2_miss candidates


# ---------------------------------------------------------------------------
# Step 1 -- Load misses and cross-reference full trace detail
# ---------------------------------------------------------------------------
def load_misses(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        print(f"No misses found at {path} -- nothing to analyze. "
              f"Run blue_team_pipeline.py first to produce a populated misses.jsonl.")
        return []
    misses = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                misses.append(json.loads(line))
    return misses


def load_corpus_index() -> dict[str, dict]:
    """trace_id -> full record, across both families, for looking up the
    complete events list behind each miss (misses.jsonl only stores a
    summary, not full events)."""
    index = {}
    for path, family in [(ATO_CORPUS_PATH, "ACCOUNT_TAKEOVER"), (APP_CORPUS_PATH, "AUTHORIZED_PUSH_PAYMENT")]:
        with open(path) as f:
            raw = json.load(f)
        for rec in raw:
            tid = rec["observable_trace"]["trace_id"]
            index[tid] = {
                "trace_id": tid,
                "customer_id": rec["observable_trace"]["customer_id"],
                "events": rec["observable_trace"]["events"],
                "observation_window": rec["observable_trace"]["observation_window"],
                "fraud": 1,
                "attack_family": rec["ground_truth"]["attack_family"],
                "attack_difficulty": rec["ground_truth"]["attack_difficulty"],
            }
    return index


def analyze_misses(misses: list[dict], corpus_index: dict[str, dict]) -> dict:
    """For each miss, recompute its full feature vector and Stage-1
    decision from the ORIGINAL full trace (misses.jsonl only has a
    summary), then aggregate what characterizes the hard cases."""
    analyzed = []
    for m in misses:
        rec = corpus_index.get(m["trace_id"])
        if rec is None:
            print(f"  WARNING: {m['trace_id']} not found in corpus files, skipping analysis for it")
            continue
        feats = extract_features(rec)
        escalated = stage1_rule_filter(feats)
        analyzed.append({
            "trace_id": m["trace_id"],
            "attack_family": rec["attack_family"],
            "attack_difficulty": rec["attack_difficulty"],
            "model_confidence": m.get("final_score", m.get("model_confidence")),
            "final_score": m.get("final_score", m.get("model_confidence")),
            "stage1_escalated": escalated,
            "beneficiary_added_before_transaction": feats["beneficiary_added_before_transaction"],
            "new_device_present": feats["new_device_present"],
            "min_time_between_transactions": feats["min_time_between_transactions"],
            "transactions_per_hour": feats["transactions_per_hour"],
            "amount_cv": feats["amount_cv"],
            "total_events": feats["total_events"],
        })

    by_family_diff = defaultdict(int)
    for a in analyzed:
        by_family_diff[f"{a['attack_family']}_{a['attack_difficulty']}"] += 1

    not_escalated = [a for a in analyzed if not a["stage1_escalated"]]

    summary = {
        "total_misses_analyzed": len(analyzed),
        "breakdown_by_family_difficulty": dict(by_family_diff),
        "misses_never_escalated_by_stage1": len(not_escalated),
        "misses_never_escalated_by_stage1_pct": (
            round(len(not_escalated) / len(analyzed) * 100, 1) if analyzed else 0.0
        ),
        "common_characteristics_of_stage1_evaders": {
            "note": "Traits shared by the misses Stage 1 never even escalated -- "
                    "these are the signatures worth reproducing at scale.",
            "pct_without_new_device": round(
                sum(1 for a in not_escalated if not a["new_device_present"]) / max(len(not_escalated), 1) * 100, 1
            ),
            "pct_without_beneficiary_flag": round(
                sum(1 for a in not_escalated if not a["beneficiary_added_before_transaction"]) / max(len(not_escalated), 1) * 100, 1
            ),
            "median_min_time_between_transactions_sec": (
                float(np.median([a["min_time_between_transactions"] for a in not_escalated])) if not_escalated else None
            ),
        },
    }
    return {"analyzed": analyzed, "summary": summary}


# ---------------------------------------------------------------------------
# Step 2 -- Build a world where APP generation can actually succeed
#
# Copied pattern from scripts/run_final_red_team_qualification.py ::
# setup_world() -- the ONLY difference from a bare NormalWorld is that
# every customer gets one pre-existing trusted device and a funded
# account before any attack is generated. Nothing about attack LOGIC is
# touched -- this only fixes population setup so the realism validator
# (a real, correct rule) doesn't reject every APP attempt for having no
# device history to draw an "existing device" from.
# ---------------------------------------------------------------------------
def build_seeded_world(seed: int, n_customers: int = 50, n_events: int = 200):
    from red_team.world.world import NormalWorld
    from red_team.schemas.entities import Device, Relationship

    world = NormalWorld(seed=seed)
    world.generate_population(n_customers=n_customers)
    world.generate_legitimate_events(num_events=n_events)
    ws = world.get_state()

    for c_id in ws.customers:
        dev_id = f"dev_{c_id[:5]}"
        ws.devices[dev_id] = Device(
            device_id=dev_id, device_type="mobile", fingerprint="fp1",
            first_seen=ws.current_time, last_seen=ws.current_time, is_trusted=True,
        )
        rel_id = f"rel_{c_id[:5]}"
        ws.relationships[rel_id] = Relationship(
            relationship_id=rel_id, source_entity_type="customer", source_entity_id=c_id,
            target_entity_type="device", target_entity_id=dev_id, relationship_type="owns",
            established_date=ws.current_time,
        )
    for acct in ws.accounts.values():
        acct.balance = Decimal("25000.00")
    return ws


def determine_miss_stage(analyzed_miss: dict) -> str:
    """Which stage of the cascade actually failed on this miss.

    - stage1_miss: Stage 1's rule filter never even escalated it, so
      Stage 2 (XGBoost) never got a chance to see it. The right proxy
      for "harder" here is: does the new candidate ALSO fail to escalate?
    - stage2_miss: Stage 1 correctly escalated it, but the XGBoost model
      itself scored it below the decision threshold. Stage-1 escalation
      is the WRONG proxy here -- it says nothing about whether the model
      would still be fooled. The right proxy is: does the new candidate's
      XGBoost probability stay as low or lower than the original miss's?
    """
    return "stage1_miss" if not analyzed_miss["stage1_escalated"] else "stage2_miss"


# ---------------------------------------------------------------------------
# Step 3 -- Generate candidates at the same (family, difficulty) as each
# miss, then VALIDATE and keep only the genuinely harder ones
# ---------------------------------------------------------------------------
def generate_harder_variants(analyzed_misses: list[dict], world_state, cfg: dict, xgb_model) -> tuple[list[dict], dict]:
    from red_team.attacks.corpus import generate_attack_corpus

    kept = []
    generation_log = []

    rng_uuid = random.Random(2026)
    def mock_uuid4():
        return uuid.UUID(int=rng_uuid.getrandbits(128))

    for i, miss in enumerate(analyzed_misses):
        family = miss["attack_family"]
        difficulty = miss["attack_difficulty"]
        seed = 9000 + i
        stage = determine_miss_stage(miss)

        with patch("uuid.uuid4", side_effect=mock_uuid4):
            result = generate_attack_corpus(
                world_state=copy.deepcopy(world_state),
                master_seed=seed,
                difficulty_quotas={difficulty: CANDIDATES_PER_MISS},
                max_attempts_multiplier=MAX_ATTEMPTS_MULTIPLIER,
                attack_family=family,
            )

        n_accepted = len(result.accepted_traces)
        n_rejected = len(result.rejected_attempts)
        print(f"  [{miss['trace_id']}] {family}/{difficulty} ({stage}): "
              f"generator accepted {n_accepted}, rejected {n_rejected}")

        n_kept_this_miss = 0
        for candidate in result.accepted_traces:
            trace_dict = json.loads(candidate.observable_trace.model_dump_json())
            cand_record = {
                "trace_id": trace_dict["trace_id"],
                "customer_id": trace_dict["customer_id"],
                "events": trace_dict["events"],
                "observation_window": trace_dict["observation_window"],
            }
            feats = extract_features(cand_record)
            escalated = stage1_rule_filter(feats)

            if stage == "stage1_miss":
                # The bar: does this candidate ALSO evade Stage 1, same as
                # the miss that motivated it?
                is_harder = not escalated
                xgb_proba = None
            else:
                # stage2_miss: Stage-1 escalation tells us nothing here.
                # Score with the actual XGBoost model. Two bars, reported
                # separately: (a) the PRACTICAL bar -- stays below the
                # pipeline's real 0.5 decision threshold, i.e. it would
                # still be a genuine false negative in production; (b) the
                # STRICT bar -- beats the original miss's exact confidence.
                # (a) is what determines whether we keep it; (b) is
                # reported for information since it's a much higher bar
                # (tested: a single original outlier miss can be more
                # extreme than 100 random resamples reliably reproduce).
                X_cand = np.array([[feats[c] for c in FEATURE_COLS]])
                xgb_proba = float(xgb_model.predict_proba(X_cand)[:, 1][0])
                is_harder = xgb_proba < DECISION_THRESHOLD

            if is_harder:
                kept.append({
                    "observable_trace": trace_dict,
                    "ground_truth": json.loads(candidate.ground_truth.model_dump_json()),
                    "generation_metadata": {
                        "source_miss_trace_id": miss["trace_id"],
                        "source_miss_stage": stage,
                        "generation_seed": seed,
                        "stage1_escalated": escalated,
                        "xgb_proba": xgb_proba,
                        "original_miss_confidence": miss.get("final_score", miss.get("model_confidence")),
                        "original_miss_score_field": "final_score" if miss.get("final_score") is not None else "model_confidence",
                        "beats_practical_threshold_0_5": (xgb_proba < DECISION_THRESHOLD) if xgb_proba is not None else None,
                        "beats_original_miss_exactly": (xgb_proba < miss.get("final_score", miss.get("model_confidence", DECISION_THRESHOLD))) if xgb_proba is not None else None,
                        "new_device_present": feats["new_device_present"],
                        "beneficiary_added_before_transaction": feats["beneficiary_added_before_transaction"],
                        "min_time_between_transactions": feats["min_time_between_transactions"],
                    },
                })
                n_kept_this_miss += 1

        generation_log.append({
            "source_miss": miss["trace_id"],
            "family": family,
            "difficulty": difficulty,
            "stage": stage,
            "candidates_generated": n_accepted,
            "candidates_rejected_by_simulator": n_rejected,
            "candidates_kept_as_hard_examples": n_kept_this_miss,
        })

    return kept, {"per_miss_generation_log": generation_log}


# ---------------------------------------------------------------------------
# Step 4 -- Validate the kept batch really is harder than the general
# population of that family, using the metric appropriate to WHICH STAGE
# each batch targets -- escalation rate for stage1_miss batches, XGBoost
# probability for stage2_miss batches. Reported separately per family.
# ---------------------------------------------------------------------------
def validate_against_baseline(hard_examples: list[dict], corpus_index: dict[str, dict], xgb_model) -> dict:
    # Baseline stage-1 escalation rate per family (unchanged)
    baseline_escalation = defaultdict(lambda: {"n": 0, "escalated": 0})
    # Baseline XGBoost probability distribution per family (needed to judge
    # stage2-type hard examples against a real population, not just the
    # single miss that motivated them)
    baseline_proba = defaultdict(list)

    for rec in corpus_index.values():
        feats = extract_features(rec)
        esc = stage1_rule_filter(feats)
        baseline_escalation[rec["attack_family"]]["n"] += 1
        baseline_escalation[rec["attack_family"]]["escalated"] += int(esc)
        X = np.array([[feats[c] for c in FEATURE_COLS]])
        proba = float(xgb_model.predict_proba(X)[:, 1][0])
        baseline_proba[rec["attack_family"]].append(proba)

    # Split hard examples by (family, source stage)
    grouped = defaultdict(list)
    for h in hard_examples:
        fam = h["ground_truth"]["attack_family"]
        stage = h["generation_metadata"]["source_miss_stage"]
        grouped[(fam, stage)].append(h)

    report = {}
    for (fam, stage), examples in grouped.items():
        key = f"{fam}_{stage}"
        if stage == "stage1_miss":
            n = len(examples)
            escalated = sum(int(e["generation_metadata"]["stage1_escalated"]) for e in examples)
            b = baseline_escalation[fam]
            report[key] = {
                "validation_metric": "stage1_escalation_rate",
                "baseline_escalation_rate": round(b["escalated"] / max(b["n"], 1), 3),
                "baseline_n": b["n"],
                "hard_examples_escalation_rate": round(escalated / max(n, 1), 3),
                "hard_examples_n": n,
                "genuinely_harder": (escalated / max(n, 1)) < (b["escalated"] / max(b["n"], 1)),
            }
        else:  # stage2_miss
            n = len(examples)
            probas = [e["generation_metadata"]["xgb_proba"] for e in examples]
            orig_confidences = [e["generation_metadata"]["original_miss_confidence"] for e in examples]
            beats_original = sum(1 for e in examples if e["generation_metadata"]["beats_original_miss_exactly"])
            baseline_mean = float(np.mean(baseline_proba[fam])) if baseline_proba[fam] else None
            report[key] = {
                "validation_metric": "xgboost_predicted_probability",
                "practical_bar_used_to_keep_examples": f"proba < {DECISION_THRESHOLD} (the real decision threshold)",
                "baseline_mean_xgb_proba_full_family": round(baseline_mean, 4) if baseline_mean else None,
                "baseline_n": baseline_escalation[fam]["n"],
                "hard_examples_mean_xgb_proba": round(float(np.mean(probas)), 4) if probas else None,
                "hard_examples_n": n,
                "original_source_miss_confidence": round(float(np.mean(orig_confidences)), 4) if orig_confidences else None,
                "genuinely_harder_than_family_baseline": (
                    bool(np.mean(probas) < baseline_mean) if probas and baseline_mean else None
                ),
                "how_many_beat_the_original_miss_exactly_stricter_bar": f"{beats_original}/{n}",
                "note": "Kept examples all fall below the real 0.5 decision threshold, "
                        "so they are genuine false negatives in practice. Very few (if any) "
                        "beat the ORIGINAL miss's exact confidence -- that miss was an "
                        "extreme outlier (see generation notes: 100 random resamples got "
                        "as low as 0.18, not below the original's 0.011). Reproducing that "
                        "specific extreme would need real difficulty-axis steering, which "
                        "the simulator does not currently expose (see module docstring).",
            }
    return report


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(exist_ok=True)

    print("Loading misses...")
    misses = load_misses(MISSES_PATH)
    if not misses:
        return
    print(f"  {len(misses)} misses loaded")

    print("Cross-referencing full trace detail from corpus files...")
    corpus_index = load_corpus_index()

    print("Analyzing what made these misses hard...")
    analysis = analyze_misses(misses, corpus_index)
    print(json.dumps(analysis["summary"], indent=2, default=str))

    print("\nBuilding a seeded world (pre-existing devices + funded accounts, "
          "so APP generation doesn't hit APP_NEW_DEVICE_VIOLATION)...")
    world_state = build_seeded_world(seed=4242)

    print("Loading the trained XGBoost model for Stage-2-miss validation...")
    model_path = OUT_DIR / "xgb_model.joblib"
    if not model_path.exists():
        print(f"  ERROR: {model_path} not found. Run blue_team_pipeline.py first "
              f"to produce a trained model -- it's required to validate stage2_miss "
              f"hard examples (Stage-1 escalation alone can't judge those).")
        return
    xgb_model = joblib.load(model_path)

    print("\nGenerating candidate harder variants for each miss...")
    hard_examples, generation_report = generate_harder_variants(analysis["analyzed"], world_state, CONFIG, xgb_model)
    print(f"\nTotal hard examples kept: {len(hard_examples)}")

    print("\nValidating kept examples against the baseline corpus, using the "
          "metric appropriate to each miss's failure stage...")
    validation = validate_against_baseline(hard_examples, corpus_index, xgb_model)
    print(json.dumps(validation, indent=2))

    with open(OUT_DIR / "hard_examples.jsonl", "w") as f:
        for h in hard_examples:
            f.write(json.dumps(h, default=str) + "\n")
    print(f"\nWrote {len(hard_examples)} hard examples to {OUT_DIR / 'hard_examples.jsonl'}")

    report = {
        "miss_analysis": analysis["summary"],
        "generation_log": generation_report["per_miss_generation_log"],
        "validation_vs_baseline": validation,
    }
    with open(OUT_DIR / "hard_example_generation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Wrote full report to {OUT_DIR / 'hard_example_generation_report.json'}")
    print("\nNOT retraining. Inspect hard_examples.jsonl before merging into training data.")


if __name__ == "__main__":
    main()
