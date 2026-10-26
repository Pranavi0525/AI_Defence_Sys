import os
import copy
from collections import Counter
import numpy as np

from red_team.world.world import NormalWorld
from red_team.attacks.corpus import generate_attack_corpus
from red_team.validation.novelty import extract_fingerprint, calculate_fingerprint_similarity

def main():
    world = NormalWorld(seed=42)
    world.generate_population(n_customers=10)
    base_state = world.get_state()

    print("================ STAGE 20 HARD INVESTIGATION ================\n")

    print("--- PART B: HARD CORPUS ANALYSIS ---")
    quotas = {"hard": 500}
    # We turn off novelty to get all raw generated HARD traces
    res = generate_attack_corpus(
        copy.deepcopy(base_state),
        master_seed=42,
        use_novelty=False,
        difficulty_quotas=quotas,
        max_attempts_multiplier=1
    )

    accepted = res.accepted_traces
    print(f"Generated {len(accepted)} HARD candidates.")

    phase_seqs = []
    event_seqs = []
    tx_counts = []
    amt_buckets = []
    norm_amts = []
    splits = []
    timings = []
    devices = []
    bens = []
    outcomes = []
    
    fps = []

    for r in accepted:
        fp = extract_fingerprint(r.observable_trace, r.ground_truth, base_state)
        fps.append(fp)
        phase_seqs.append(fp.phase_sequence)
        event_seqs.append(fp.event_sequence)
        tx_counts.append(fp.transaction_count)
        amt_buckets.append(fp.amount_buckets)
        norm_amts.append(round(fp.normalized_amount_sum, 1))
        splits.append(fp.split_count)
        timings.append(fp.timing_category)
        devices.append(fp.device_pattern)
        bens.append(fp.beneficiary_pattern)
        outcomes.append(fp.outcome_pattern)

    print("\nFrequency Distributions:")
    print("Phase Sequences:", len(set(phase_seqs)), "unique")
    print("Event Sequences:", len(set(event_seqs)), "unique")
    print("Amount Buckets:", Counter(amt_buckets))
    print("Normalized Amount Sums (rounded 0.1):", Counter(norm_amts))
    print("Splits:", Counter(splits))
    print("Timings:", Counter(timings))
    print("Devices:", Counter(devices))
    print("Beneficiaries:", Counter(bens))
    print("Outcomes:", Counter(outcomes))

    print("\n--- PART D: WHAT ACTUALLY VARIES? ---")
    print("Distinct Values per Dimension:")
    print(f"Phase Sequence: {len(set(phase_seqs))}")
    print(f"Event Sequence: {len(set(event_seqs))}")
    print(f"Amount Bucket: {len(set(amt_buckets))}")
    print(f"Normalized Amount: {len(set(norm_amts))}")
    print(f"Split Count: {len(set(splits))}")
    print(f"Timing Category: {len(set(timings))}")
    print(f"Device Pattern: {len(set(devices))}")
    print(f"Beneficiary Pattern: {len(set(bens))}")
    print(f"Outcome Pattern: {len(set(outcomes))}")

    # Determine what is constant
    constants = []
    if len(set(phase_seqs)) <= 1: constants.append("Phase Sequence")
    if len(set(event_seqs)) <= 1: constants.append("Event Sequence")
    if len(set(amt_buckets)) <= 1: constants.append("Amount Bucket")
    if len(set(norm_amts)) <= 1: constants.append("Normalized Amount")
    if len(set(splits)) <= 1: constants.append("Split Count")
    if len(set(timings)) <= 1: constants.append("Timing Category")
    if len(set(devices)) <= 1: constants.append("Device Pattern")
    if len(set(bens)) <= 1: constants.append("Beneficiary Pattern")
    if len(set(outcomes)) <= 1: constants.append("Outcome Pattern")
    
    print("Effectively Constant Dimensions (< 2 distinct values):", constants)

    print("\n--- PART E: FINGERPRINT SENSITIVITY ---")
    if fps:
        fp_base = fps[0]
        print("Base FP:", fp_base)
        
        def check_sim(fp_mod, name):
            sim = calculate_fingerprint_similarity(fp_base, fp_mod)
            print(f"Change {name} -> Similarity: {sim:.2f}")

        # Modify phase sequence
        fp_mod = fp_base.model_copy(update={"phase_sequence": tuple(list(fp_base.phase_sequence) + ["FAKE_PHASE"])})
        check_sim(fp_mod, "phase_sequence")
        
        fp_mod = fp_base.model_copy(update={"event_sequence": tuple(list(fp_base.event_sequence) + ["FAKE_EVENT"])})
        check_sim(fp_mod, "event_sequence")
        
        fp_mod = fp_base.model_copy(update={"amount_buckets": ("large",)})
        check_sim(fp_mod, "amount_buckets")
        
        fp_mod = fp_base.model_copy(update={"normalized_amount_sum": fp_base.normalized_amount_sum + 0.5})
        check_sim(fp_mod, "normalized_amount_sum (+0.5)")
        
        fp_mod = fp_base.model_copy(update={"split_count": fp_base.split_count + 5})
        check_sim(fp_mod, "split_count (which is actually not checked in similarity metric!)")
        
        fp_mod = fp_base.model_copy(update={"timing_category": "rapid"})
        check_sim(fp_mod, "timing_category")
        
        fp_mod = fp_base.model_copy(update={"device_pattern": ("known", "known", "known")})
        check_sim(fp_mod, "device_pattern")

        fp_mod = fp_base.model_copy(update={"beneficiary_pattern": ("new", "new")})
        check_sim(fp_mod, "beneficiary_pattern")
        
        fp_mod = fp_base.model_copy(update={"outcome_pattern": ("failed",)})
        check_sim(fp_mod, "outcome_pattern")

if __name__ == "__main__":
    main()
