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

    print("================ STAGE 21 HARD CORRECTION EXPERIMENT ================\n")

    print("--- PART 3: HARD-ONLY DIAGNOSTIC CORPUS ---")
    quotas = {"hard": 500}
    # Turn off novelty to see the raw generated traces
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
    
    fps = [extract_fingerprint(r.observable_trace, r.ground_truth, base_state) for r in accepted]
    
    exploitation_reached = 0

    for r in accepted:
        fp = extract_fingerprint(r.observable_trace, r.ground_truth, base_state)
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
        
        if any(p.phase == "EXPLOITATION" for p in r.ground_truth.phases_executed):
            exploitation_reached += 1

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
    print(f"Percentage reaching EXPLOITATION: {exploitation_reached / len(accepted):.2%}")
    
    print("\n--- PART 6: RE-RUN DIFFICULTY-BALANCED CORPUS ---")
    quotas_full = {
        "easy": 25,
        "medium": 25,
        "hard": 25,
        "advanced": 25
    }
    
    res_full = generate_attack_corpus(
        copy.deepcopy(base_state),
        master_seed=42,
        use_novelty=True,
        novelty_threshold=0.85,
        difficulty_quotas=quotas_full,
        max_attempts_multiplier=20
    )
    
    stats = res_full.generation_statistics
    
    for d in ["easy", "medium", "hard", "advanced"]:
        req = stats.requested_by_difficulty.get(d, 0)
        att = stats.attempted_by_difficulty.get(d, 0)
        acc = stats.accepted_by_difficulty.get(d, 0)
        nov_rej = stats.novelty_rejections_by_difficulty.get(d, 0)
        shortfall = stats.shortfall_by_difficulty.get(d, 0)
        status = stats.status_by_difficulty.get(d, "UNKNOWN")
        
        print(f"\n[{d.upper()}] Status: {status}")
        print(f"  Target: {req}")
        print(f"  Accepted: {acc}")
        print(f"  Shortfall: {shortfall}")
        print(f"  Attempts: {att} (Max Budget: {req * 20})")
        print(f"  Novelty Rejections: {nov_rej}")

    # Calculate HARD Near Dupes in full run
    hard_fps = [
        extract_fingerprint(r.observable_trace, r.ground_truth, base_state) 
        for r in res_full.accepted_traces 
        if r.ground_truth.attack_difficulty == "hard"
    ]
    hard_near_dup = 0
    total_hard_pairs = (len(hard_fps) * (len(hard_fps) - 1)) / 2 if len(hard_fps) > 1 else 0
    for i in range(len(hard_fps)):
        for j in range(i + 1, len(hard_fps)):
            if calculate_fingerprint_similarity(hard_fps[i], hard_fps[j]) >= 0.8:
                hard_near_dup += 1
                
    if total_hard_pairs > 0:
        print(f"\nHARD Near-duplicate rate (>=0.8 sim): {hard_near_dup / total_hard_pairs:.2%} ({hard_near_dup} pairs)")

if __name__ == "__main__":
    main()
