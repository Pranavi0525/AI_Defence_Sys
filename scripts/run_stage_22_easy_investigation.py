import os
import copy
from collections import Counter
import numpy as np

from red_team.world.world import NormalWorld
from red_team.attacks.corpus import generate_attack_corpus
from red_team.validation.novelty import extract_fingerprint, calculate_fingerprint_similarity

def main():
    world = NormalWorld(seed=42)
    world.generate_population(n_customers=20) # Need enough customers for large generation
    base_state = world.get_state()

    print("================ STAGE 22 EASY INVESTIGATION ================\n")

    print("--- 1. REPRODUCE THE EASY CEILING ---")
    quotas = {"easy": 100}
    res = generate_attack_corpus(
        copy.deepcopy(base_state),
        master_seed=42,
        use_novelty=True,
        difficulty_quotas=quotas,
        max_attempts_multiplier=20 # max_attempts = 2000
    )

    stats = res.generation_statistics
    
    print(f"Target: {quotas['easy']}")
    print(f"Accepted: {stats.accepted_by_difficulty.get('easy', 0)}")
    print(f"Attempts: {stats.attempted_by_difficulty.get('easy', 0)}")
    print(f"Novelty Rejections: {stats.novelty_rejections_by_difficulty.get('easy', 0)}")
    print(f"Realism Rejections: {stats.realism_rejections_by_difficulty.get('easy', 0)}")
    print(f"Structural Rejections: {stats.structural_rejections_by_difficulty.get('easy', 0)}")

    accepted_fps = [extract_fingerprint(r.observable_trace, r.ground_truth, base_state) for r in res.accepted_traces]
    unique_fps = len(set(fp.model_dump_json() for fp in accepted_fps))
    print(f"Unique Fingerprints: {unique_fps}")
    print(f"Unique Structural Paths (Phases): {stats.unique_phase_sequences}")

    print("\n--- 2. DETERMINE WHY EASY CANDIDATES COLLIDE ---")
    
    # We will generate without novelty to see the raw pool of 2000 attempts
    print("Generating raw EASY pool (2000 attempts) to measure collision space...")
    quotas_raw = {"easy": 2000}
    res_raw = generate_attack_corpus(
        copy.deepcopy(base_state),
        master_seed=42,
        use_novelty=False,
        difficulty_quotas=quotas_raw,
        max_attempts_multiplier=1
    )
    
    raw_fps = [extract_fingerprint(r.observable_trace, r.ground_truth, base_state) for r in res_raw.accepted_traces]
    
    print(f"Raw Accepted pool (no novelty): {len(raw_fps)}")
    
    print("Distributions among raw candidates:")
    print("Phase Sequences:", len(set(fp.phase_sequence for fp in raw_fps)))
    print("Event Sequences:", len(set(fp.event_sequence for fp in raw_fps)))
    print("Transaction Counts:", Counter(fp.transaction_count for fp in raw_fps))
    print("Amount Buckets:", Counter(fp.amount_buckets for fp in raw_fps))
    print("Normalized Amount Sums:", Counter(round(fp.normalized_amount_sum, 1) for fp in raw_fps))
    print("Split Counts:", Counter(fp.split_count for fp in raw_fps))
    print("Timing Categories:", Counter(fp.timing_category for fp in raw_fps))
    print("Device Patterns:", Counter(fp.device_pattern for fp in raw_fps))
    print("Beneficiary Patterns:", Counter(fp.beneficiary_pattern for fp in raw_fps))
    print("Outcome Patterns:", Counter(fp.outcome_pattern for fp in raw_fps))

    print("\n--- 6. COMPARE EASY WITH HARD ---")
    
    quotas_hard = {"hard": 2000}
    res_hard = generate_attack_corpus(
        copy.deepcopy(base_state),
        master_seed=42,
        use_novelty=False,
        difficulty_quotas=quotas_hard,
        max_attempts_multiplier=1
    )
    hard_fps = [extract_fingerprint(r.observable_trace, r.ground_truth, base_state) for r in res_hard.accepted_traces]
    
    def get_metrics(fps):
        return {
            "phases": len(set(fp.phase_sequence for fp in fps)),
            "events": len(set(fp.event_sequence for fp in fps)),
            "timings": len(set(fp.timing_category for fp in fps)),
            "amounts": len(set(round(fp.normalized_amount_sum, 1) for fp in fps)),
            "splits": len(set(fp.split_count for fp in fps)),
            "devices": len(set(fp.device_pattern for fp in fps)),
            "bens": len(set(fp.beneficiary_pattern for fp in fps)),
            "outcomes": len(set(fp.outcome_pattern for fp in fps)),
            "unique_fps": len(set(fp.model_dump_json() for fp in fps)),
            "exploit_rate": sum(1 for fp in fps if "EXPLOITATION" in fp.phase_sequence) / len(fps) if fps else 0
        }
        
    easy_m = get_metrics(raw_fps)
    hard_m = get_metrics(hard_fps)
    
    print(f"{'Metric':<25} | {'EASY':<10} | {'HARD':<10}")
    print("-" * 50)
    for k in easy_m:
        val_e = f"{easy_m[k]:.2f}" if isinstance(easy_m[k], float) else str(easy_m[k])
        val_h = f"{hard_m[k]:.2f}" if isinstance(hard_m[k], float) else str(hard_m[k])
        print(f"{k:<25} | {val_e:<10} | {val_h:<10}")

if __name__ == "__main__":
    main()
