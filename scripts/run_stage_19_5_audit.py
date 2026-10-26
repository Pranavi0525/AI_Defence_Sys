import os
import copy
import statistics
import numpy as np

from red_team.world.world import NormalWorld
from red_team.attacks.corpus import generate_attack_corpus
from red_team.validation.novelty import extract_fingerprint

def main():
    world = NormalWorld(seed=42)
    world.generate_population(n_customers=10)
    base_state = world.get_state()

    print("================ STAGE 19.5 DIFFICULTY-BALANCED CORPUS EXPERIMENT ================\n")

    quotas = {
        "easy": 25,
        "medium": 25,
        "hard": 25,
        "advanced": 25
    }

    # Generate Corpus
    print("Generating Difficulty-Balanced Corpus with max_attempts_multiplier=20...")
    res = generate_attack_corpus(
        copy.deepcopy(base_state),
        master_seed=42,
        use_novelty=True,
        novelty_threshold=0.85,
        difficulty_quotas=quotas,
        max_attempts_multiplier=20
    )

    stats = res.generation_statistics

    print("\n--- PER-DIFFICULTY RESULTS ---")
    for d in ["easy", "medium", "hard", "advanced"]:
        req = stats.requested_by_difficulty.get(d, 0)
        att = stats.attempted_by_difficulty.get(d, 0)
        acc = stats.accepted_by_difficulty.get(d, 0)
        nov_rej = stats.novelty_rejections_by_difficulty.get(d, 0)
        real_rej = stats.realism_rejections_by_difficulty.get(d, 0)
        struct_rej = stats.structural_rejections_by_difficulty.get(d, 0)
        shortfall = stats.shortfall_by_difficulty.get(d, 0)
        status = stats.status_by_difficulty.get(d, "UNKNOWN")
        
        print(f"\n[{d.upper()}] Status: {status}")
        print(f"  Target: {req}")
        print(f"  Accepted: {acc}")
        print(f"  Shortfall: {shortfall}")
        print(f"  Attempts: {att} (Max Budget: {req * 20})")
        print(f"  Novelty Rejections: {nov_rej}")
        print(f"  Realism Rejections: {real_rej}")
        print(f"  Structural Rejections: {struct_rej}")
        
    print("\n--- DIVERSITY ANALYSIS ---")
    amounts = []
    timings = []
    splits = []
    
    fps = [extract_fingerprint(r.observable_trace, r.ground_truth, base_state) for r in res.accepted_traces]
    
    amounts = [f.normalized_amount_sum for f in fps]
    timings = [f.timing_category for f in fps]
    splits = [f.split_count for f in fps]
    
    print(f"Normalized Amount StdDev: {np.std(amounts) if amounts else 0:.4f}")
    print(f"Mean Split Count: {np.mean(splits) if splits else 0:.2f}")
    from collections import Counter
    print(f"Timing Distribution: {dict(Counter(timings))}")
    
    print("\n--- DUPLICATE ANALYSIS ---")
    from red_team.validation.novelty import calculate_fingerprint_similarity
    sim_threshold = 0.8
    near_dup = 0
    total_pairs = (len(fps) * (len(fps) - 1)) / 2
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            if calculate_fingerprint_similarity(fps[i], fps[j]) >= sim_threshold:
                near_dup += 1
                
    print(f"Near-duplicate rate (>=0.8 sim): {near_dup / total_pairs if total_pairs else 0:.2%} ({near_dup} pairs)")

if __name__ == "__main__":
    main()
