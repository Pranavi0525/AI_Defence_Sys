import os
import copy
from collections import Counter
import statistics

from red_team.world.world import NormalWorld
from red_team.attacks.corpus import generate_attack_corpus

def analyze_corpus(corpus_result, name):
    accepted = corpus_result.accepted_traces
    print(f"\n================ {name} ================")
    print(f"Attempts: {corpus_result.generation_statistics.attempted}")
    print(f"Accepted: {len(accepted)}")
    print(f"Acceptance Rate: {len(accepted) / corpus_result.generation_statistics.attempted:.2%}")
    
    paths = [tuple(p.phase for p in r.ground_truth.phases_executed) for r in accepted]
    print(f"Unique structural paths: {len(set(paths))}")
    
    # Exact duplicates check
    # Trace ID and timestamps differ, but event ordering, amounts, etc.
    # To check exact, we check phase sequence + event sequence + exact amounts + exact gaps
    from red_team.validation.novelty import extract_fingerprint, calculate_fingerprint_similarity
    
    # Check near-duplicates using fingerprint sim >= 0.8
    sim_threshold = 0.8
    near_dup = 0
    total_pairs = (len(accepted) * (len(accepted) - 1)) / 2
    
    w = NormalWorld(seed=42)
    w.generate_population(10) # dummy world state to pass in for balance
    state = w.get_state()
    
    fps = [extract_fingerprint(r.observable_trace, r.ground_truth, state) for r in accepted]
    
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = calculate_fingerprint_similarity(fps[i], fps[j])
            if sim >= sim_threshold:
                near_dup += 1
                
    print(f"Near-duplicate rate (>=0.8 sim): {near_dup / total_pairs:.2%} ({near_dup} pairs)")

    # Diversity metrics
    amounts = [f.normalized_amount_sum for f in fps]
    splits = [f.split_count for f in fps]
    timings = [f.timing_category for f in fps]
    
    print(f"Amount stddev: {statistics.stdev(amounts) if len(amounts)>1 else 0:.4f}")
    print(f"Split count mean: {statistics.mean(splits):.2f}")
    print(f"Timing patterns: {dict(Counter(timings))}")
    
    if hasattr(accepted[0], "novelty") and accepted[0].novelty:
        scores = [r.novelty.novelty_score for r in accepted]
        print(f"Mean Novelty Score: {statistics.mean(scores):.4f}")

def main():
    world = NormalWorld(seed=42)
    world.generate_population(n_customers=10)
    base_state = world.get_state()
    
    # Corpus A: Without novelty
    res_a = generate_attack_corpus(
        copy.deepcopy(base_state), 
        target_count=100, 
        master_seed=42, 
        max_attempts=5000,
        use_novelty=False
    )
    analyze_corpus(res_a, "Corpus A (No Novelty)")
    
    # Corpus B: With novelty
    res_b = generate_attack_corpus(
        copy.deepcopy(base_state), 
        target_count=100, 
        master_seed=42, 
        max_attempts=5000,
        use_novelty=True,
        novelty_threshold=0.85
    )
    analyze_corpus(res_b, "Corpus B (With Novelty)")

if __name__ == "__main__":
    main()
