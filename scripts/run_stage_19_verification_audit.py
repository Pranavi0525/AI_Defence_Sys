import os
import copy
import statistics
from collections import Counter
import numpy as np

from red_team.world.world import NormalWorld
from red_team.attacks.corpus import generate_attack_corpus
from red_team.validation.novelty import extract_fingerprint

def main():
    world = NormalWorld(seed=42)
    world.generate_population(n_customers=10)
    base_state = world.get_state()

    print("================ STAGE 19 VERIFICATION AUDIT ================\n")

    # 1. Corpus A: No Novelty
    print("Generating Corpus A (No Novelty)...")
    res_a = generate_attack_corpus(
        copy.deepcopy(base_state),
        target_count=100,
        master_seed=42,
        max_attempts=5000,
        use_novelty=False
    )
    
    # 2. Corpus B: With Novelty
    print("Generating Corpus B (With Novelty)...")
    res_b = generate_attack_corpus(
        copy.deepcopy(base_state),
        target_count=100,
        master_seed=42,
        max_attempts=5000,
        use_novelty=True,
        novelty_threshold=0.85
    )

    # 3. Reproducibility Check
    print("Generating Corpus C (Reproducibility)...")
    res_c = generate_attack_corpus(
        copy.deepcopy(base_state),
        target_count=100,
        master_seed=42,
        max_attempts=5000,
        use_novelty=True,
        novelty_threshold=0.85
    )

    # A. Rejection Accounting
    print("\n--- A. REJECTION ACCOUNTING (Corpus B) ---")
    total_attempts = res_b.generation_statistics.attempted
    accepted_count = len(res_b.accepted_traces)
    
    structural_rejections = 0
    novelty_rejections = 0
    validation_rejections = 0
    simulation_errors = 0
    other_rejections = 0
    
    for rej in res_b.rejected_attempts:
        cat = rej.get("failure_category", "unknown")
        if cat == "structural_rejection":
            structural_rejections += 1
        elif cat == "novelty_rejection":
            novelty_rejections += 1
        elif cat == "validation_rejection":
            validation_rejections += 1
        elif cat == "simulation_error":
            simulation_errors += 1
        else:
            other_rejections += 1

    total_sum = accepted_count + structural_rejections + novelty_rejections + validation_rejections + simulation_errors + other_rejections
    print(f"Total Attempts: {total_attempts}")
    print(f"Accepted: {accepted_count}")
    print(f"Structural Rejections: {structural_rejections}")
    print(f"Novelty Rejections: {novelty_rejections}")
    print(f"Realism Rejections (Validation): {validation_rejections}")
    print(f"Simulation Errors: {simulation_errors}")
    print(f"Other Rejections: {other_rejections}")
    print(f"Sum matches total attempts: {total_sum == total_attempts}")

    # B & C & D Helper function
    def analyze_corpus(res, state_ref):
        diff_stats = {"easy": [], "medium": [], "hard": [], "advanced": []}
        timings = []
        
        amounts = []
        norm_amounts = []
        splits = []
        novelty_scores = []
        
        for r in res.accepted_traces:
            d = r.ground_truth.attack_difficulty
            diff_stats[d].append(r)
            
            fp = extract_fingerprint(r.observable_trace, r.ground_truth, state_ref)
            timings.append(fp.timing_category)
            splits.append(fp.split_count)
            norm_amounts.append(fp.normalized_amount_sum)
            
            for e in r.observable_trace.events:
                if e.event_type == "TRANSACTION":
                    amounts.append(float(getattr(e, "amount", 0)))
                    
            if hasattr(r, "novelty") and getattr(r, "novelty") is not None:
                novelty_scores.append(r.novelty.novelty_score)
                
        return diff_stats, timings, amounts, norm_amounts, splits, novelty_scores

    diff_a, timings_a, amt_a, norm_a, splits_a, _ = analyze_corpus(res_a, base_state)
    diff_b, timings_b, amt_b, norm_b, splits_b, nov_b = analyze_corpus(res_b, base_state)

    print("\n--- B. DIFFICULTY DISTRIBUTION ---")
    def print_diff(name, diff_stats):
        print(f"\n{name}:")
        total = sum(len(v) for v in diff_stats.values())
        for d in ["easy", "medium", "hard", "advanced"]:
            traces = diff_stats[d]
            if not traces:
                print(f"  {d.upper()}: 0 (0.00%)")
                continue
            pct = len(traces) / total
            phases = [len(t.ground_truth.phases_executed) for t in traces]
            events = [len(t.observable_trace.events) for t in traces]
            txs = [sum(1 for e in t.observable_trace.events if e.event_type=="TRANSACTION") for t in traces]
            fp_splits = [extract_fingerprint(t.observable_trace, t.ground_truth, base_state).split_count for t in traces]
            print(f"  {d.upper()}: {len(traces)} ({pct:.2%}) | Ph: {statistics.mean(phases):.2f} | Ev: {statistics.mean(events):.2f} | Tx: {statistics.mean(txs):.2f} | Splt: {statistics.mean(fp_splits):.2f}")

    print_diff("Without Novelty (Corpus A)", diff_a)
    print_diff("With Novelty (Corpus B)", diff_b)

    print("\n--- C. TIMING DISTRIBUTION ---")
    print(f"Without Novelty: {dict(Counter(timings_a))}")
    print(f"With Novelty: {dict(Counter(timings_b))}")

    print("\n--- D. BEHAVIORAL VARIATION ---")
    print(f"Without Novelty: Amt Std={np.std(amt_a):.2f}, Norm Amt Std={np.std(norm_a):.4f}, Mean Splits={np.mean(splits_a):.2f}")
    print(f"With Novelty: Amt Std={np.std(amt_b):.2f}, Norm Amt Std={np.std(norm_b):.4f}, Mean Splits={np.mean(splits_b):.2f}")

    print("\n--- E. NOVELTY SCORE DISTRIBUTION (Corpus B) ---")
    if nov_b:
        print(f"Min: {np.min(nov_b):.4f}")
        print(f"Max: {np.max(nov_b):.4f}")
        print(f"Mean: {np.mean(nov_b):.4f}")
        print(f"Median: {np.median(nov_b):.4f}")
        print(f"p10: {np.percentile(nov_b, 10):.4f}")
        print(f"p25: {np.percentile(nov_b, 25):.4f}")
        print(f"p75: {np.percentile(nov_b, 75):.4f}")
        print(f"p90: {np.percentile(nov_b, 90):.4f}")
    else:
        print("No novelty scores found.")

    print("\n--- J. REPRODUCIBILITY ---")
    diff_c, timings_c, amt_c, norm_c, splits_c, nov_c = analyze_corpus(res_c, base_state)
    print(f"Corpus B accepted: {len(res_b.accepted_traces)}")
    print(f"Corpus C accepted: {len(res_c.accepted_traces)}")
    print(f"Trace counts match: {len(res_b.accepted_traces) == len(res_c.accepted_traces)}")
    print(f"Attempts match: {res_b.generation_statistics.attempted == res_c.generation_statistics.attempted}")
    print(f"Timings exact match: {timings_b == timings_c}")
    print(f"Diff dist exact match: {dict(Counter([t.ground_truth.attack_difficulty for t in res_b.accepted_traces])) == dict(Counter([t.ground_truth.attack_difficulty for t in res_c.accepted_traces]))}")

if __name__ == "__main__":
    main()
