import os
import sys
import copy
from collections import defaultdict, Counter
import statistics
import json

from red_team.world.world import NormalWorld
from red_team.attacks.corpus import generate_attack_corpus
from red_team.attacks.simulator import DIFFICULTY_PROFILES
from red_team.schemas.events import EventType

def main():
    world = NormalWorld(seed=42)
    world.generate_population(n_customers=10)
    base_state = world.get_state()
    
    # 1. Generate Controlled Sample (master_seed=42)
    state_copy = copy.deepcopy(base_state)
    result = generate_attack_corpus(
        state_copy, 
        target_count=100, 
        master_seed=42, 
        max_attempts=1000
    )
    
    accepted = result.accepted_traces
    print(f"Total accepted: {len(accepted)}")
    
    # 2. Amount Variation
    tx_amounts = []
    tx_completed = 0
    tx_failed = 0
    for r in accepted:
        for e in r.observable_trace.events:
            if e.event_type == "TRANSACTION":
                amt = float(getattr(e, "amount", 0))
                tx_amounts.append(amt)
                status = getattr(e, "transaction_status", "none")
                if status == "completed":
                    tx_completed += 1
                elif status == "failed":
                    tx_failed += 1
                
    if tx_amounts:
        print("\n--- AMOUNT VARIATION ---")
        print(f"Unique amounts: {len(set(tx_amounts))}")
        print(f"Total transactions: {len(tx_amounts)}")
        print(f"Variance: {statistics.variance(tx_amounts) if len(tx_amounts) > 1 else 0}")
        print(f"Min: {min(tx_amounts)}, Max: {max(tx_amounts)}")
        print(f"Mean: {statistics.mean(tx_amounts)}, Median: {statistics.median(tx_amounts)}")
        
    print(f"\n--- OUTCOMES ---")
    print(f"Completed: {tx_completed}")
    print(f"Failed: {tx_failed}")
    
    # 3. Difficulty Audit
    diff_stats = defaultdict(lambda: {"phases": [], "events": [], "tx_count": 0})
    for r in accepted:
        d = r.ground_truth.attack_difficulty
        ds = diff_stats[d]
        ds["phases"].append(len(r.ground_truth.phases_executed))
        ds["events"].append(len(r.observable_trace.events))
        txs = sum(1 for e in r.observable_trace.events if e.event_type == "TRANSACTION")
        ds["tx_count"] += txs

    print("\n--- DIFFICULTY AUDIT ---")
    for d, s in diff_stats.items():
        print(f"Difficulty: {d.upper()}")
        print(f"  Traces: {len(s['phases'])}")
        print(f"  Mean phases: {statistics.mean(s['phases'])}")
        print(f"  Mean events: {statistics.mean(s['events'])}")
        print(f"  Total txs: {s['tx_count']}")
        
    # 4. Path Diversity
    paths = [tuple(p.phase for p in r.ground_truth.phases_executed) for r in accepted]
    path_counts = Counter(paths)
    print("\n--- PATH DIVERSITY ---")
    print(f"Unique paths: {len(path_counts)}")
    print(f"Most common: {path_counts.most_common(1)[0]}")
    
    # 5. Timing Audit
    all_gaps = []
    trace_durs = []
    for r in accepted:
        times = [e.timestamp for e in r.observable_trace.events]
        if len(times) > 1:
            dur = (times[-1] - times[0]).total_seconds()
            trace_durs.append(dur)
            for i in range(1, len(times)):
                all_gaps.append((times[i] - times[i-1]).total_seconds())
                
    if all_gaps:
        print("\n--- TIMING AUDIT ---")
        print(f"Trace durations: min={min(trace_durs)}, max={max(trace_durs)}, mean={statistics.mean(trace_durs)}")
        print(f"Event gaps: min={min(all_gaps)}, max={max(all_gaps)}, mean={statistics.mean(all_gaps)}, median={statistics.median(all_gaps)}")
        
    # 6. Duplicates
    from run_stage_17_duplicates import calculate_similarity
    sim_threshold = 0.8
    near_dup = 0
    total_pairs = (len(accepted) * (len(accepted) - 1)) / 2
    for i in range(len(accepted)):
        for j in range(i + 1, len(accepted)):
            sim = calculate_similarity(accepted[i], accepted[j])
            if sim >= sim_threshold:
                near_dup += 1
    print("\n--- DUPLICATE AUDIT ---")
    print(f"Near duplicate pairs (>=0.8): {near_dup} ({near_dup / total_pairs:.2%} of pairs)")

if __name__ == "__main__":
    main()
