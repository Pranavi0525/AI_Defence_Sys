import os
import sys
import logging
from collections import Counter, defaultdict
import json
import statistics

from red_team.world.world import NormalWorld
from red_team.attacks.corpus import generate_attack_corpus

logging.basicConfig(level=logging.ERROR)

def main():
    world = NormalWorld(seed=42)
    world.generate_population(n_customers=10)
    
    result = generate_attack_corpus(
        world.get_state(), 
        target_count=100, 
        master_seed=42, 
        max_attempts=1000
    )
    
    stats = result.generation_statistics
    
    print(f"Total Attempts: {stats.attempted}")
    print(f"Accepted: {stats.accepted}")
    print(f"Rejected: {stats.rejected}")
    print(f"Acceptance Rate: {stats.acceptance_rate:.2%}")
    
    customers = [rec.observable_trace.customer_id for rec in result.accepted_traces]
    customer_counts = Counter(customers)
    print(f"\nUnique customers attacked: {len(customer_counts)}")
    print(f"Top customer %: {customer_counts.most_common(1)[0][1] / len(customers):.2%}")
    top_5 = sum(v for k, v in customer_counts.most_common(5))
    top_10 = sum(v for k, v in customer_counts.most_common(10))
    print(f"Top 5 customers %: {top_5 / len(customers):.2%}")
    print(f"Top 10 customers %: {top_10 / len(customers):.2%}")
    
    phases_per_trace = []
    events_per_trace = []
    paths = []
    diff_stats = defaultdict(lambda: {"count": 0, "phases": [], "events": [], "paths": set(), "tx_total": 0, "tx_declined": 0, "tx_approved": 0, "dev_sess": 0, "bene": 0})
    
    all_events = Counter()
    all_tx_amounts = []
    all_tx_types = Counter()
    all_gaps = []
    trace_durations = []
    
    for rec in result.accepted_traces:
        trace = rec.observable_trace
        gt = rec.ground_truth
        
        # Difficulty
        diff = gt.attack_difficulty
        ds = diff_stats[diff]
        ds["count"] += 1
        
        # Paths & Phases
        phase_seq = tuple(p.phase for p in gt.phases_executed)
        paths.append(phase_seq)
        ds["paths"].add(phase_seq)
        
        num_phases = len(phase_seq)
        phases_per_trace.append(num_phases)
        ds["phases"].append(num_phases)
        
        # Events
        num_events = len(trace.events)
        events_per_trace.append(num_events)
        ds["events"].append(num_events)
        
        timestamps = []
        for e in trace.events:
            all_events[e.event_type] += 1
            timestamps.append(e.timestamp)
            
            if e.event_type == "TRANSACTION":
                ds["tx_total"] += 1
                all_tx_types[getattr(e, "transaction_type", "unknown")] += 1
                amt = getattr(e, "amount", 0)
                all_tx_amounts.append(float(amt))
                if getattr(e, "transaction_status", "") == "failed":
                    ds["tx_declined"] += 1
                else:
                    ds["tx_approved"] += 1
            elif e.event_type in ("SESSION_LOGIN", "DEVICE_REGISTRATION"):
                ds["dev_sess"] += 1
            elif e.event_type == "BENEFICIARY_ADDITION":
                ds["bene"] += 1
                
        # Timing
        if len(timestamps) > 1:
            dur = (timestamps[-1] - timestamps[0]).total_seconds()
            trace_durations.append(dur)
            for i in range(1, len(timestamps)):
                gap = (timestamps[i] - timestamps[i-1]).total_seconds()
                all_gaps.append(gap)
                
    path_counts = Counter(paths)
    print("\n--- ATTACK PATH DIVERSITY ---")
    print(f"Unique paths: {len(path_counts)}")
    print(f"Most common path: {path_counts.most_common(1)[0][0]} ({path_counts.most_common(1)[0][1]} times)")
    print(f"Phases per trace: min={min(phases_per_trace)}, max={max(phases_per_trace)}, mean={statistics.mean(phases_per_trace):.2f}, median={statistics.median(phases_per_trace)}")
    
    print("\n--- EVENT DIVERSITY ---")
    print(f"Events per trace: min={min(events_per_trace)}, max={max(events_per_trace)}, mean={statistics.mean(events_per_trace):.2f}, median={statistics.median(events_per_trace)}")
    print(f"Event types: {dict(all_events)}")
    print(f"Tx types: {dict(all_tx_types)}")
    print(f"Tx amounts: min={min(all_tx_amounts)}, max={max(all_tx_amounts)}, mean={statistics.mean(all_tx_amounts):.2f}, median={statistics.median(all_tx_amounts)}")
    
    print("\n--- TIMING DIVERSITY ---")
    print(f"Trace durations (s): min={min(trace_durations)}, max={max(trace_durations)}, mean={statistics.mean(trace_durations):.2f}, median={statistics.median(trace_durations)}")
    print(f"Event gaps (s): min={min(all_gaps)}, max={max(all_gaps)}, mean={statistics.mean(all_gaps):.2f}, median={statistics.median(all_gaps)}")
    
    print("\n--- DIFFICULTY VALIDITY ---")
    for diff in ["easy", "medium", "hard", "advanced"]:
        if diff in diff_stats:
            ds = diff_stats[diff]
            print(f"{diff.upper()}: {ds['count']} traces")
            print(f"  Unique paths: {len(ds['paths'])}")
            print(f"  Mean phases: {statistics.mean(ds['phases']):.2f}")
            print(f"  Mean events: {statistics.mean(ds['events']):.2f}")
            print(f"  Tx total: {ds['tx_total']} (Appr: {ds['tx_approved']}, Decl: {ds['tx_declined']})")
            print(f"  Dev/Sess: {ds['dev_sess']}, Bene: {ds['bene']}")
    
if __name__ == "__main__":
    main()
