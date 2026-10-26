import os
import sys
import logging
import itertools
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple

from red_team.world.world import NormalWorld
from red_team.attacks.corpus import generate_attack_corpus

@dataclass
class StructuredPlan:
    attack_family: str
    entry_path: str
    phases: List[str]
    phase_order: List[str]
    skipped_phases: List[str]
    loops: int
    path_length: int
    variation_axes: dict
    signal_families: List[str]
    affected_entity_categories: List[str]

def extract_structured_plan(rec) -> StructuredPlan:
    gt = rec.ground_truth
    phases = [p.phase for p in gt.phases_executed]
    
    # Estimate loops
    loops = len(phases) - len(set(phases))
    
    return StructuredPlan(
        attack_family=gt.attack_family,
        entry_path=phases[0] if phases else "unknown",
        phases=list(set(phases)),
        phase_order=phases,
        skipped_phases=[], # Not strictly tracked in gt, but could deduce
        loops=loops,
        path_length=len(phases),
        variation_axes={"difficulty": gt.attack_difficulty},
        signal_families=[],
        affected_entity_categories=list(set([e.event_type for e in rec.observable_trace.events]))
    )

def calculate_similarity(rec1, rec2):
    t1 = rec1.observable_trace
    t2 = rec2.observable_trace
    
    gt1 = rec1.ground_truth
    gt2 = rec2.ground_truth
    
    p1 = [p.phase for p in gt1.phases_executed]
    p2 = [p.phase for p in gt2.phases_executed]
    phase_seq_sim = 1.0 if p1 == p2 else 0.0
    phase_set_sim = len(set(p1) & set(p2)) / max(1, len(set(p1) | set(p2)))
    
    e1 = [e.event_type for e in t1.events]
    e2 = [e.event_type for e in t2.events]
    event_seq_sim = 1.0 if e1 == e2 else 0.0
    
    entry_sim = 1.0 if p1[0] == p2[0] else 0.0
    
    o1 = [getattr(e, "transaction_status", "none") for e in t1.events if e.event_type == "TRANSACTION"]
    o2 = [getattr(e, "transaction_status", "none") for e in t2.events if e.event_type == "TRANSACTION"]
    outcome_sim = 1.0 if o1 == o2 else 0.0
    
    return (phase_seq_sim + phase_set_sim + event_seq_sim + entry_sim + outcome_sim) / 5.0

def main():
    world = NormalWorld(seed=42)
    world.generate_population(n_customers=10)
    
    result = generate_attack_corpus(
        world.get_state(), 
        target_count=100, 
        master_seed=42, 
        max_attempts=1000
    )
    
    traces = result.accepted_traces
    
    # Near duplicates
    near_duplicates = 0
    threshold = 0.8
    for i in range(len(traces)):
        for j in range(i + 1, len(traces)):
            sim = calculate_similarity(traces[i], traces[j])
            if sim >= threshold:
                near_duplicates += 1
                
    total_pairs = (len(traces) * (len(traces) - 1)) / 2
    print(f"Total pairs: {total_pairs}")
    print(f"Near duplicates (sim >= {threshold}): {near_duplicates} ({near_duplicates/total_pairs:.2%})")
    
    # Exact duplicate paths
    paths = [tuple(p.phase for p in rec.ground_truth.phases_executed) for rec in traces]
    path_counts = Counter(paths)
    exact_duplicates = sum(v - 1 for v in path_counts.values())
    print(f"Exact duplicate paths: {exact_duplicates}")
    
    # Structured Plan
    p1 = extract_structured_plan(traces[0])
    print(f"Sample structured plan: {p1}")
    
if __name__ == "__main__":
    main()
