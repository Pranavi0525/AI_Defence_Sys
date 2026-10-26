import json
import logging
import time
from collections import Counter
from datetime import datetime

from red_team.world.world import NormalWorld
from red_team.attacks.corpus import generate_attack_corpus
from red_team.validation.novelty import NoveltyIndex, APPAttackFingerprint
from red_team.schemas.observable import ObservableAttackTrace

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_world(seed: int, n_customers: int = 500, n_events: int = 1000):
    world = NormalWorld(seed=seed)
    world.generate_population(n_customers=n_customers)
    world.generate_legitimate_events(num_events=n_events)
    ws = world.get_state()
    # Inject devices so APP can run smoothly
    from red_team.schemas.entities import Device, Relationship
    for c_id in ws.customers:
        dev_id = f"dev_{c_id[:5]}"
        ws.devices[dev_id] = Device(
            device_id=dev_id, device_type="mobile", fingerprint="fp1",
            first_seen=ws.current_time, last_seen=ws.current_time, is_trusted=True
        )
        rel_id = f"rel_{c_id[:5]}"
        ws.relationships[rel_id] = Relationship(
            relationship_id=rel_id, source_entity_type="customer", source_entity_id=c_id,
            target_entity_type="device", target_entity_id=dev_id, relationship_type="owns",
            established_date=ws.current_time
        )
    return ws

def run_qualification():
    logging.info("Starting Stage 26 Qualification...")
    ws = setup_world(42)

    # 3. GENERATE A LARGE RAW APP POOL
    logging.info("Generating 2000 RAW APP traces (novelty disabled)...")
    start_time = time.time()
    raw_result = generate_attack_corpus(
        world_state=ws.model_copy(deep=True), target_count=2000, master_seed=42, use_novelty=False,
        difficulty_quotas={"easy": 500, "medium": 500, "hard": 500, "advanced": 500},
        attack_family="AUTHORIZED_PUSH_PAYMENT"
    )
    raw_duration = time.time() - start_time
    
    # Analyze raw pool
    raw_accepted = len(raw_result.accepted_traces)
    raw_rejected = len(raw_result.rejected_attempts)
    
    fps = [t.novelty_fingerprint_id for t in raw_result.accepted_traces]
    unique_fps = len(set(fps))
    
    # 6. NOVELTY SATURATION
    logging.info("Generating 100 novel traces per difficulty...")
    sat_start = time.time()
    sat_result = generate_attack_corpus(
        world_state=ws.model_copy(deep=True), target_count=400, master_seed=42, use_novelty=True,
        difficulty_quotas={"easy": 100, "medium": 100, "hard": 100, "advanced": 100},
        attack_family="AUTHORIZED_PUSH_PAYMENT"
    )
    sat_duration = time.time() - sat_start

    # 14. REPRODUCIBILITY
    logging.info("Running Reproducibility Check (Run 2)...")
    sat_result_2 = generate_attack_corpus(
        world_state=ws.model_copy(deep=True), target_count=400, master_seed=42, use_novelty=True,
        difficulty_quotas={"easy": 100, "medium": 100, "hard": 100, "advanced": 100},
        attack_family="AUTHORIZED_PUSH_PAYMENT"
    )
    
    match_count = sum(1 for a, b in zip(sat_result.accepted_traces, sat_result_2.accepted_traces)
                      if [e.event_type for e in a.observable_trace.events] == [e.event_type for e in b.observable_trace.events])
    
    # Build Stats
    stats = {
        "raw_pool": {
            "target": 2000,
            "accepted": raw_accepted,
            "rejected": raw_rejected,
            "unique_fingerprints": unique_fps,
            "runtime_sec": round(raw_duration, 2)
        },
        "novelty_saturation": {
            "target": 400,
            "accepted": len(sat_result.accepted_traces),
            "rejected": len(sat_result.rejected_attempts),
            "runtime_sec": round(sat_duration, 2)
        },
        "reproducibility": {
            "exact_sequence_matches": match_count,
            "total_accepted": len(sat_result.accepted_traces)
        }
    }
    
    # Detailed difficulty analysis from raw
    diff_stats = {"easy": [], "medium": [], "hard": [], "advanced": []}
    for t in raw_result.accepted_traces:
        diff_stats[t.ground_truth.attack_difficulty].append(t)
        
    stats["difficulty_semantics"] = {}
    for diff, traces in diff_stats.items():
        if not traces:
            continue
        avg_events = sum(len(t.observable_trace.events) for t in traces) / len(traces)
        avg_txs = sum(sum(1 for e in t.observable_trace.events if e.event_type == "TRANSACTION") for t in traces) / len(traces)
        unique_fps_diff = len(set(t.novelty_fingerprint_id for t in traces))
        stats["difficulty_semantics"][diff] = {
            "count": len(traces),
            "avg_events": avg_events,
            "avg_transactions": avg_txs,
            "unique_fingerprints": unique_fps_diff
        }
        
    sat_diff_stats = {"easy": 0, "medium": 0, "hard": 0, "advanced": 0}
    for t in sat_result.accepted_traces:
        sat_diff_stats[t.ground_truth.attack_difficulty] += 1
        
    stats["novelty_saturation"]["difficulty_counts"] = sat_diff_stats
    
    # Determine reasons
    reasons = Counter()
    for rej in sat_result.rejected_attempts:
        reasons[rej.get("rejection_reason", "UNKNOWN")] += 1
    stats["novelty_saturation"]["reasons"] = dict(reasons)

    with open("reports/stage_26_qualification_data.json", "w") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    run_qualification()
