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

def setup_world(seed: int, n_customers: int = 20, n_events: int = 50):
    world = NormalWorld(seed=seed)
    world.generate_population(n_customers=n_customers)
    world.generate_legitimate_events(num_events=n_events)
    ws = world.get_state()
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

def run_experiment():
    logging.info("Starting Stage 27 Entropy Analysis...")
    ws = setup_world(42)

    diffs = ["easy", "medium", "hard", "advanced"]
    stats = {}

    for diff in diffs:
        logging.info(f"Generating 2000 RAW APP traces for {diff.upper()}...")
        start_time = time.time()
        
        # We run 2000 per difficulty
        result = generate_attack_corpus(
            world_state=ws, target_count=2000, master_seed=42, use_novelty=False,
            difficulty_quotas={diff: 2000},
            attack_family="AUTHORIZED_PUSH_PAYMENT"
        )
        
        duration = time.time() - start_time
        fps = [t.novelty_fingerprint_id for t in result.accepted_traces]
        unique_fps = len(set(fps))
        fp_counts = Counter(fps)
        exact_dups = sum(count - 1 for count in fp_counts.values())
        
        events_counts = [len(t.observable_trace.events) for t in result.accepted_traces]
        tx_counts = [sum(1 for e in t.observable_trace.events if e.event_type == "TRANSACTION") for t in result.accepted_traces]
        
        stats[diff] = {
            "accepted": len(result.accepted_traces),
            "unique_fingerprints": unique_fps,
            "exact_duplicates": exact_dups,
            "avg_events": sum(events_counts)/len(events_counts) if events_counts else 0,
            "avg_txs": sum(tx_counts)/len(tx_counts) if tx_counts else 0,
            "runtime_sec": round(duration, 2)
        }
        logging.info(f"{diff.upper()}: {unique_fps} unique FPs out of {len(result.accepted_traces)} accepted. Took {duration:.2f}s")

    with open("reports/stage_27_entropy_data.json", "w") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    run_experiment()
