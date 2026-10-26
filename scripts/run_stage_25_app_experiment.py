import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from pprint import pprint

from red_team.world.state import WorldState
from red_team.attacks.corpus import generate_attack_corpus

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_experiment():
    logging.info("Starting Stage 25 APP Experiment...")
    
    # Setup world
    from red_team.world.world import NormalWorld
    world = NormalWorld(seed=42)
    world.generate_population(n_customers=500)
    world.generate_legitimate_events(num_events=1000)
    world_state = world.get_state()
    
    # Because normal generation doesn't eagerly generate devices/relationships for all,
    # let's inject a device for every customer so APP can proceed flawlessly
    from red_team.schemas.entities import Device, Relationship
    for c_id in world_state.customers:
        dev_id = f"dev_{c_id[:5]}"
        world_state.devices[dev_id] = Device(
            device_id=dev_id, device_type="mobile", fingerprint="fp1",
            first_seen=world_state.current_time, last_seen=world_state.current_time, is_trusted=True
        )
        rel_id = f"rel_{c_id[:5]}"
        world_state.relationships[rel_id] = Relationship(
            relationship_id=rel_id, source_entity_type="customer", source_entity_id=c_id,
            target_entity_type="device", target_entity_id=dev_id, relationship_type="owns",
            established_date=world_state.current_time
        )
    
    logging.info("Generating ATO traces for comparison...")
    ato_result = generate_attack_corpus(
        world_state=world_state,
        target_count=20,
        master_seed=42,
        use_novelty=True,
        novelty_threshold=0.8,
        difficulty_quotas={"easy": 5, "medium": 5, "hard": 5, "advanced": 5},
        attack_family="ACCOUNT_TAKEOVER"
    )
    logging.info(f"ATO Generated: {len(ato_result.accepted_traces)}")
    
    logging.info("Generating APP traces...")
    app_result = generate_attack_corpus(
        world_state=world_state,
        target_count=20,
        master_seed=42,
        use_novelty=True,
        novelty_threshold=0.8,
        difficulty_quotas={"easy": 5, "medium": 5, "hard": 5, "advanced": 5},
        attack_family="AUTHORIZED_PUSH_PAYMENT"
    )
    logging.info(f"APP Generated: {len(app_result.accepted_traces)}")
    
    logging.info("Analyzing traces...")
    ato_events = Counter(e.event_type for t in ato_result.accepted_traces for e in t.observable_trace.events)
    app_events = Counter(e.event_type for t in app_result.accepted_traces for e in t.observable_trace.events)
    
    stats = {
        "ato": {
            "accepted": len(ato_result.accepted_traces),
            "rejections": len(ato_result.rejected_attempts),
            "events": dict(ato_events)
        },
        "app": {
            "accepted": len(app_result.accepted_traces),
            "rejections": len(app_result.rejected_attempts),
            "events": dict(app_events)
        }
    }
    
    pprint(stats)
    
    with open("reports/stage_25_experiment.json", "w") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    run_experiment()
