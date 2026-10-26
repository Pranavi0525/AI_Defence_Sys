import json
import logging
import time
from collections import Counter
from decimal import Decimal

from red_team.world.world import NormalWorld
from red_team.attacks.corpus import generate_attack_corpus
from red_team.validation.novelty import calculate_fingerprint_similarity, APPAttackFingerprint, AttackFingerprint

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
    # Ensure they have money
    for acct in ws.accounts.values():
        acct.balance = Decimal("10000.00")
    return ws

def verify_ground_truth_isolation(trace_events):
    forbidden_keys = [
        "attack_family", "difficulty", "variation_profile", 
        "planner", "strategy", "target_amount", "logical_split", "sim_"
    ]
    for event in trace_events:
        dump = event.model_dump()
        for k in dump.keys():
            for fb in forbidden_keys:
                if fb in k.lower():
                    return False
    return True

def run_experiment():
    logging.info("Starting Stage 29 Cross-Family Audit...")
    ws = setup_world(42)

    stats = {}
    
    # Generate ATO
    logging.info("Generating ATO Corpus...")
    ato_res = generate_attack_corpus(
        world_state=ws, target_count=400, master_seed=42, use_novelty=True,
        difficulty_quotas={"easy": 100, "medium": 100, "hard": 100, "advanced": 100},
        attack_family="ACCOUNT_TAKEOVER"
    )
    
    # Generate APP
    logging.info("Generating APP Corpus...")
    app_res = generate_attack_corpus(
        world_state=ws, target_count=400, master_seed=42, use_novelty=True,
        difficulty_quotas={"easy": 100, "medium": 100, "hard": 100, "advanced": 100},
        attack_family="AUTHORIZED_PUSH_PAYMENT"
    )
    
    def process_result(res, family):
        accepted = len(res.accepted_traces)
        rejected = len(res.rejected_attempts)
        from red_team.validation.novelty import extract_fingerprint
        fps = [hash(extract_fingerprint(t.observable_trace, t.ground_truth, ws)) for t in res.accepted_traces]
        unique_fps = len(set(fps))
        
        diff_counts = Counter([t.ground_truth.attack_difficulty for t in res.accepted_traces])
        isolation_clean = all(verify_ground_truth_isolation(t.observable_trace.events) for t in res.accepted_traces)
        
        # event counts
        evt_counts = [len(t.observable_trace.events) for t in res.accepted_traces]
        tx_counts = [sum(1 for e in t.observable_trace.events if e.event_type == "TRANSACTION") for t in res.accepted_traces]
        
        return {
            "family": family,
            "target": 400,
            "accepted": accepted,
            "rejected": rejected,
            "unique_fps": unique_fps,
            "difficulty_distribution": dict(diff_counts),
            "isolation_clean": isolation_clean,
            "avg_events": sum(evt_counts)/len(evt_counts) if evt_counts else 0,
            "avg_txs": sum(tx_counts)/len(tx_counts) if tx_counts else 0
        }

    stats["ato"] = process_result(ato_res, "ACCOUNT_TAKEOVER")
    stats["app"] = process_result(app_res, "AUTHORIZED_PUSH_PAYMENT")
    
    # Cross-family novelty isolation test
    app_fp = APPAttackFingerprint(
        attack_family="AUTHORIZED_PUSH_PAYMENT", phase_sequence=("SOCIAL_ENGINEERING",),
        event_sequence=("SESSION_LOGIN",), transaction_count=1, amount_buckets=("small",),
        normalized_amount_sum=0.1, split_count=1, timing_category="rapid", device_continuity=True,
        session_continuity=True, beneficiary_novelty="new", outcome_pattern=("completed",),
        hesitation_category="immediate", amount_trend="single"
    )
    ato_fp = AttackFingerprint(
        attack_family="ACCOUNT_TAKEOVER", phase_sequence=("SOCIAL_ENGINEERING",),
        event_sequence=("SESSION_LOGIN",), transaction_count=1, amount_buckets=("small",),
        normalized_amount_sum=0.1, split_count=1, timing_category="rapid", device_continuity=True,
        session_continuity=True, beneficiary_novelty="new"
    )
    
    sim_score = calculate_fingerprint_similarity(app_fp, ato_fp)
    stats["novelty_isolation"] = {
        "identical_behavior_cross_family_similarity": sim_score
    }
    
    with open("reports/stage_29_audit_data.json", "w") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    run_experiment()
