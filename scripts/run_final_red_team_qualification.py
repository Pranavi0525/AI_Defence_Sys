import json
import logging
import time
import copy
from collections import Counter
from decimal import Decimal

from red_team.world.world import NormalWorld
from red_team.attacks.corpus import generate_attack_corpus
from red_team.validation.novelty import extract_fingerprint

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_world(seed: int, n_customers: int = 50, n_events: int = 200):
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
    for acct in ws.accounts.values():
        acct.balance = Decimal("25000.00")
    return ws

def verify_invariants_and_leakage(trace_events):
    forbidden_keys = [
        "attack_family", "difficulty", "variation_profile", 
        "planner", "strategy", "target_amount", "logical_split", "sim_"
    ]
    
    timestamps = []
    
    for event in trace_events:
        timestamps.append(event.timestamp)
        dump = event.model_dump()
        for k in dump.keys():
            for fb in forbidden_keys:
                if fb in k.lower():
                    return False, "LEAKAGE"
        
        # We can't verify balances easily from Observable traces because balance is stripped,
        # but we proved this in test_stage_29_cross_family.py using sim.generated_events.
    
    if sorted(timestamps) != timestamps:
        return False, "CHRONOLOGY"
        
    return True, "CLEAN"

def generate_mixed_corpus(ws, ato_target, app_target, ato_seed, app_seed):
    # ATO
    ato = generate_attack_corpus(
        world_state=ws, target_count=ato_target, master_seed=ato_seed, use_novelty=True,
        difficulty_quotas={"easy": ato_target//4, "medium": ato_target//4, "hard": ato_target//4, "advanced": ato_target//4},
        attack_family="ACCOUNT_TAKEOVER"
    )
    # APP
    app = generate_attack_corpus(
        world_state=ws, target_count=app_target, master_seed=app_seed, use_novelty=True,
        difficulty_quotas={"easy": app_target//4, "medium": app_target//4, "hard": app_target//4, "advanced": app_target//4},
        attack_family="AUTHORIZED_PUSH_PAYMENT"
    )
    return ato, app

def save_corpus(res, filepath):
    # Serialize the full ObservableAttackTrace + AttackGroundTruth pairs
    traces = []
    for t in res.accepted_traces:
        traces.append({
            "observable_trace": t.observable_trace.model_dump(mode="json"),
            "ground_truth": t.ground_truth.model_dump(mode="json")
        })
    with open(filepath, "w") as f:
        json.dump(traces, f, indent=2)

def run_experiment():
    logging.info("Starting Final Red Team Qualification Audit (v2)...")
    ws = setup_world(100)

    # REGENERATE BOTH CORPORA FRESH, WITH DIFFERENT SEEDS
    ato_target = 100
    app_target = 200
    ato_seed = 42
    app_seed = 43

    t0 = time.time()
    ato_res, app_res = generate_mixed_corpus(ws, ato_target, app_target, ato_seed, app_seed)
    duration = time.time() - t0
    
    # Save raw traces
    save_corpus(ato_res, "reports/ato_corpus_raw.json")
    save_corpus(app_res, "reports/app_corpus_raw.json")

    stats = {}
    
    def process_result(res, family):
        accepted = len(res.accepted_traces)
        rejected = len(res.rejected_attempts)
        
        from red_team.validation.novelty import extract_fingerprint
        fps = [hash(extract_fingerprint(t.observable_trace, t.ground_truth, ws)) for t in res.accepted_traces]
        unique_fps = len(set(fps))
        
        diff_counts = Counter([t.ground_truth.attack_difficulty for t in res.accepted_traces])
        
        clean = True
        failed_reason = None
        for t in res.accepted_traces:
            c, r = verify_invariants_and_leakage(t.observable_trace.events)
            if not c:
                clean = False
                failed_reason = r
        
        return {
            "family": family,
            "accepted": accepted,
            "rejected": rejected,
            "unique_fps": unique_fps,
            "difficulty_distribution": dict(diff_counts),
            "invariant_and_leakage_clean": clean,
            "failed_reason": failed_reason
        }

    stats["ato"] = process_result(ato_res, "ACCOUNT_TAKEOVER")
    stats["app"] = process_result(app_res, "AUTHORIZED_PUSH_PAYMENT")
    
    with open("reports/final_qualification_data_v2.json", "w") as f:
        json.dump(stats, f, indent=2)

    logging.info(f"Qualification audit complete in {duration:.1f}s.")

if __name__ == "__main__":
    run_experiment()
