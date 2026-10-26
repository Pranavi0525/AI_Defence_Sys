import json
import logging
from collections import Counter
from datetime import datetime
from decimal import Decimal
from copy import deepcopy

from red_team.world.world import NormalWorld
from red_team.attacks.corpus import generate_attack_corpus
from red_team.validation.novelty import NoveltyIndex, ATOAttackFingerprint, APPAttackFingerprint
from red_team.schemas.observable import ObservableAttackTrace

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def perform_audit():
    logging.info("Starting Stage 25 Verification Audit...")

    # Setup world
    world = NormalWorld(seed=42)
    world.generate_population(n_customers=500)
    world.generate_legitimate_events(num_events=1000)
    world_state = world.get_state()

    # Pre-populate devices so APP isn't blocked by missing legitimate devices
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

    # 1. GENERATE APP AND ATO
    logging.info("Generating ATO traces...")
    ato_result = generate_attack_corpus(
        world_state=world_state, target_count=20, master_seed=42, use_novelty=True,
        difficulty_quotas={"easy": 5, "medium": 5, "hard": 5, "advanced": 5},
        attack_family="ACCOUNT_TAKEOVER"
    )

    logging.info("Generating APP traces (Run 1)...")
    app_result_1 = generate_attack_corpus(
        world_state=world_state, target_count=20, master_seed=42, use_novelty=True,
        difficulty_quotas={"easy": 5, "medium": 5, "hard": 5, "advanced": 5},
        attack_family="AUTHORIZED_PUSH_PAYMENT"
    )
    
    logging.info("Generating APP traces (Run 2 - Determinism Check)...")
    app_result_2 = generate_attack_corpus(
        world_state=world_state, target_count=20, master_seed=42, use_novelty=True,
        difficulty_quotas={"easy": 5, "medium": 5, "hard": 5, "advanced": 5},
        attack_family="AUTHORIZED_PUSH_PAYMENT"
    )

    stats = {}

    # --------------------------------------------------
    # DETERMINISM
    # --------------------------------------------------
    # Compare dumps (minus UUIDs or simply checking lengths/amounts)
    match_count = sum(1 for a, b in zip(app_result_1.accepted_traces, app_result_2.accepted_traces)
                      if [e.event_type for e in a.observable_trace.events] == [e.event_type for e in b.observable_trace.events])
    stats["determinism"] = {
        "run1_accepted": len(app_result_1.accepted_traces),
        "run2_accepted": len(app_result_2.accepted_traces),
        "exact_sequence_matches": match_count
    }

    # --------------------------------------------------
    # REJECTION AUDIT
    # --------------------------------------------------
    def classify_rejections(result):
        reasons = Counter()
        for rej in result.rejected_attempts:
            reason = rej.get("rejection_reason", "UNKNOWN")
            if reason == "NOVELTY_REJECTION":
                reasons["novelty"] += 1
            elif "REALISM" in reason or reason == "REALISM_REJECTION":
                reasons["realism"] += 1
            else:
                reasons[reason] += 1
        return dict(reasons)

    stats["rejections"] = {
        "app_total": len(app_result_1.rejected_attempts),
        "app_reasons": classify_rejections(app_result_1),
        "ato_total": len(ato_result.rejected_attempts),
        "ato_reasons": classify_rejections(ato_result)
    }

    # --------------------------------------------------
    # SEMANTIC, ISOLATION, ORDERING, CONTINUITY, OUTCOME, CHRONOLOGY
    # --------------------------------------------------
    isolation_leaks = 0
    bad_chronology = 0
    bad_balances = 0
    bad_devices = 0
    bad_beneficiaries = 0
    
    gt_keywords = ["attack_family", "difficulty", "variation_profile", "objective", "hidden", "ground_truth"]
    
    difficulty_stats = {"easy": [], "medium": [], "hard": [], "advanced": []}
    fingerprints = set()
    
    for trace in app_result_1.accepted_traces:
        # Isolation
        dump_str = json.dumps(trace.observable_trace.model_dump(), default=str).lower()
        if any(kw in dump_str for kw in gt_keywords):
            isolation_leaks += 1

        # Chronology
        ts = [e.timestamp for e in trace.observable_trace.events]
        if ts != sorted(ts):
            bad_chronology += 1
            
        # Device Continuity & Beneficiary & Balances
        known_devices = []
        bens_added = set()
        
        events = trace.observable_trace.events
        for i, ev in enumerate(events):
            if ev.event_type == "DEVICE_REGISTRATION":
                bad_devices += 1
            if ev.event_type == "SESSION_LOGIN":
                known_devices.append(getattr(ev, "device_id", None))
            if ev.event_type == "BENEFICIARY_ADDITION":
                bens_added.add(getattr(ev, "beneficiary_id", None))
            if ev.event_type == "TRANSACTION":
                ben_id = getattr(ev, "beneficiary_id", None)
                if ben_id and ben_id not in world_state.beneficiaries and ben_id not in bens_added:
                    bad_beneficiaries += 1
                
                # Balances
                if getattr(ev, "status", None) == "completed":
                    if getattr(ev, "pre_balance", Decimal('0')) - getattr(ev, "amount", Decimal('0')) != getattr(ev, "post_balance", Decimal('0')):
                        bad_balances += 1
                else:
                    if getattr(ev, "pre_balance", Decimal('0')) != getattr(ev, "post_balance", Decimal('0')):
                        bad_balances += 1

        difficulty_stats[trace.ground_truth.difficulty].append({
            "event_count": len(events),
            "payment_count": sum(1 for e in events if e.event_type == "TRANSACTION")
        })
        
        # Diversity
        fingerprints.add(trace.novelty_fingerprint_id)

    stats["audit_checks"] = {
        "isolation_leaks": isolation_leaks,
        "bad_chronology": bad_chronology,
        "bad_devices": bad_devices,
        "bad_beneficiaries": bad_beneficiaries,
        "bad_balances": bad_balances
    }
    
    stats["difficulty"] = {d: {"avg_events": sum(x["event_count"] for x in l)/len(l) if l else 0,
                               "avg_payments": sum(x["payment_count"] for x in l)/len(l) if l else 0} 
                           for d, l in difficulty_stats.items()}
                           
    stats["diversity"] = {
        "unique_fingerprints": len(fingerprints)
    }
    
    with open("reports/stage_25_audit_data.json", "w") as f:
        json.dump(stats, f, indent=2)

if __name__ == "__main__":
    perform_audit()
