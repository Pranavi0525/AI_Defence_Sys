import pytest
from red_team.world.world import NormalWorld
from red_team.attacks.app_signature import get_app_signature
from red_team.attacks.ato_signature import get_ato_signature
from red_team.attacks.simulator import StatefulSimulator, AttackPlan
from red_team.validation.novelty import calculate_fingerprint_similarity, extract_fingerprint, AttackFingerprint, APPAttackFingerprint
from red_team.validation.realism import validate_attack_realism
from red_team.schemas.entities import Device, Relationship

def setup_world():
    world = NormalWorld(seed=42)
    world.generate_population(n_customers=5)
    world.generate_legitimate_events(num_events=10)
    state = world.get_state()
    c_id = list(state.customers.keys())[0]
    dev_id = f"dev_{c_id[:5]}"
    state.devices[dev_id] = Device(
        device_id=dev_id, device_type="mobile", fingerprint="fp1",
        first_seen=state.current_time, last_seen=state.current_time, is_trusted=True
    )
    rel_id = f"rel_{c_id[:5]}"
    state.relationships[rel_id] = Relationship(
        relationship_id=rel_id, source_entity_type="customer", source_entity_id=c_id,
        target_entity_type="device", target_entity_id=dev_id, relationship_type="owns",
        established_date=state.current_time
    )
    return state, c_id

def test_cross_family_generation():
    state, c_id = setup_world()
    
    # 1. ATO
    import copy
    ato_state = copy.deepcopy(state)
    ato_sim = StatefulSimulator(ato_state, get_ato_signature(), seed=42)
    ato_plan = AttackPlan(attack_family="ACCOUNT_TAKEOVER", difficulty="hard", entry_state="ACCOUNT_ACCESS", max_phases=4)
    ato_trace, ato_gt = ato_sim.generate_attack(ato_plan, c_id)
    
    # 2. APP
    app_state = copy.deepcopy(state)
    app_sim = StatefulSimulator(app_state, get_app_signature(), seed=42)
    app_plan = AttackPlan(attack_family="AUTHORIZED_PUSH_PAYMENT", difficulty="hard", entry_state="SOCIAL_ENGINEERING", max_phases=4)
    app_trace, app_gt = app_sim.generate_attack(app_plan, c_id)
    
    # Realism Isolation
    assert validate_attack_realism(ato_trace, ato_gt, get_ato_signature(), state).structural.passed
    assert validate_attack_realism(app_trace, app_gt, get_app_signature(), state).structural.passed

    # Observable Isolation
    assert not any("attack_family" in k for e in ato_trace.events for k in e.model_dump().keys())
    assert not any("attack_family" in k for e in app_trace.events for k in e.model_dump().keys())
    
    # Novelty Isolation
    ato_fp = extract_fingerprint(ato_trace, ato_gt, ato_state)
    app_fp = extract_fingerprint(app_trace, app_gt, app_state)
    
    assert ato_fp.attack_family == "ACCOUNT_TAKEOVER"
    assert app_fp.attack_family == "AUTHORIZED_PUSH_PAYMENT"
    
    # Same base structure but different families MUST have 0 similarity
    assert calculate_fingerprint_similarity(ato_fp, app_fp) == 0.0

def test_balance_and_chronology_invariants():
    state, c_id = setup_world()
    app_sim = StatefulSimulator(state, get_app_signature(), seed=10)
    plan = AttackPlan(attack_family="AUTHORIZED_PUSH_PAYMENT", difficulty="hard", entry_state="SOCIAL_ENGINEERING")
    trace, gt = app_sim.generate_attack(plan, c_id)
    
    # Chronology
    timestamps = [e.timestamp for e in trace.events]
    assert sorted(timestamps) == timestamps, "Chronology failed!"
    
    # Balances
    for e in app_sim.generated_events:
        if e.envelope.event_type == "TRANSACTION":
            pre = e.payload.pre_balance
            post = e.payload.post_balance
            amt = e.payload.transaction.amount
            status = e.payload.transaction.status
            if status == "completed":
                assert pre - amt == post
            else:
                assert pre == post
