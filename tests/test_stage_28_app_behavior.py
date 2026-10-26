import pytest
from red_team.world.world import NormalWorld
from red_team.attacks.app_signature import get_app_signature
from red_team.attacks.simulator import StatefulSimulator, AttackPlan
from red_team.validation.novelty import extract_fingerprint
from red_team.validation.realism import validate_attack_realism
from red_team.schemas.entities import Device, Relationship

def test_app_behavioral_generation():
    world = NormalWorld(seed=42)
    world.generate_population(n_customers=5)
    world.generate_legitimate_events(num_events=10)
    state = world.get_state()
    
    # Setup device
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

    signature = get_app_signature()
    
    # HARD difficulty enables retries and hesitation
    plan = AttackPlan(
        attack_family="AUTHORIZED_PUSH_PAYMENT",
        difficulty="hard",
        entry_state="SOCIAL_ENGINEERING",
        max_phases=6
    )
    
    import copy
    state_copy = copy.deepcopy(state)
    sim = StatefulSimulator(state_copy, signature, seed=42)
    trace, gt = sim.generate_attack(plan, c_id)
    
    # Realism check
    report = validate_attack_realism(trace, gt, signature, state)
    assert report.structural.passed, "Realism failed!"
    
    # Extract fingerprint
    fp = extract_fingerprint(trace, gt, state_copy)
    assert fp.attack_family == "AUTHORIZED_PUSH_PAYMENT"
    
    # Verify new fields exist
    assert hasattr(fp, "hesitation_category")
    assert hasattr(fp, "outcome_pattern")
    assert hasattr(fp, "amount_trend")
    
    # Assert isolation
    assert not any("hesitation" in k for e in trace.events for k in e.model_dump().keys())
    assert not any("attack_family" in k for e in trace.events for k in e.model_dump().keys())
    
    # Transaction balances invariant must hold
    for i, e in enumerate(sim.generated_events):
        if e.envelope.event_type == "TRANSACTION":
            tx = e.payload.transaction
            pre = e.payload.pre_balance
            post = e.payload.post_balance
                
            if tx.status == "completed":
                assert pre - tx.amount == post
            else:
                assert pre == post
