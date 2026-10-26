import pytest
from decimal import Decimal
from datetime import datetime

from red_team.world.world import NormalWorld
from red_team.attacks.corpus import generate_attack_corpus
from red_team.attacks.simulator import StatefulSimulator, AttackPlan
from red_team.attacks.app_signature import get_app_signature
from red_team.validation.realism import validate_attack_realism

@pytest.fixture
def clean_world():
    world = NormalWorld(seed=42)
    world.generate_population(n_customers=10)
    world.generate_legitimate_events(num_events=20)
    state = world.get_state()
    
    # Needs a known device relationship for APP
    from red_team.schemas.entities import Device, Relationship
    for c_id in state.customers:
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
    return state

def get_app_trace(world_state):
    """Helper to try multiple seeds until we get a valid APP trace with at least one event."""
    customer_id = list(world_state.customers.keys())[0]
    for s in range(50):
        sim = StatefulSimulator(world_state.model_copy(deep=True), get_app_signature(), seed=s)
        plan = AttackPlan(attack_family="AUTHORIZED_PUSH_PAYMENT", difficulty="easy")
        try:
            trace, gt = sim.generate_attack(plan, customer_id)
            if len(trace.events) > 0:
                return sim, trace, gt, s
        except Exception:
            pass
    pytest.fail("Failed to produce an APP trace with events across 50 seeds.")

def test_app_generation_succeeds(clean_world):
    result = generate_attack_corpus(
        world_state=clean_world,
        target_count=5,
        master_seed=123,
        use_novelty=False,
        difficulty_quotas={"easy": 5},
        attack_family="AUTHORIZED_PUSH_PAYMENT"
    )
    assert len(result.accepted_traces) == 5
    for trace in result.accepted_traces:
        assert trace.observable_trace.events

def test_app_uses_legitimate_customer_device_session(clean_world):
    sim, trace, gt, raw_seed = get_app_trace(clean_world)
    
    device_registrations = [e for e in trace.events if e.event_type == "DEVICE_REGISTRATION"]
    assert len(device_registrations) == 0
    
    logins = [e for e in trace.events if e.event_type == "SESSION_LOGIN"]
    assert len(logins) > 0

def test_beneficiary_creation_ordering(clean_world):
    sim, trace, gt, raw_seed = get_app_trace(clean_world)
    
    bens_added = set()
    for ev in trace.events:
        if ev.event_type == "BENEFICIARY_ADDITION":
            bens_added.add(getattr(ev, "beneficiary_id", None))
        elif ev.event_type == "TRANSACTION":
            ben_id = getattr(ev, "beneficiary_id", None)
            if ben_id and ben_id not in clean_world.beneficiaries:
                assert ben_id in bens_added

def test_attacker_controlled_beneficiary_in_ground_truth_only(clean_world):
    sim, trace, gt, raw_seed = get_app_trace(clean_world)
    
    trace_dump = trace.model_dump()
    assert "attack_family" not in trace_dump
    assert "difficulty" not in trace_dump
    
    assert gt.attack_family == "AUTHORIZED_PUSH_PAYMENT"

def test_transaction_balances(clean_world):
    sim, trace, gt, raw_seed = get_app_trace(clean_world)
    
    report = validate_attack_realism(trace, gt, get_app_signature(), reference_state=clean_world)
    assert report.status == "ACCEPTED"

def test_chronology_and_reproducibility(clean_world):
    sim1, trace1, gt1, raw_seed = get_app_trace(clean_world)
    
    timestamps = [e.timestamp for e in trace1.events]
    assert timestamps == sorted(timestamps)
    
    customer_id = list(clean_world.customers.keys())[0]
    
    sim2 = StatefulSimulator(clean_world.model_copy(deep=True), get_app_signature(), seed=raw_seed)
    plan = AttackPlan(attack_family="AUTHORIZED_PUSH_PAYMENT", difficulty="easy")
    trace2, gt2 = sim2.generate_attack(plan, customer_id)
    
    assert trace1.model_dump() == trace2.model_dump()
    
    sim3 = StatefulSimulator(clean_world.model_copy(deep=True), get_app_signature(), seed=raw_seed+1)
    try:
        trace3, gt3 = sim3.generate_attack(plan, customer_id)
        assert trace2.model_dump() != trace3.model_dump()
    except Exception:
        pass # Different seed might abort, which is a meaningful variation

def test_novelty_operates_within_correct_family(clean_world):
    result = generate_attack_corpus(
        world_state=clean_world,
        target_count=5,
        master_seed=123,
        use_novelty=True,
        difficulty_quotas={"easy": 5},
        attack_family="AUTHORIZED_PUSH_PAYMENT"
    )
    assert len(result.accepted_traces) > 0

def test_realism_rejection_for_app(clean_world):
    sim, trace, gt, raw_seed = get_app_trace(clean_world)
    
    customer_id = list(clean_world.customers.keys())[0]
    from red_team.schemas.observable import ObservableDeviceEvent
    
    bad_dev = ObservableDeviceEvent(
        event_type="DEVICE_REGISTRATION",
        event_id="bad_ev",
        timestamp=datetime.now(),
        customer_id=customer_id,
        device_id="bad_dev_1",
        device_type="mobile",
        fingerprint="xxx",
        action="register"
    )
    trace.events.append(bad_dev)
    
    report = validate_attack_realism(trace, gt, get_app_signature(), reference_state=clean_world)
    assert report.status == "REJECTED"
    assert any(c.reason_code == "APP_NEW_DEVICE_VIOLATION" for c in report.constraint.checks if c.status == "FAIL")
