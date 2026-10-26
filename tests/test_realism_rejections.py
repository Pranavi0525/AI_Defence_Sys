import pytest
from datetime import datetime
from red_team.world.state import WorldState
from red_team.world.world import NormalWorld
from red_team.schemas.observable import ObservableAttackTrace, ObservableTransactionEvent
from red_team.schemas.ground_truth import AttackGroundTruth, AttackPhaseRecord
from red_team.validation.realism import validate_attack_realism
from red_team.attacks.ato_signature import get_ato_signature
from red_team.attacks.simulator import StatefulSimulator, AttackPlan
from decimal import Decimal

def test_multiple_simultaneous_failures():
    world = NormalWorld(seed=1)
    world.generate_population(n_customers=1)
    customer_id = list(world.get_state().customers.keys())[0]
    
    plan = AttackPlan(attack_family="ACCOUNT_TAKEOVER", difficulty="medium")
    sim = StatefulSimulator(world.get_state(), get_ato_signature(), seed=42)
    trace, gt = sim.generate_attack(plan, customer_id)
    
    # Mutate trace to cause multiple failures
    # 1. Reverse the order of events to cause TEMPORAL_ORDER_VIOLATION
    trace.events.reverse()
    
    # 2. Add a phantom entity reference
    e_phantom = ObservableTransactionEvent(
        event_type="TRANSACTION",
        event_id="e_phantom",
        timestamp=trace.events[-1].timestamp,
        customer_id=customer_id,
        account_id="phantom_acct",
        amount=Decimal("100.0"),
        currency="USD",
        transaction_type="transfer",
        channel="mobile",
        transaction_status="COMPLETED"
    )
    trace.events.append(e_phantom)
    
    report = validate_attack_realism(trace, gt, get_ato_signature(), world.get_state())
    
    assert report.status == "REJECTED"
    
    reasons = [r.reason_code for r in report.rejection_reasons]
    
    # GROUND_TRUTH_MISMATCH because we added e_phantom which isn't in GT
    assert "GROUND_TRUTH_MISMATCH" in reasons
    assert "TEMPORAL_ORDER_VIOLATION" in reasons
    assert "PHANTOM_ENTITY_REFERENCE" in reasons
    
    assert report.primary_failure.reason_code in ["GROUND_TRUTH_MISMATCH", "TEMPORAL_ORDER_VIOLATION", "PHANTOM_ENTITY_REFERENCE"]
    assert len(report.secondary_failures) > 0
    
def test_all_known_rejection_reasons_available():
    world = NormalWorld(seed=1)
    world.generate_population(n_customers=1)
    customer_id = list(world.get_state().customers.keys())[0]
    
    plan = AttackPlan(attack_family="ACCOUNT_TAKEOVER", difficulty="medium")
    sim = StatefulSimulator(world.get_state(), get_ato_signature(), seed=42)
    trace, gt = sim.generate_attack(plan, customer_id)
    
    # Cause a GROUND_TRUTH_MISMATCH by changing an event ID in trace
    if trace.events:
        trace.events[0].event_id = "hacked_id"
        
    report = validate_attack_realism(trace, gt, get_ato_signature(), world.get_state())
    
    assert report.status == "REJECTED"
    assert report.primary_failure.reason_code == "GROUND_TRUTH_MISMATCH"
    
    # We can also check serialization works cleanly
    dump = report.model_dump()
    assert "rejection_reasons" in dump
    assert dump["rejection_reasons"][0]["reason_code"] == "GROUND_TRUTH_MISMATCH"
