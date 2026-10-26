import pytest
from decimal import Decimal
from datetime import datetime
from copy import deepcopy

from red_team.world.world import NormalWorld
from red_team.world.state import WorldState
from red_team.attacks.simulator import StatefulSimulator, AttackPlan
from red_team.attacks.ato_signature import get_ato_signature
from red_team.schemas.events import EventType, TransactionEventPayload
from red_team.validation.realism import validate_attack_realism
from red_team.schemas.observable import ObservableTransactionEvent

# Negative tests for validator / invariants

def test_negative_completed_transaction_overdrawing():
    world = NormalWorld(seed=42)
    world.generate_population(n_customers=1)
    customer_id = list(world.get_state().customers.keys())[0]
    
    # 0 funds
    for acct in world.get_state().accounts.values():
        acct.balance = Decimal("0.00")
        
    sim = StatefulSimulator(world.get_state(), get_ato_signature(), seed=42)
    plan = AttackPlan(attack_family="ACCOUNT_TAKEOVER", difficulty="easy")
    trace, gt = sim.generate_attack(plan, customer_id)
    
    # Trace contains FAILED txs. Mutate to COMPLETED.
    for e in trace.events:
        if e.event_type == "TRANSACTION":
            e.transaction_status = "completed"
            
    # Validator checks it
    report = validate_attack_realism(trace, gt, get_ato_signature(), world.get_state())
    assert report.status == "REJECTED"
    assert not report.constraint.passed
    reasons = [r.reason_code for r in report.rejection_reasons]
    assert "BALANCE_CONSTRAINT_VIOLATION" in reasons


def test_negative_completed_inconsistent_post_balance():
    # The validator cannot see post_balance (observable isolation),
    # but we can test that the simulator NEVER produces such a state.
    world = NormalWorld(seed=42)
    world.generate_population(n_customers=1)
    customer_id = list(world.get_state().customers.keys())[0]
    
    sim = StatefulSimulator(world.get_state(), get_ato_signature(), seed=42)
    plan = AttackPlan(attack_family="ACCOUNT_TAKEOVER", difficulty="easy")
    trace, gt = sim.generate_attack(plan, customer_id)
    
    # Verify all generated events have perfectly consistent balances
    tx_events = [e for e in sim.generated_events if e.envelope.event_type == EventType.TRANSACTION]
    for e in tx_events:
        payload = e.payload
        if payload.transaction.status == "completed":
            assert payload.pre_balance - payload.transaction.amount == payload.post_balance
        elif payload.transaction.status == "failed":
            assert payload.pre_balance == payload.post_balance

def test_negative_failed_mutating_balance():
    world = NormalWorld(seed=42)
    world.generate_population(n_customers=1)
    customer_id = list(world.get_state().customers.keys())[0]
    
    # 0 funds
    for acct in world.get_state().accounts.values():
        acct.balance = Decimal("0.00")
        
    initial_balances = {a_id: acct.balance for a_id, acct in world.get_state().accounts.items()}
    
    sim = StatefulSimulator(world.get_state(), get_ato_signature(), seed=42)
    plan = AttackPlan(attack_family="ACCOUNT_TAKEOVER", difficulty="easy")
    trace, gt = sim.generate_attack(plan, customer_id)
    
    # The simulator generates a failed transaction.
    # The WorldState balance MUST remain 0.00.
    for a_id, acct in world.get_state().accounts.items():
        assert acct.balance == initial_balances[a_id]

def test_negative_phantom_account():
    world = NormalWorld(seed=42)
    world.generate_population(n_customers=1)
    customer_id = list(world.get_state().customers.keys())[0]
    
    sim = StatefulSimulator(world.get_state(), get_ato_signature(), seed=42)
    plan = AttackPlan(attack_family="ACCOUNT_TAKEOVER", difficulty="easy")
    trace, gt = sim.generate_attack(plan, customer_id)
    
    for e in trace.events:
        if e.event_type == "TRANSACTION":
            e.account_id = "nonexistent-acct"
            
    report = validate_attack_realism(trace, gt, get_ato_signature(), world.get_state())
    assert report.status == "REJECTED"
    assert not report.constraint.passed
    reasons = [r.reason_code for r in report.rejection_reasons]
    assert "PHANTOM_ENTITY_REFERENCE" in reasons

def test_negative_chronologically_impossible():
    world = NormalWorld(seed=42)
    world.generate_population(n_customers=1)
    customer_id = list(world.get_state().customers.keys())[0]
    
    sim = StatefulSimulator(world.get_state(), get_ato_signature(), seed=42)
    plan = AttackPlan(attack_family="ACCOUNT_TAKEOVER", difficulty="easy")
    trace, gt = sim.generate_attack(plan, customer_id)
    
    trace.events.reverse()
    
    report = validate_attack_realism(trace, gt, get_ato_signature(), world.get_state())
    assert report.status == "REJECTED"
    assert not report.structural.passed
    reasons = [r.reason_code for r in report.rejection_reasons]
    assert "TEMPORAL_ORDER_VIOLATION" in reasons
