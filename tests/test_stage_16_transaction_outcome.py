import pytest
from decimal import Decimal
from datetime import datetime

from red_team.world.world import NormalWorld
from red_team.world.state import WorldState
from red_team.attacks.simulator import StatefulSimulator, AttackPlan
from red_team.attacks.ato_signature import get_ato_signature
from red_team.schemas.events import EventType, TransactionEventPayload
from red_team.validation.realism import validate_attack_realism
from red_team.schemas.observable import ObservableTransactionEvent

def run_until_transaction(setup_func):
    """Helper to run simulation across multiple seeds until a transaction is produced."""
    for seed in range(50):
        world = NormalWorld(seed=seed)
        world.generate_population(n_customers=1)
        customer_id = list(world.get_state().customers.keys())[0]
        
        setup_func(world.get_state())
        
        sim = StatefulSimulator(world.get_state(), get_ato_signature(), seed=seed)
        plan = AttackPlan(attack_family="ACCOUNT_TAKEOVER", difficulty="easy")
        trace, gt = sim.generate_attack(plan, customer_id)
        
        tx_events = [e for e in sim.generated_events if e.envelope.event_type == EventType.TRANSACTION]
        if tx_events:
            return sim, trace, gt
    pytest.fail("Failed to produce a transaction event across 50 seeds.")

def test_approved_sufficient_funds():
    def setup(ws):
        for acct in ws.accounts.values():
            acct.balance = Decimal("10000.00")
            
    sim, trace, gt = run_until_transaction(setup)
    tx_events = [e for e in sim.generated_events if e.envelope.event_type == EventType.TRANSACTION]
    
    # Verify properties independent of seed
    for e in tx_events:
        payload = e.payload
        assert isinstance(payload, TransactionEventPayload)
        assert payload.transaction.status == "completed"
        # balance before - amount = balance after
        assert payload.post_balance == payload.pre_balance - payload.transaction.amount

def test_approved_cannot_overdraw():
    def setup(ws):
        for acct in ws.accounts.values():
            acct.balance = Decimal("0.00")
            
    sim, trace, gt = run_until_transaction(setup)
    tx_events = [e for e in sim.generated_events if e.envelope.event_type == EventType.TRANSACTION]
    
    for e in tx_events:
        payload = e.payload
        assert isinstance(payload, TransactionEventPayload)
        assert payload.transaction.status == "failed"
        
def test_declined_insufficient_funds():
    def setup(ws):
        for acct in ws.accounts.values():
            acct.balance = Decimal("10.00")
            
    sim, trace, gt = run_until_transaction(setup)
    tx_events = [e for e in sim.generated_events if e.envelope.event_type == EventType.TRANSACTION]
    for e in tx_events:
        payload = e.payload
        assert payload.transaction.status == "failed"
        assert payload.post_balance == payload.pre_balance

def test_validator_accepts_realistic_declined_transaction():
    def setup(ws):
        for acct in ws.accounts.values():
            acct.balance = Decimal("0.00")
            
    sim, trace, gt = run_until_transaction(setup)
    
    txs = [e for e in trace.events if e.event_type == "TRANSACTION"]
    assert any(getattr(t, "transaction_status", "") == "failed" for t in txs)
    
    report = validate_attack_realism(trace, gt, get_ato_signature(), sim.state)
    assert report.status == "ACCEPTED"
    assert report.constraint.passed

def test_validator_rejects_impossible_approved_transaction():
    def setup(ws):
        for acct in ws.accounts.values():
            acct.balance = Decimal("0.00")
            
    sim, trace, gt = run_until_transaction(setup)
    
    for e in trace.events:
        if e.event_type == "TRANSACTION":
            e.transaction_status = "completed"
            
    report = validate_attack_realism(trace, gt, get_ato_signature(), sim.state)
    assert report.status == "REJECTED"
    assert not report.constraint.passed
    assert any("Impossible balance transition" in f for f in report.failures)
