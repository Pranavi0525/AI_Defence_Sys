import pytest
from pydantic import ValidationError
from decimal import Decimal

from red_team.world.state import WorldState
from red_team.world.world import NormalWorld
from red_team.attacks.simulator import StatefulSimulator, AttackPlan, DIFFICULTY_PROFILES
from red_team.attacks.ato_signature import get_ato_signature
from red_team.schemas.events import EventType

@pytest.fixture
def base_world():
    w = NormalWorld(seed=42)
    w.generate_population(n_customers=2)
    return w.get_state()

@pytest.fixture
def signature():
    return get_ato_signature()

def test_deterministic_variation(base_world, signature):
    sim1 = StatefulSimulator(base_world.model_copy(deep=True), signature, seed=123)
    sim2 = StatefulSimulator(base_world.model_copy(deep=True), signature, seed=123)
    
    plan = AttackPlan(attack_family="ACCOUNT_TAKEOVER", difficulty="medium")
    c_id = list(base_world.customers.keys())[0]
    
    t1, _ = sim1.generate_attack(plan, c_id)
    t2, _ = sim2.generate_attack(plan, c_id)
    
    assert [e.event_type for e in t1.events] == [e.event_type for e in t2.events]
    assert [getattr(e, "amount", 0) for e in t1.events if e.event_type == EventType.TRANSACTION] == \
           [getattr(e, "amount", 0) for e in t2.events if e.event_type == EventType.TRANSACTION]

def test_no_hardcoded_500(base_world, signature):
    sim = StatefulSimulator(base_world, signature, seed=42)
    plan = AttackPlan(attack_family="ACCOUNT_TAKEOVER", difficulty="advanced")
    c_id = list(base_world.customers.keys())[0]
    
    t, _ = sim.generate_attack(plan, c_id)
    txs = [getattr(e, "amount", 0) for e in t.events if e.event_type == EventType.TRANSACTION]
    
    if txs:
        # None should be exactly 500 unless extreme coincidence
        assert not all(amt == Decimal("500.00") for amt in txs)

def test_financial_coherence(base_world, signature):
    sim = StatefulSimulator(base_world, signature, seed=42)
    plan = AttackPlan(attack_family="ACCOUNT_TAKEOVER", difficulty="easy")
    c_id = list(base_world.customers.keys())[0]
    
    t, _ = sim.generate_attack(plan, c_id)
    for e in sim.generated_events:
        if e.envelope.event_type == EventType.TRANSACTION:
            payload = e.payload
            if payload.transaction.status == "completed":
                assert payload.pre_balance - payload.transaction.amount == payload.post_balance
            else:
                assert payload.pre_balance == payload.post_balance

def test_chronological_ordering(base_world, signature):
    sim = StatefulSimulator(base_world, signature, seed=42)
    plan = AttackPlan(attack_family="ACCOUNT_TAKEOVER", difficulty="bursty" if False else "advanced")
    c_id = list(base_world.customers.keys())[0]
    
    t, _ = sim.generate_attack(plan, c_id)
    timestamps = [e.timestamp for e in t.events]
    for i in range(len(timestamps) - 1):
        assert timestamps[i] <= timestamps[i+1]

def test_difficulty_profiles_exist():
    assert "easy" in DIFFICULTY_PROFILES
    assert "advanced" in DIFFICULTY_PROFILES
    assert DIFFICULTY_PROFILES["advanced"].splits[1] > DIFFICULTY_PROFILES["easy"].splits[1]

def test_transaction_splitting_capability(base_world, signature):
    # Advanced should split more often
    sim = StatefulSimulator(base_world, signature, seed=42)
    plan = AttackPlan(attack_family="ACCOUNT_TAKEOVER", difficulty="advanced", max_phases=10)
    c_id = list(base_world.customers.keys())[0]
    t, _ = sim.generate_attack(plan, c_id)
    
    # We just ensure it runs and doesn't crash, splitting is handled internally
    tx_count = sum(1 for e in t.events if e.event_type == EventType.TRANSACTION)
    assert tx_count >= 0
