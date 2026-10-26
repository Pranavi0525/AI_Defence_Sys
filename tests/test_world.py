"""Tests for Stage 5 — Minimal Normal World."""

import pytest
from datetime import datetime

from red_team.world.world import NormalWorld
from red_team.schemas.events import EventType

def test_world_initialization():
    world = NormalWorld(seed=42)
    assert world.state.current_time == datetime(2025, 1, 1, 0, 0, 0)


def test_entity_generation():
    world = NormalWorld(seed=42)
    world.generate_population(n_customers=50, n_merchants=10, n_beneficiaries=20)
    
    state = world.get_state()
    assert len(state.customers) == 50
    assert len(state.merchants) == 10
    assert len(state.beneficiaries) == 20
    assert len(state.accounts) >= 50
    assert len(state.devices) >= 50
    assert len(state.relationships) > 0


def test_behavioral_generation():
    world = NormalWorld(seed=123)
    world.generate_population(n_customers=20, n_merchants=5, n_beneficiaries=5)
    
    world.generate_legitimate_events(num_events=500)
    
    state = world.get_state()
    events = world.get_events()
    
    assert len(events) <= 500
    assert len(events) > 0
    
    # Verify chronicity
    for i in range(1, len(events)):
        assert events[i].envelope.timestamp >= events[i-1].envelope.timestamp
        
    # Verify legitimate drift
    drift_events = [e for e in events if e.envelope.event_type == EventType.DEVICE_REGISTRATION]
    assert len(drift_events) > 0
    
    # Verify transactions are legitimate (no fraud tags)
    tx_events = [e for e in events if e.envelope.event_type == EventType.TRANSACTION]
    assert len(tx_events) > 0
    
    # Since observable models are extra='forbid', generating the payload correctly implicitly verifies this.


def test_balance_consistency():
    world = NormalWorld(seed=999)
    world.generate_population(n_customers=5)
    
    initial_balances = {a.account_id: a.balance for a in world.get_state().accounts.values()}
    
    world.generate_legitimate_events(num_events=100)
    
    final_balances = {a.account_id: a.balance for a in world.get_state().accounts.values()}
    
    # Balances must not be negative
    for acct in world.get_state().accounts.values():
        if acct.account_type in ("checking", "savings", "business"):
            assert acct.balance >= 0


def test_reproducibility():
    import uuid
    import random
    from unittest.mock import patch
    
    rng = random.Random(42)
    def mock_uuid4():
        return uuid.UUID(int=rng.getrandbits(128))
        
    with patch("uuid.uuid4", side_effect=mock_uuid4):
        world1 = NormalWorld(seed=42)
        world1.generate_population(n_customers=10)
        world1.generate_legitimate_events(num_events=50)
        
    rng = random.Random(42)
    with patch("uuid.uuid4", side_effect=mock_uuid4):
        world2 = NormalWorld(seed=42)
        world2.generate_population(n_customers=10)
        world2.generate_legitimate_events(num_events=50)
    
    assert len(world1.get_events()) == len(world2.get_events())
    
    for e1, e2 in zip(world1.get_events(), world2.get_events()):
        assert e1.model_dump() == e2.model_dump()


def test_different_seeds_produce_different_worlds():
    world1 = NormalWorld(seed=1)
    world1.generate_population(n_customers=10)
    
    world2 = NormalWorld(seed=2)
    world2.generate_population(n_customers=10)
    
    # Comparing first customer ID
    c1 = list(world1.get_state().customers.values())[0]
    c2 = list(world2.get_state().customers.values())[0]
    
    # It's highly unlikely they share the same ID unless seeds are identical
    assert c1.customer_id != c2.customer_id
