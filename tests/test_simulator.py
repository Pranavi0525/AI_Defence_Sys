"""Tests for Stage 9 — Stateful ATO Simulator."""

import pytest

from red_team.world.world import NormalWorld
from red_team.attacks.ato_signature import get_ato_signature
from red_team.attacks.simulator import StatefulSimulator, AttackPlan


def test_basic_generation():
    # Setup world
    world = NormalWorld(seed=1)
    world.generate_population(n_customers=5)
    world.generate_legitimate_events(num_events=10)
    
    customer_id = list(world.get_state().customers.keys())[0]
    sig = get_ato_signature()
    plan = AttackPlan(
        attack_family="ACCOUNT_TAKEOVER",
        difficulty="medium",
        entry_state="RECONNAISSANCE"
    )
    
    sim = StatefulSimulator(world.get_state(), sig, seed=42)
    trace, ground_truth = sim.generate_attack(plan, customer_id)
    
    assert trace.customer_id == customer_id
    assert len(trace.events) > 0
    assert len(ground_truth.linked_event_ids) == len(trace.events)
    assert ground_truth.attack_family == "ACCOUNT_TAKEOVER"
    assert ground_truth.attack_difficulty == "medium"
    assert len(ground_truth.phases_executed) > 0


def test_entry_path_selection():
    world = NormalWorld(seed=1)
    world.generate_population(n_customers=5)
    customer_id = list(world.get_state().customers.keys())[0]
    sig = get_ato_signature()
    
    # Test specific entry
    plan = AttackPlan(
        attack_family="ACCOUNT_TAKEOVER",
        difficulty="easy",
        entry_state="ACCOUNT_ACCESS"
    )
    
    sim = StatefulSimulator(world.get_state(), sig, seed=2)
    trace, gt = sim.generate_attack(plan, customer_id)
    
    assert gt.phases_executed[0].phase == "ACCOUNT_ACCESS"


def test_reproducibility():
    import uuid
    import random
    from unittest.mock import patch
    
    rng = random.Random(42)
    def mock_uuid4():
        return uuid.UUID(int=rng.getrandbits(128))
        
    with patch("uuid.uuid4", side_effect=mock_uuid4):
        world = NormalWorld(seed=42)
        world.generate_population(n_customers=2)
        customer_id = list(world.get_state().customers.keys())[0]
        sig = get_ato_signature()
        plan = AttackPlan(
            attack_family="ACCOUNT_TAKEOVER",
            difficulty="hard"
        )
        sim1 = StatefulSimulator(world.get_state(), sig, seed=123)
        trace1, gt1 = sim1.generate_attack(plan, customer_id)
        
    rng = random.Random(42)
    with patch("uuid.uuid4", side_effect=mock_uuid4):
        world2 = NormalWorld(seed=42)
        world2.generate_population(n_customers=2)
        customer_id2 = list(world2.get_state().customers.keys())[0]
        sim2 = StatefulSimulator(world2.get_state(), sig, seed=123)
        trace2, gt2 = sim2.generate_attack(plan, customer_id2)
    
    dump1 = gt1.model_dump()
    dump2 = gt2.model_dump()
    dump1['generation_metadata']['generated_at'] = None
    dump2['generation_metadata']['generated_at'] = None
    
    assert trace1.model_dump() == trace2.model_dump()
    assert dump1 == dump2


def test_bounded_execution():
    world = NormalWorld(seed=1)
    world.generate_population(n_customers=2)
    customer_id = list(world.get_state().customers.keys())[0]
    sig = get_ato_signature()
    plan = AttackPlan(
        attack_family="ACCOUNT_TAKEOVER",
        difficulty="easy",
        max_phases=2
    )
    
    sim = StatefulSimulator(world.get_state(), sig, seed=99)
    trace, gt = sim.generate_attack(plan, customer_id)
    
    assert len(gt.phases_executed) <= 2


def test_chronological_ordering():
    world = NormalWorld(seed=1)
    world.generate_population(n_customers=2)
    customer_id = list(world.get_state().customers.keys())[0]
    sig = get_ato_signature()
    plan = AttackPlan(
        attack_family="ACCOUNT_TAKEOVER",
        difficulty="easy"
    )
    
    sim = StatefulSimulator(world.get_state(), sig, seed=1)
    trace, gt = sim.generate_attack(plan, customer_id)
    
    prev_time = trace.events[0].timestamp
    for e in trace.events[1:]:
        assert e.timestamp >= prev_time
        prev_time = e.timestamp


def test_invalid_customer_rejected():
    world = NormalWorld(seed=1)
    world.generate_population(n_customers=2)
    sig = get_ato_signature()
    plan = AttackPlan(
        attack_family="ACCOUNT_TAKEOVER",
        difficulty="easy"
    )
    
    sim = StatefulSimulator(world.get_state(), sig, seed=1)
    with pytest.raises(ValueError, match="not found"):
        sim.generate_attack(plan, "INVALID_CUSTOMER")


def test_difficulty_variants():
    world = NormalWorld(seed=1)
    world.generate_population(n_customers=2)
    customer_id = list(world.get_state().customers.keys())[0]
    sig = get_ato_signature()
    
    for diff in ["easy", "medium", "hard", "advanced"]:
        plan = AttackPlan(
            attack_family="ACCOUNT_TAKEOVER",
            difficulty=diff
        )
        sim = StatefulSimulator(world.get_state(), sig, seed=42)
        trace, gt = sim.generate_attack(plan, customer_id)
        assert len(trace.events) > 0
        assert gt.attack_difficulty == diff
