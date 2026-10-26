"""Tests for Stage 10 — Observable / Ground-Truth Isolation Integration.

Proves the Stage 9 simulator pipeline strictly enforces separation between
what is observable (Blue Team) and what is ground truth (Internal evaluation).
"""

import json
import pytest

from red_team.world.world import NormalWorld
from red_team.attacks.ato_signature import get_ato_signature
from red_team.attacks.simulator import StatefulSimulator, AttackPlan
from red_team.schemas.observable import ObservableAttackTrace, extract_observable


FORBIDDEN_FIELDS = {
    "attack_id", "attack_family", "attack_phase", "is_fraud", "label",
    "hidden_objective", "attacker_intent", "genai_used", "attack_type",
    "ground_truth", "planner_metadata", "generation_metadata",
    "evaluation_metadata", "random_seed", "configuration_hash",
    "signature_version", "provenance_registry_version", "model_version",
    "prompt_hash", "realism_score", "novelty_score", "difficulty"
}


def _recursive_key_search(data, keys_found):
    """Recursively search for any keys in a dictionary or list."""
    if isinstance(data, dict):
        for k, v in data.items():
            keys_found.add(k)
            _recursive_key_search(v, keys_found)
    elif isinstance(data, list):
        for item in data:
            _recursive_key_search(item, keys_found)


def test_full_attack_generation_artifacts():
    world = NormalWorld(seed=1)
    world.generate_population(n_customers=2)
    customer_id = list(world.get_state().customers.keys())[0]
    
    plan = AttackPlan(attack_family="ACCOUNT_TAKEOVER", difficulty="medium")
    sim = StatefulSimulator(world.get_state(), get_ato_signature(), seed=42)
    
    trace, gt = sim.generate_attack(plan, customer_id)
    
    assert trace is not None
    assert gt is not None


def test_observable_serialization_no_leakage():
    world = NormalWorld(seed=1)
    world.generate_population(n_customers=2)
    customer_id = list(world.get_state().customers.keys())[0]
    
    plan = AttackPlan(attack_family="ACCOUNT_TAKEOVER", difficulty="hard")
    sim = StatefulSimulator(world.get_state(), get_ato_signature(), seed=42)
    
    trace, gt = sim.generate_attack(plan, customer_id)
    
    # 1. Inspect dict dump
    trace_dict = trace.model_dump()
    found_keys = set()
    _recursive_key_search(trace_dict, found_keys)
    
    for forbidden in FORBIDDEN_FIELDS:
        assert forbidden not in found_keys, f"Leakage found: '{forbidden}' in trace."
        
    # 2. Inspect JSON dump string manually for exact word matches (extra safety)
    trace_json = trace.model_dump_json()
    for forbidden in FORBIDDEN_FIELDS:
        assert f'"{forbidden}"' not in trace_json, f"Leakage found in JSON string: '{forbidden}'"


def test_ground_truth_serialization():
    world = NormalWorld(seed=1)
    world.generate_population(n_customers=2)
    customer_id = list(world.get_state().customers.keys())[0]
    
    plan = AttackPlan(attack_family="ACCOUNT_TAKEOVER", difficulty="easy")
    sim = StatefulSimulator(world.get_state(), get_ato_signature(), seed=42)
    
    trace, gt = sim.generate_attack(plan, customer_id)
    
    gt_dump = gt.model_dump()
    assert "attack_id" in gt_dump
    assert gt_dump["attack_family"] == "ACCOUNT_TAKEOVER"
    assert gt_dump["attack_difficulty"] == "easy"
    assert "phases_executed" in gt_dump
    assert "generation_metadata" in gt_dump
    assert "planner_metadata" in gt_dump
    
    # Verify events aren't duplicated fully
    assert "linked_event_ids" in gt_dump
    assert isinstance(gt_dump["linked_event_ids"], list)
    assert len(gt_dump["linked_event_ids"]) > 0


def test_event_id_linkage():
    world = NormalWorld(seed=1)
    world.generate_population(n_customers=2)
    customer_id = list(world.get_state().customers.keys())[0]
    
    plan = AttackPlan(attack_family="ACCOUNT_TAKEOVER", difficulty="medium")
    sim = StatefulSimulator(world.get_state(), get_ato_signature(), seed=42)
    
    trace, gt = sim.generate_attack(plan, customer_id)
    
    trace_event_ids = {e.event_id for e in trace.events}
    gt_event_ids = set(gt.linked_event_ids)
    
    assert trace_event_ids == gt_event_ids, "Linked events do not exactly match generated observable events."


def test_transformation_layer():
    world = NormalWorld(seed=1)
    world.generate_population(n_customers=2)
    customer_id = list(world.get_state().customers.keys())[0]
    
    plan = AttackPlan(attack_family="ACCOUNT_TAKEOVER", difficulty="medium")
    sim = StatefulSimulator(world.get_state(), get_ato_signature(), seed=42)
    
    trace, gt = sim.generate_attack(plan, customer_id)
    
    # Use extract_observable manually
    manual_trace = extract_observable(sim.generated_events, gt.attack_id)
    assert trace.model_dump() == manual_trace.model_dump()


def test_tampering_rejection():
    world = NormalWorld(seed=1)
    world.generate_population(n_customers=2)
    customer_id = list(world.get_state().customers.keys())[0]
    
    plan = AttackPlan(attack_family="ACCOUNT_TAKEOVER", difficulty="medium")
    sim = StatefulSimulator(world.get_state(), get_ato_signature(), seed=42)
    
    trace, gt = sim.generate_attack(plan, customer_id)
    
    # Attempt to load a dictionary with forbidden fields into ObservableAttackTrace
    tampered_data = trace.model_dump()
    tampered_data["attack_family"] = "ACCOUNT_TAKEOVER"
    tampered_data["difficulty"] = "medium"
    tampered_data["random_seed"] = 42
    
    import pydantic
    with pytest.raises(pydantic.ValidationError) as exc_info:
        ObservableAttackTrace.model_validate(tampered_data)
        
    errors = str(exc_info.value)
    assert "Extra inputs are not permitted" in errors
    assert "attack_family" in errors
    assert "difficulty" in errors
    assert "random_seed" in errors


def test_reproducibility_and_isolation():
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
        plan = AttackPlan(attack_family="ACCOUNT_TAKEOVER", difficulty="hard")
        sim1 = StatefulSimulator(world.get_state(), get_ato_signature(), seed=123)
        trace1, gt1 = sim1.generate_attack(plan, customer_id)
        
    rng = random.Random(42)
    with patch("uuid.uuid4", side_effect=mock_uuid4):
        world2 = NormalWorld(seed=42)
        world2.generate_population(n_customers=2)
        customer_id2 = list(world2.get_state().customers.keys())[0]
        sim2 = StatefulSimulator(world2.get_state(), get_ato_signature(), seed=123)
        trace2, gt2 = sim2.generate_attack(plan, customer_id2)
        
    dump1 = gt1.model_dump()
    dump2 = gt2.model_dump()
    dump1['generation_metadata']['generated_at'] = None
    dump2['generation_metadata']['generated_at'] = None
    
    assert trace1.model_dump() == trace2.model_dump()
    assert dump1 == dump2
    
    # Verify isolation on both traces independently
    found_keys = set()
    _recursive_key_search(trace1.model_dump(), found_keys)
    _recursive_key_search(trace2.model_dump(), found_keys)
    for forbidden in FORBIDDEN_FIELDS:
        assert forbidden not in found_keys
