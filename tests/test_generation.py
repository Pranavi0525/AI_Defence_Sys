"""Tests for Stage 12 — Generate and Validate 100 ATO Traces."""

import pytest
import uuid
import random
from unittest.mock import patch

from red_team.world.world import NormalWorld
from red_team.attacks.corpus import generate_attack_corpus


def test_generation_protocol():
    # Deterministic generation
    rng = random.Random(42)
    def mock_uuid4():
        return uuid.UUID(int=rng.getrandbits(128))
        
    with patch("uuid.uuid4", side_effect=mock_uuid4):
        world = NormalWorld(seed=42)
        world.generate_population(n_customers=10)
        
        # Test count 100
        result = generate_attack_corpus(world.get_state(), target_count=100, master_seed=42, max_attempts=500)
        
        assert result.generation_statistics.accepted == 100, f"Only generated {result.generation_statistics.accepted} valid traces."
        
        # Isolation: zero hidden field leakage
        forbidden = {"attack_family", "difficulty", "random_seed"}
        for rec in result.accepted_traces:
            trace_json = rec.observable_trace.model_dump_json()
            for f in forbidden:
                assert f'"{f}"' not in trace_json, f"Leakage {f} found in observable trace!"
                
        # Ground truth structure
        for rec in result.accepted_traces:
            assert rec.ground_truth.attack_id == rec.observable_trace.trace_id
            
        # Structure and Constraint gates
        for rec in result.accepted_traces:
            assert rec.validation_metadata.status == "ACCEPTED"
            assert rec.validation_metadata.structural.passed
            assert rec.validation_metadata.constraint.passed
            
        # Diversity
        stats = result.generation_statistics
        assert stats.unique_phase_sequences > 1
        assert stats.unique_entry_paths > 1
        assert stats.unique_customers_attacked > 1
        assert len(stats.difficulty_distribution) > 1


def test_reproducibility():
    rng = random.Random(42)
    def mock_uuid4():
        return uuid.UUID(int=rng.getrandbits(128))
        
    with patch("uuid.uuid4", side_effect=mock_uuid4):
        world = NormalWorld(seed=42)
        world.generate_population(n_customers=10)
        
        result1 = generate_attack_corpus(world.get_state(), target_count=10, master_seed=42)
        
    rng = random.Random(42)
    with patch("uuid.uuid4", side_effect=mock_uuid4):
        world2 = NormalWorld(seed=42)
        world2.generate_population(n_customers=10)
        
        result2 = generate_attack_corpus(world2.get_state(), target_count=10, master_seed=42)
        
    # Ignoring generated_at
    for i in range(10):
        dump1 = result1.accepted_traces[i].model_dump()
        dump2 = result2.accepted_traces[i].model_dump()
        dump1["ground_truth"]["generation_metadata"]["generated_at"] = None
        dump2["ground_truth"]["generation_metadata"]["generated_at"] = None
        assert dump1 == dump2
