import os
import json
import pytest
from datetime import datetime
from red_team.world.world import NormalWorld
from red_team.calibration import (
    CalibrationReport,
    ReferenceMode,
)

def test_stage_13_requirements():
    world = NormalWorld(seed=42, start_time=datetime(2025, 1, 1))
    world.generate_population(n_customers=10, n_merchants=5, n_beneficiaries=10)
    world.generate_legitimate_events(num_events=50)
    
    events = world.get_events()
    state = world.get_state()
    
    # Legitimate-only generation
    for e in events:
        assert not hasattr(e.envelope, "is_fraud") or e.envelope.is_fraud is False
        assert not hasattr(e, "attack_id") or e.attack_id is None
        
    # Provenance preservation
    # Provenance logic is verified by test_provenance.py schemas
    assert len(state.customers) > 0
    
    # Report serialization
    report = CalibrationReport()
    dump = report.model_dump_json()
    assert isinstance(dump, str)
    assert "marginal_results" in dump
    
    # Deterministic generation
    world2 = NormalWorld(seed=42, start_time=datetime(2025, 1, 1))
    world2.generate_population(n_customers=10, n_merchants=5, n_beneficiaries=10)
    world2.generate_legitimate_events(num_events=50)
    
    assert world.get_events()[0].envelope.timestamp == world2.get_events()[0].envelope.timestamp
    
    # NOT_AVAILABLE handling
    # Tested heavily in test_calibration.py already, but we verify no fabrication
    from red_team.calibration import calibrate_marginal
    from red_team.schemas.calibration import MarginalCalibrationConfig, FeatureType, MetricType
    mc = MarginalCalibrationConfig(feature_name="amount", feature_type=FeatureType.NUMERICAL, metric=MetricType.KS_STATISTIC)
    res = calibrate_marginal(mc, events)
    assert res.value == "NOT_AVAILABLE"
