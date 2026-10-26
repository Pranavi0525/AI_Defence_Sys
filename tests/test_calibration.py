"""Tests for Stage 7 — Calibration Metrics."""

import pytest

from red_team.world.world import NormalWorld
from red_team.schemas.calibration import (
    MarginalCalibrationConfig,
    FeaturePairCalibration,
    FeatureType,
    MetricType,
    ThresholdDirection,
)
from red_team.calibration import (
    CalibrationReport,
    ReferenceMode,
    calibrate_marginal,
    calibrate_dependency,
    calibrate_temporal,
    calibrate_behavioral,
    calibrate_graph_metrics,
    validate_structural,
)


def test_marginal_calibration_no_reference():
    config = MarginalCalibrationConfig(
        feature_name="amount",
        feature_type=FeatureType.NUMERICAL,
        metric=MetricType.KS_STATISTIC,
        threshold=0.05,
    )
    result = calibrate_marginal(config, [], reference_statistic=None)
    assert result.reference_mode == ReferenceMode.REFERENCE_STATISTICS
    assert result.value == "NOT_AVAILABLE"
    assert result.pass_fail == "NOT_EVALUATED"


def test_marginal_calibration_with_reference():
    config = MarginalCalibrationConfig(
        feature_name="amount",
        feature_type=FeatureType.NUMERICAL,
        metric=MetricType.KS_STATISTIC,
        threshold=0.05,
    )
    result = calibrate_marginal(config, [], reference_statistic={"median": 100})
    assert result.reference_mode == ReferenceMode.REFERENCE_STATISTICS
    assert result.value == "NOT_AVAILABLE"  # We don't have the math implemented yet, just the structure
    assert result.pass_fail == "NOT_EVALUATED"


def test_dependency_calibration():
    config = FeaturePairCalibration(
        feature_a="amount",
        feature_b="tx_type",
        feature_types=(FeatureType.NUMERICAL, FeatureType.CATEGORICAL),
        metric=MetricType.MUTUAL_INFORMATION,
        reason="Check if amount varies by type",
    )
    result = calibrate_dependency(config, [])
    assert result.value == "NOT_AVAILABLE"


def test_temporal_calibration():
    result = calibrate_temporal("inter_event_time", [])
    assert result.value == "NOT_AVAILABLE"


def test_behavioral_calibration():
    result = calibrate_behavioral("merchant_persistence", [])
    assert result.value == "NOT_AVAILABLE"


def test_graph_calibration():
    world = NormalWorld(seed=1)
    world.generate_population(n_customers=5)
    result = calibrate_graph_metrics("devices_per_customer", world.get_state().graph, reference_statistic=2.0)
    assert result.synthetic_value != "NOT_CALCULATED"
    assert result.value == "NOT_AVAILABLE"  # Distance not calculated


def test_structural_validation():
    world = NormalWorld(seed=1)
    world.generate_population(n_customers=5)
    world.generate_legitimate_events(num_events=10)
    result = validate_structural(world.get_state())
    assert result.value is True
    assert result.pass_fail == "PASS"


def test_end_to_end_calibration():
    world = NormalWorld(seed=42)
    world.generate_population(n_customers=10)
    world.generate_legitimate_events(num_events=50)
    
    report = CalibrationReport()
    
    # 1. Marginal
    mc = MarginalCalibrationConfig(
        feature_name="amount",
        feature_type=FeatureType.NUMERICAL,
        metric=MetricType.KS_STATISTIC,
    )
    report.add_result("marginal", calibrate_marginal(mc, world.get_events()))
    
    # 2. Dependency
    dc = FeaturePairCalibration(
        feature_a="amount",
        feature_b="channel",
        feature_types=(FeatureType.NUMERICAL, FeatureType.CATEGORICAL),
        metric=MetricType.MUTUAL_INFORMATION,
        reason="Testing",
    )
    report.add_result("dependency", calibrate_dependency(dc, world.get_events()))
    
    # 3. Temporal
    report.add_result("temporal", calibrate_temporal("events_per_day", world.get_events()))
    
    # 4. Behavioral
    report.add_result("behavioral", calibrate_behavioral("tx_per_customer", world.get_events()))
    
    # 5. Graph
    report.add_result("graph", calibrate_graph_metrics("node_count", world.get_state().graph))
    
    # 6. Structural
    report.add_result("structural", validate_structural(world.get_state()))
    
    assert len(report.marginal_results) == 1
    assert len(report.dependency_results) == 1
    assert len(report.temporal_results) == 1
    assert len(report.behavioral_results) == 1
    assert len(report.graph_results) == 1
    assert len(report.structural_results) == 1
    
    # Verify everything safely returned NOT_AVAILABLE except structural
    assert report.marginal_results[0].value == "NOT_AVAILABLE"
    assert report.structural_results[0].pass_fail == "PASS"


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
        
    r1 = validate_structural(world1.get_state())
    r2 = validate_structural(world2.get_state())
    
    assert r1.model_dump() == r2.model_dump()
