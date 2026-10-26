"""Tests for Stage 3 — Registry Population."""

import pytest
from pydantic import ValidationError

from red_team.registry.population import build_first_slice_registry
from red_team.schemas.provenance import ProvenanceTier


def test_registry_population_validates():
    """Ensure the hardcoded initial configuration strictly resolves."""
    reg = build_first_slice_registry()
    
    # Verify dataset population
    assert len(reg.datasets) == 2
    assert reg.get_dataset("paysim_kaggle_v1") is not None
    assert reg.get_dataset("ieee_cis_fraud_v1") is not None
    
    # Verify feature count and tier diversity
    features = reg.features
    assert len(features) >= 5
    tiers = {f.provenance_tier for f in features}
    assert ProvenanceTier.TIER_1_LEARNED in tiers
    assert ProvenanceTier.TIER_2_DERIVED in tiers
    assert ProvenanceTier.TIER_3_DOMAIN_MODELED in tiers
    
    # Verify that all calibration references resolve correctly
    cal_def = reg.get_calibration_definition()
    assert len(cal_def.marginal_configs) >= 3
    assert len(cal_def.pair_configs) >= 1
    
    # Verify reference stats
    stat = reg.get_reference_statistic("stat_paysim_fraud_rate")
    assert stat.source_dataset_id == "paysim_kaggle_v1"
    assert stat.externally_reported is True


def test_registry_population_breaks_on_bad_reference():
    """Verify that manually breaking the reference causes validation to fail."""
    reg_valid = build_first_slice_registry()
    
    # Deliberately break a dataset reference in the first feature
    bad_features = [f.model_copy() for f in reg_valid.features]
    bad_features[0].source_datasets = ["unknown_dataset"]
    
    with pytest.raises(ValidationError, match="references unknown source_dataset: unknown_dataset"):
        type(reg_valid)(
            datasets=reg_valid.datasets,
            features=bad_features,
            calibration_definitions=reg_valid.calibration_definitions,
            reference_statistics=reg_valid.reference_statistics
        )
