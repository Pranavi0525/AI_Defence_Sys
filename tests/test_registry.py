"""Tests for Stage 2.6 — Registry Integration."""

import pytest
from pydantic import ValidationError

from red_team.registry.registry import ProvenanceRegistry
from red_team.schemas.calibration import (
    CalibrationDefinition,
    FeaturePairCalibration,
    FeatureType,
    MarginalCalibrationConfig,
    MetricType,
)
from red_team.schemas.provenance import (
    DatasetSource,
    DatasetSourceType,
    FeatureProvenance,
    ProvenanceTier,
    ReferenceStatistic,
)


def _make_dataset(ds_id: str = "ds-1") -> DatasetSource:
    return DatasetSource(
        id=ds_id,
        name="PaySim",
        source_type=DatasetSourceType.SYNTHETIC_RESEARCH_DATASET,
        access_method="download",
        license="CC",
        redistribution_allowed=True,
        commercial_use_allowed=False,
        citation_required=True,
        raw_storage_policy="local_only",
    )


def _make_feature(fname: str = "amount", ds_id: str = "ds-1") -> FeatureProvenance:
    return FeatureProvenance(
        feature_name=fname,
        entity="transaction",
        data_type="numerical",
        provenance_tier=ProvenanceTier.TIER_1_LEARNED,
        source_datasets=[ds_id],
        source_fields=["amount"],
        derivation_method="kde",
        required_for_world_model=True,
    )


def _make_ref_stat(stat_id: str = "rs-1", ds_id: str = "ds-1") -> ReferenceStatistic:
    return ReferenceStatistic(
        id=stat_id,
        statistic_name="avg_amt",
        value=100.0,
        source_dataset_id=ds_id,
        source="Internal",
        statistic_definition="Average amount",
        externally_reported=False,
    )


def test_registry_dataset_valid():
    ds = _make_dataset()
    reg = ProvenanceRegistry(datasets=[ds])
    assert reg.get_dataset("ds-1").id == "ds-1"


def test_registry_dataset_duplicate_rejected():
    ds1 = _make_dataset("ds-1")
    ds2 = _make_dataset("ds-1")
    with pytest.raises(ValidationError, match="Duplicate DatasetSource ID: ds-1"):
        ProvenanceRegistry(datasets=[ds1, ds2])


def test_registry_dataset_not_found():
    reg = ProvenanceRegistry()
    with pytest.raises(KeyError, match="not found"):
        reg.get_dataset("unknown")


def test_registry_feature_valid():
    ds = _make_dataset("ds-1")
    fp = _make_feature("amount", "ds-1")
    reg = ProvenanceRegistry(datasets=[ds], features=[fp])
    assert reg.get_feature("amount").feature_name == "amount"


def test_registry_feature_duplicate_rejected():
    ds = _make_dataset("ds-1")
    fp1 = _make_feature("amount", "ds-1")
    fp2 = _make_feature("amount", "ds-1")
    with pytest.raises(ValidationError, match="Duplicate FeatureProvenance name: amount"):
        ProvenanceRegistry(datasets=[ds], features=[fp1, fp2])


def test_registry_feature_not_found():
    reg = ProvenanceRegistry()
    with pytest.raises(KeyError, match="not found"):
        reg.get_feature("unknown")


def test_registry_feature_references_unknown_dataset():
    fp = _make_feature("amount", "unknown-ds")
    with pytest.raises(ValidationError, match="references unknown source_dataset: unknown-ds"):
        ProvenanceRegistry(features=[fp])


def test_registry_calibration_valid():
    ds = _make_dataset("ds-1")
    fp1 = _make_feature("amount", "ds-1")
    fp2 = _make_feature("velocity", "ds-1")
    
    cal_def = CalibrationDefinition(
        marginal_configs=[
            MarginalCalibrationConfig(
                feature_name="amount",
                feature_type=FeatureType.NUMERICAL,
                metric=MetricType.KS_STATISTIC,
            )
        ],
        pair_configs=[
            FeaturePairCalibration(
                feature_a="amount",
                feature_b="velocity",
                feature_types=(FeatureType.NUMERICAL, FeatureType.NUMERICAL),
                metric=MetricType.PEARSON,
                reason="Relate",
            )
        ]
    )

    reg = ProvenanceRegistry(
        datasets=[ds],
        features=[fp1, fp2],
        calibration_definitions=[cal_def]
    )
    assert len(reg.calibration_definitions) == 1
    merged = reg.get_calibration_definition()
    assert len(merged.marginal_configs) == 1


def test_registry_calibration_unknown_marginal_feature():
    ds = _make_dataset("ds-1")
    fp1 = _make_feature("amount", "ds-1")
    
    cal_def = CalibrationDefinition(
        marginal_configs=[
            MarginalCalibrationConfig(
                feature_name="unknown_feature",
                feature_type=FeatureType.NUMERICAL,
                metric=MetricType.KS_STATISTIC,
            )
        ]
    )

    with pytest.raises(ValidationError, match="references unknown feature: unknown_feature"):
        ProvenanceRegistry(datasets=[ds], features=[fp1], calibration_definitions=[cal_def])


def test_registry_calibration_unknown_pair_feature_a():
    ds = _make_dataset("ds-1")
    fp2 = _make_feature("velocity", "ds-1")
    
    cal_def = CalibrationDefinition(
        pair_configs=[
            FeaturePairCalibration(
                feature_a="unknown_feature",
                feature_b="velocity",
                feature_types=(FeatureType.NUMERICAL, FeatureType.NUMERICAL),
                metric=MetricType.PEARSON,
                reason="Relate",
            )
        ]
    )

    with pytest.raises(ValidationError, match="unknown feature_a: unknown_feature"):
        ProvenanceRegistry(datasets=[ds], features=[fp2], calibration_definitions=[cal_def])


def test_registry_calibration_duplicate_across_definitions():
    ds = _make_dataset("ds-1")
    fp1 = _make_feature("amount", "ds-1")
    
    cal_def1 = CalibrationDefinition(
        marginal_configs=[
            MarginalCalibrationConfig(
                feature_name="amount",
                feature_type=FeatureType.NUMERICAL,
                metric=MetricType.KS_STATISTIC,
            )
        ]
    )
    cal_def2 = CalibrationDefinition(
        marginal_configs=[
            MarginalCalibrationConfig(
                feature_name="amount",
                feature_type=FeatureType.NUMERICAL,
                metric=MetricType.WASSERSTEIN,
            )
        ]
    )

    with pytest.raises(ValidationError, match="Duplicate marginal calibration for feature: amount"):
        ProvenanceRegistry(
            datasets=[ds],
            features=[fp1],
            calibration_definitions=[cal_def1, cal_def2]
        )


def test_registry_ref_stat_valid():
    ds = _make_dataset("ds-1")
    rs = _make_ref_stat("rs-1", "ds-1")
    reg = ProvenanceRegistry(datasets=[ds], reference_statistics=[rs])
    assert reg.get_reference_statistic("rs-1").id == "rs-1"


def test_registry_ref_stat_unknown_dataset():
    rs = _make_ref_stat("rs-1", "unknown-ds")
    with pytest.raises(ValidationError, match="references unknown source_dataset_id: unknown-ds"):
        ProvenanceRegistry(reference_statistics=[rs])


def test_registry_ref_stat_not_found():
    reg = ProvenanceRegistry()
    with pytest.raises(KeyError, match="not found"):
        reg.get_reference_statistic("unknown")


def test_registry_ref_stat_duplicate_rejected():
    ds = _make_dataset("ds-1")
    rs1 = _make_ref_stat("rs-1", "ds-1")
    rs2 = _make_ref_stat("rs-1", "ds-1")
    with pytest.raises(ValidationError, match="Duplicate ReferenceStatistic ID: rs-1"):
        ProvenanceRegistry(datasets=[ds], reference_statistics=[rs1, rs2])


def test_full_registry_graph():
    """Construct a complete small example with fully resolved references."""
    ds1 = _make_dataset("ds-1")
    ds2 = _make_dataset("ds-2")
    
    fp1 = _make_feature("amount", "ds-1")
    fp2 = _make_feature("velocity", "ds-1")
    fp3 = _make_feature("country", "ds-2")
    fp3.data_type = "categorical"
    
    cal_def = CalibrationDefinition(
        marginal_configs=[
            MarginalCalibrationConfig(
                feature_name="amount", feature_type=FeatureType.NUMERICAL, metric=MetricType.KS_STATISTIC
            ),
            MarginalCalibrationConfig(
                feature_name="country", feature_type=FeatureType.CATEGORICAL, metric=MetricType.CATEGORICAL_FREQUENCY
            )
        ],
        pair_configs=[
            FeaturePairCalibration(
                feature_a="amount", feature_b="velocity",
                feature_types=(FeatureType.NUMERICAL, FeatureType.NUMERICAL),
                metric=MetricType.PEARSON, reason="R"
            )
        ]
    )

    rs1 = _make_ref_stat("rs-1", "ds-1")
    rs2 = _make_ref_stat("rs-2", "ds-2")

    reg = ProvenanceRegistry(
        datasets=[ds1, ds2],
        features=[fp1, fp2, fp3],
        calibration_definitions=[cal_def],
        reference_statistics=[rs1, rs2]
    )
    
    # Verify everything looks ok
    assert len(reg.datasets) == 2
    assert len(reg.features) == 3
    assert len(reg.get_calibration_definition().marginal_configs) == 2
    assert len(reg.get_calibration_definition().pair_configs) == 1
    assert len(reg.reference_statistics) == 2
