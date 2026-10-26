"""Tests for Stage 2.4 — Provenance Models."""

from typing import Any

import pytest
from pydantic import ValidationError

from red_team.schemas.provenance import (
    CalibrationMode,
    DatasetSource,
    DatasetSourceType,
    FeatureProvenance,
    ProvenanceTier,
    ReferenceStatistic,
)


def test_dataset_source_valid():
    source = DatasetSource(
        id="ds-001",
        name="PaySim",
        source_type=DatasetSourceType.SYNTHETIC_RESEARCH_DATASET,
        access_method="download",
        license="CC BY 4.0",
        redistribution_allowed=True,
        commercial_use_allowed=False,
        citation_required=True,
        raw_storage_policy="local_only",
    )
    assert source.id == "ds-001"
    assert source.source_type == DatasetSourceType.SYNTHETIC_RESEARCH_DATASET
    assert source.raw_storage_policy == "local_only"


def test_dataset_source_invalid_type():
    with pytest.raises(ValidationError):
        DatasetSource(
            id="ds-001",
            name="PaySim",
            source_type="INVALID_TYPE", # type: ignore
            access_method="download",
            license="CC BY 4.0",
            redistribution_allowed=True,
            commercial_use_allowed=False,
            citation_required=True,
            raw_storage_policy="local_only",
        )


def test_dataset_source_invalid_storage_policy():
    with pytest.raises(ValidationError):
        DatasetSource(
            id="ds-001",
            name="PaySim",
            source_type=DatasetSourceType.SYNTHETIC_RESEARCH_DATASET,
            access_method="download",
            license="CC BY 4.0",
            redistribution_allowed=True,
            commercial_use_allowed=False,
            citation_required=True,
            raw_storage_policy="INVALID_POLICY", # type: ignore
        )


def test_dataset_source_missing_required():
    with pytest.raises(ValidationError):
        DatasetSource(
            id="ds-001",
            # missing name, etc.
        )


def test_dataset_source_serialization_roundtrip():
    source = DatasetSource(
        id="ds-001",
        name="PaySim",
        source_type=DatasetSourceType.SYNTHETIC_RESEARCH_DATASET,
        access_method="download",
        license="CC BY 4.0",
        redistribution_allowed=True,
        commercial_use_allowed=False,
        citation_required=True,
        raw_storage_policy="local_only",
    )
    data = source.model_dump()
    source2 = DatasetSource.model_validate(data)
    assert source2 == source


def test_provenance_tier_valid_values():
    assert ProvenanceTier.TIER_1_LEARNED == "TIER_1_LEARNED"
    assert ProvenanceTier.TIER_2_DERIVED == "TIER_2_DERIVED"
    assert ProvenanceTier.TIER_3_DOMAIN_MODELED == "TIER_3_DOMAIN_MODELED"


def test_calibration_mode_valid_values():
    assert CalibrationMode.RAW_DATA == "RAW_DATA"
    assert CalibrationMode.REFERENCE_STATISTICS == "REFERENCE_STATISTICS"


def test_feature_provenance_tier_1_valid():
    fp = FeatureProvenance(
        feature_name="transaction_amount",
        entity="transaction",
        data_type="numerical",
        provenance_tier=ProvenanceTier.TIER_1_LEARNED,
        source_datasets=["ds-001"],
        source_fields=["amount"],
        derivation_method="kde_fit",
        required_for_world_model=True,
    )
    assert fp.feature_name == "transaction_amount"


def test_feature_provenance_tier_1_invalid_no_source_dataset():
    with pytest.raises(ValidationError, match="TIER_1_LEARNED requires at least one source dataset"):
        FeatureProvenance(
            feature_name="transaction_amount",
            entity="transaction",
            data_type="numerical",
            provenance_tier=ProvenanceTier.TIER_1_LEARNED,
            source_datasets=[],
            source_fields=["amount"],
            derivation_method="kde_fit",
            required_for_world_model=True,
        )


def test_feature_provenance_tier_1_invalid_no_source_fields():
    with pytest.raises(ValidationError, match="TIER_1_LEARNED requires at least one source field"):
        FeatureProvenance(
            feature_name="transaction_amount",
            entity="transaction",
            data_type="numerical",
            provenance_tier=ProvenanceTier.TIER_1_LEARNED,
            source_datasets=["ds-001"],
            source_fields=[],
            derivation_method="kde_fit",
            required_for_world_model=True,
        )


def test_feature_provenance_tier_1_invalid_no_derivation_method():
    with pytest.raises(ValidationError, match="TIER_1_LEARNED requires a derivation_method"):
        FeatureProvenance(
            feature_name="transaction_amount",
            entity="transaction",
            data_type="numerical",
            provenance_tier=ProvenanceTier.TIER_1_LEARNED,
            source_datasets=["ds-001"],
            source_fields=["amount"],
            required_for_world_model=True,
        )


def test_feature_provenance_tier_2_valid():
    fp = FeatureProvenance(
        feature_name="transaction_velocity",
        entity="transaction",
        data_type="numerical",
        provenance_tier=ProvenanceTier.TIER_2_DERIVED,
        source_datasets=["ds-001"],
        derivation_method="moving_average_24h",
        required_for_world_model=True,
    )
    assert fp.feature_name == "transaction_velocity"


def test_feature_provenance_tier_2_invalid_no_derivation_method():
    with pytest.raises(ValidationError, match="TIER_2_DERIVED requires a derivation_method"):
        FeatureProvenance(
            feature_name="transaction_velocity",
            entity="transaction",
            data_type="numerical",
            provenance_tier=ProvenanceTier.TIER_2_DERIVED,
            source_datasets=["ds-001"],
            required_for_world_model=True,
        )


def test_feature_provenance_tier_3_valid():
    fp = FeatureProvenance(
        feature_name="beneficiary_relationship_age",
        entity="relationship",
        data_type="temporal",
        provenance_tier=ProvenanceTier.TIER_3_DOMAIN_MODELED,
        generation_method="random_exponential",
        required_for_world_model=True,
        assumptions=["Older accounts have longer relationships on average."],
    )
    assert fp.feature_name == "beneficiary_relationship_age"


def test_feature_provenance_tier_3_invalid_no_generation_method():
    with pytest.raises(ValidationError, match="TIER_3_DOMAIN_MODELED requires a generation_method"):
        FeatureProvenance(
            feature_name="beneficiary_relationship_age",
            entity="relationship",
            data_type="temporal",
            provenance_tier=ProvenanceTier.TIER_3_DOMAIN_MODELED,
            required_for_world_model=True,
            assumptions=["Assumption 1"],
        )


def test_feature_provenance_tier_3_invalid_no_assumptions():
    with pytest.raises(ValidationError, match="TIER_3_DOMAIN_MODELED requires at least one assumption"):
        FeatureProvenance(
            feature_name="beneficiary_relationship_age",
            entity="relationship",
            data_type="temporal",
            provenance_tier=ProvenanceTier.TIER_3_DOMAIN_MODELED,
            generation_method="random_exponential",
            required_for_world_model=True,
            assumptions=[],
        )


def test_feature_provenance_invalid_entity():
    with pytest.raises(ValidationError):
        FeatureProvenance(
            feature_name="test",
            entity="invalid_entity", # type: ignore
            data_type="numerical",
            provenance_tier=ProvenanceTier.TIER_3_DOMAIN_MODELED,
            generation_method="test",
            required_for_world_model=True,
            assumptions=["test"],
        )


def test_feature_provenance_invalid_data_type():
    with pytest.raises(ValidationError):
        FeatureProvenance(
            feature_name="test",
            entity="customer",
            data_type="invalid_type", # type: ignore
            provenance_tier=ProvenanceTier.TIER_3_DOMAIN_MODELED,
            generation_method="test",
            required_for_world_model=True,
            assumptions=["test"],
        )


def test_feature_provenance_duplicate_source_datasets():
    with pytest.raises(ValidationError, match="Duplicate source_datasets"):
        FeatureProvenance(
            feature_name="test",
            entity="customer",
            data_type="numerical",
            provenance_tier=ProvenanceTier.TIER_1_LEARNED,
            source_datasets=["ds-001", "ds-001"],
            source_fields=["field1"],
            derivation_method="test",
            required_for_world_model=True,
        )


def test_reference_statistic_valid_internal():
    rs = ReferenceStatistic(
        id="rs-001",
        statistic_name="avg_amount",
        value=100.0,
        source="Internal Team",
        statistic_definition="Average transaction amount",
        externally_reported=False,
    )
    assert rs.value == 100.0


def test_reference_statistic_valid_external():
    rs = ReferenceStatistic(
        id="rs-002",
        statistic_name="fraud_rate",
        value=0.01,
        source="Industry Report 2024",
        citation="DOI:10.1234/5678",
        statistic_definition="Percentage of transactions marked as fraud",
        externally_reported=True,
    )
    assert rs.value == 0.01


def test_reference_statistic_external_missing_citation():
    with pytest.raises(ValidationError, match="Externally reported statistics require a citation"):
        ReferenceStatistic(
            id="rs-003",
            statistic_name="fraud_rate",
            value=0.01,
            source="Industry Report 2024",
            statistic_definition="Percentage of transactions marked as fraud",
            externally_reported=True,
        )


def test_reference_statistic_unverified_estimate_valid():
    rs = ReferenceStatistic(
        id="rs-005",
        statistic_name="some_estimate",
        value=50.0,
        source="Community",
        citation="Forum post",
        statistic_definition="Just an estimate",
        externally_reported=True,
        verification_status="UNVERIFIED_ESTIMATE",
        derivation_notes="source reports/estimates this value; independently derived from raw data: no. Just a guess."
    )
    assert rs.verification_status == "UNVERIFIED_ESTIMATE"

def test_reference_statistic_unverified_estimate_invalid_notes():
    with pytest.raises(ValidationError, match="requires derivation_notes to include"):
        ReferenceStatistic(
            id="rs-006",
            statistic_name="some_estimate",
            value=50.0,
            source="Community",
            citation="Forum post",
            statistic_definition="Just an estimate",
            externally_reported=True,
            verification_status="UNVERIFIED_ESTIMATE",
            derivation_notes="I think this is right."
        )

def test_reference_statistic_unverified_estimate_must_be_external():
    with pytest.raises(ValidationError, match="UNVERIFIED_ESTIMATE must be externally_reported=True"):
        ReferenceStatistic(
            id="rs-007",
            statistic_name="some_estimate",
            value=50.0,
            source="Internal",
            statistic_definition="Just an estimate",
            externally_reported=False,
            verification_status="UNVERIFIED_ESTIMATE",
            derivation_notes="source reports/estimates this value; independently derived from raw data: no."
        )

def test_reference_statistic_serialization_roundtrip():
    rs = ReferenceStatistic(
        id="rs-004",
        statistic_name="fraud_rate",
        value={"high_risk": 0.05, "low_risk": 0.001},
        source="Internal Team",
        statistic_definition="Fraud rate by risk tier",
        externally_reported=False,
    )
    data = rs.model_dump()
    rs2 = ReferenceStatistic.model_validate(data)
    assert rs2 == rs


def test_separation_of_concepts():
    """Verify that source_type, provenance_tier, and calibration_mode can coexist independently."""
    
    source = DatasetSource(
        id="paysim-01",
        name="PaySim",
        source_type=DatasetSourceType.SYNTHETIC_RESEARCH_DATASET,
        access_method="download",
        license="Open",
        redistribution_allowed=True,
        commercial_use_allowed=False,
        citation_required=True,
        raw_storage_policy="local_only",
    )
    
    fp = FeatureProvenance(
        feature_name="transaction_amount",
        entity="transaction",
        data_type="numerical",
        provenance_tier=ProvenanceTier.TIER_1_LEARNED,
        source_datasets=[source.id],
        source_fields=["amount"],
        derivation_method="empirical_distribution",
        required_for_world_model=True,
    )
    
    # We calibrate using REFERENCE_STATISTICS (not raw data)
    mode = CalibrationMode.REFERENCE_STATISTICS
    
    assert source.source_type == DatasetSourceType.SYNTHETIC_RESEARCH_DATASET
    assert fp.provenance_tier == ProvenanceTier.TIER_1_LEARNED
    assert mode == CalibrationMode.REFERENCE_STATISTICS
