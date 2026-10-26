"""The single source of truth for all provenance and calibration metadata.

Provides construction-time cross-reference validation to ensure:
    1. FeatureProvenance references existing DatasetSources.
    2. CalibrationDefinitions reference existing FeatureProvenances.
    3. ReferenceStatistics reference existing DatasetSources (if applicable).
    4. Identifiers are unique across all tracked entities.
"""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field, model_validator

from red_team.schemas.calibration import CalibrationDefinition
from red_team.schemas.provenance import (
    DatasetSource,
    FeatureProvenance,
    ReferenceStatistic,
)


class ProvenanceRegistry(BaseModel):
    """Authoritative registry for datasets, features, calibration, and statistics.

    Validates all cross-references at construction time to ensure the
    metadata graph is complete and structurally sound.
    """

    datasets: list[DatasetSource] = Field(default_factory=list)
    features: list[FeatureProvenance] = Field(default_factory=list)
    calibration_definitions: list[CalibrationDefinition] = Field(default_factory=list)
    reference_statistics: list[ReferenceStatistic] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_cross_references(self) -> ProvenanceRegistry:
        """Enforce unique identifiers and validate all cross-references."""
        
        # 1. Enforce uniqueness and build lookup dictionaries
        dataset_lookup: Dict[str, DatasetSource] = {}
        for ds in self.datasets:
            if ds.id in dataset_lookup:
                raise ValueError(f"Duplicate DatasetSource ID: {ds.id}")
            dataset_lookup[ds.id] = ds

        feature_lookup: Dict[str, FeatureProvenance] = {}
        for fp in self.features:
            if fp.feature_name in feature_lookup:
                raise ValueError(f"Duplicate FeatureProvenance name: {fp.feature_name}")
            feature_lookup[fp.feature_name] = fp

        stat_lookup: Dict[str, ReferenceStatistic] = {}
        for rs in self.reference_statistics:
            if rs.id in stat_lookup:
                raise ValueError(f"Duplicate ReferenceStatistic ID: {rs.id}")
            stat_lookup[rs.id] = rs

        # For CalibrationDefinitions, we track all marginal and pair identities 
        # to ensure they are unique across the ENTIRE registry, not just within 
        # a single definition block.
        seen_marginals: set[str] = set()
        seen_pairs: set[str] = set()

        for cal_def in self.calibration_definitions:
            for marginal in cal_def.marginal_configs:
                if marginal.feature_name in seen_marginals:
                    raise ValueError(f"Duplicate marginal calibration for feature: {marginal.feature_name}")
                seen_marginals.add(marginal.feature_name)
                
                # Check feature exists in registry
                if marginal.feature_name not in feature_lookup:
                    raise ValueError(
                        f"Marginal calibration references unknown feature: {marginal.feature_name}"
                    )

            for pair in cal_def.pair_configs:
                canonical_name = pair.canonical_pair_name
                if canonical_name in seen_pairs:
                    raise ValueError(f"Duplicate feature pair calibration for: {canonical_name}")
                seen_pairs.add(canonical_name)

                # Check both features exist in registry
                if pair.feature_a not in feature_lookup:
                    raise ValueError(f"Feature pair calibration references unknown feature_a: {pair.feature_a}")
                if pair.feature_b not in feature_lookup:
                    raise ValueError(f"Feature pair calibration references unknown feature_b: {pair.feature_b}")

        # 2. Validate FeatureProvenance -> DatasetSource references
        for fp in self.features:
            for ds_id in fp.source_datasets:
                if ds_id not in dataset_lookup:
                    raise ValueError(
                        f"Feature '{fp.feature_name}' references unknown source_dataset: {ds_id}"
                    )

        # 3. Validate ReferenceStatistic -> DatasetSource references
        for rs in self.reference_statistics:
            if rs.source_dataset_id is not None:
                if rs.source_dataset_id not in dataset_lookup:
                    raise ValueError(
                        f"ReferenceStatistic '{rs.id}' references unknown source_dataset_id: {rs.source_dataset_id}"
                    )

        return self

    # ---------------------------------------------------------------------------
    # Typed Access Methods
    # ---------------------------------------------------------------------------

    def get_dataset(self, dataset_id: str) -> DatasetSource:
        """Retrieve a dataset source by ID."""
        for ds in self.datasets:
            if ds.id == dataset_id:
                return ds
        raise KeyError(f"DatasetSource not found: {dataset_id}")

    def get_feature(self, feature_name: str) -> FeatureProvenance:
        """Retrieve a feature provenance by name."""
        for fp in self.features:
            if fp.feature_name == feature_name:
                return fp
        raise KeyError(f"FeatureProvenance not found: {feature_name}")

    def get_reference_statistic(self, stat_id: str) -> ReferenceStatistic:
        """Retrieve a reference statistic by ID."""
        for rs in self.reference_statistics:
            if rs.id == stat_id:
                return rs
        raise KeyError(f"ReferenceStatistic not found: {stat_id}")

    def get_calibration_definition(self) -> CalibrationDefinition:
        """Return a consolidated view of all calibration definitions.
        
        Since definitions are typically constructed modularly but represent
        a single global contract, this merges them.
        """
        merged = CalibrationDefinition(marginal_configs=[], pair_configs=[])
        for cd in self.calibration_definitions:
            merged.marginal_configs.extend(cd.marginal_configs)
            merged.pair_configs.extend(cd.pair_configs)
        return merged
