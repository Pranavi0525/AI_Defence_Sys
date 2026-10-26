"""Provenance schemas for Red Team AI datasets, features, and statistics.

Ensures strict tracking of data origins, derivation methods, and assumptions
to make it impossible to silently represent a domain assumption as learned data.

Design decisions:
    - ProvenanceTier answers: "How did our system obtain/derive this feature?"
    - DatasetSourceType answers: "What kind of source is this?"
    - CalibrationMode answers: "Did we calculate it or use published statistics?"
    - FeatureProvenance strictly enforces tier-specific requirements (e.g.,
      Tier 3 requires assumptions, Tier 1 requires source datasets).
    - No registry-level cross-reference validation is performed here.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ProvenanceTier(str, Enum):
    """Rigorous classification of how a feature was derived."""

    TIER_1_LEARNED = "TIER_1_LEARNED"
    TIER_2_DERIVED = "TIER_2_DERIVED"
    TIER_3_DOMAIN_MODELED = "TIER_3_DOMAIN_MODELED"


class VerificationStatus(str, Enum):
    """Classification of reference statistic integrity."""

    VERIFIED_EXTERNAL = "VERIFIED_EXTERNAL"
    DERIVED_FROM_VERIFIED_EXTERNAL = "DERIVED_FROM_VERIFIED_EXTERNAL"
    UNVERIFIED_ESTIMATE = "UNVERIFIED_ESTIMATE"


class DatasetSourceType(str, Enum):
    """The fundamental nature of the dataset source."""

    REAL_WORLD_PRODUCTION = "REAL_WORLD_PRODUCTION"
    REAL_WORLD_RESEARCH = "REAL_WORLD_RESEARCH"
    SYNTHETIC_RESEARCH_DATASET = "SYNTHETIC_RESEARCH_DATASET"
    PUBLISHED_STATISTICS = "PUBLISHED_STATISTICS"
    DOMAIN_KNOWLEDGE = "DOMAIN_KNOWLEDGE"


class CalibrationMode(str, Enum):
    """How the calibration statistic was obtained."""

    RAW_DATA = "RAW_DATA"
    REFERENCE_STATISTICS = "REFERENCE_STATISTICS"


# ---------------------------------------------------------------------------
# Dataset Source Model
# ---------------------------------------------------------------------------

class DatasetSource(BaseModel):
    """Metadata for a dataset source."""

    id: str = Field(..., min_length=1, description="Unique dataset identifier.")
    name: str = Field(..., min_length=1, description="Human-readable name.")
    source_url: str | None = Field(default=None, description="URL to the source data or paper.")
    source_type: DatasetSourceType = Field(..., description="Nature of the source.")
    access_method: str = Field(..., min_length=1, description="How the data was accessed.")
    license: str = Field(..., min_length=1, description="Dataset license.")
    license_url: str | None = Field(default=None, description="URL to license text.")
    redistribution_allowed: bool = Field(..., description="Can we redistribute raw data?")
    commercial_use_allowed: bool = Field(..., description="Can we use commercially?")
    citation_required: bool = Field(..., description="Must we cite the source?")
    raw_storage_policy: Literal["local_only", "derived_stats_only", "not_stored"] = Field(
        ..., description="Policy for storing the raw data."
    )
    known_limitations: str | None = Field(default=None, description="Known issues/bias.")
    notes: str | None = Field(default=None, description="General notes.")


# ---------------------------------------------------------------------------
# Feature Provenance Model
# ---------------------------------------------------------------------------

class FeatureProvenance(BaseModel):
    """Metadata tracking the origin and derivation of a specific feature."""

    feature_name: str = Field(..., min_length=1, description="Name of the feature.")
    entity: Literal[
        "customer", "account", "device", "merchant",
        "beneficiary", "session", "transaction", "relationship"
    ] = Field(..., description="Canonical entity this feature belongs to.")
    data_type: Literal["numerical", "categorical", "temporal", "boolean"] = Field(
        ..., description="Nature of the data."
    )
    provenance_tier: ProvenanceTier = Field(..., description="How the feature was derived.")
    
    source_datasets: list[str] = Field(
        default_factory=list, description="IDs of source datasets."
    )
    source_fields: list[str] = Field(
        default_factory=list, description="Fields in the source datasets used."
    )
    derivation_method: str | None = Field(
        default=None, description="How the feature was derived from sources."
    )
    generation_method: str | None = Field(
        default=None, description="How the feature is generated synthetically."
    )
    calibration_metric: str | None = Field(
        default=None, description="Metric used for calibration."
    )
    required_for_world_model: bool = Field(
        ..., description="Is this feature mandatory for world state?"
    )
    assumptions: list[str] = Field(
        default_factory=list, description="Domain assumptions made."
    )

    @model_validator(mode="after")
    def _validate_tier_requirements(self) -> FeatureProvenance:
        """Enforce requirements specific to each ProvenanceTier."""
        tier = self.provenance_tier

        # Source dataset ID uniqueness
        if len(self.source_datasets) != len(set(self.source_datasets)):
            raise ValueError(f"Duplicate source_datasets found in {self.feature_name}")
            
        # Empty string checks for list elements
        if any(not d.strip() for d in self.source_datasets):
            raise ValueError("source_datasets cannot contain empty strings")
        if any(not f.strip() for f in self.source_fields):
            raise ValueError("source_fields cannot contain empty strings")
        if any(not a.strip() for a in self.assumptions):
            raise ValueError("assumptions cannot contain empty strings")

        if tier == ProvenanceTier.TIER_1_LEARNED:
            if not self.source_datasets:
                raise ValueError("TIER_1_LEARNED requires at least one source dataset.")
            if not self.source_fields:
                raise ValueError("TIER_1_LEARNED requires at least one source field.")
            if not self.derivation_method or not self.derivation_method.strip():
                raise ValueError("TIER_1_LEARNED requires a derivation_method.")

        elif tier == ProvenanceTier.TIER_2_DERIVED:
            if not self.source_datasets and not self.derivation_method:
                raise ValueError(
                    "TIER_2_DERIVED requires at least one source dataset OR a derivation_method."
                )
            if not self.derivation_method or not self.derivation_method.strip():
                raise ValueError("TIER_2_DERIVED requires a derivation_method.")

        elif tier == ProvenanceTier.TIER_3_DOMAIN_MODELED:
            if not self.generation_method or not self.generation_method.strip():
                raise ValueError("TIER_3_DOMAIN_MODELED requires a generation_method.")
            if not self.assumptions:
                raise ValueError("TIER_3_DOMAIN_MODELED requires at least one assumption.")

        return self


# ---------------------------------------------------------------------------
# Reference Statistic Model
# ---------------------------------------------------------------------------

class ReferenceStatistic(BaseModel):
    """An independently calculated or externally published reference statistic."""

    id: str = Field(..., min_length=1, description="Unique statistic identifier.")
    statistic_name: str = Field(..., min_length=1, description="Name of the statistic.")
    value: Any = Field(..., description="Statistic value (float, dict, list).")
    source_dataset_id: str | None = Field(
        default=None, description="Dataset ID if derived from one."
    )
    source: str = Field(..., min_length=1, description="Source name/author.")
    citation: str | None = Field(default=None, description="Formal citation or DOI.")
    statistic_definition: str = Field(..., min_length=1, description="What this statistic means.")
    derivation_notes: str | None = Field(default=None, description="How it was derived.")
    externally_reported: bool = Field(
        ..., description="True if from a paper/report, False if calculated internally."
    )
    calibration_mode: CalibrationMode = Field(
        default=CalibrationMode.REFERENCE_STATISTICS,
        description="Whether this was computed from RAW_DATA or taken from REFERENCE_STATISTICS."
    )
    verification_status: VerificationStatus = Field(
        default=VerificationStatus.VERIFIED_EXTERNAL,
        description="Integrity classification of the statistic."
    )

    @model_validator(mode="after")
    def _validate_externally_reported(self) -> ReferenceStatistic:
        """Ensure external stats are properly documented."""
        if self.externally_reported:
            if not self.citation or not self.citation.strip():
                raise ValueError("Externally reported statistics require a citation.")

        if self.verification_status == VerificationStatus.UNVERIFIED_ESTIMATE:
            if not self.externally_reported:
                raise ValueError("UNVERIFIED_ESTIMATE must be externally_reported=True.")
            
            # Enforce derivation_notes message
            required_note = "source reports/estimates this value; independently derived from raw data: no."
            if not self.derivation_notes or required_note not in self.derivation_notes:
                raise ValueError(f"UNVERIFIED_ESTIMATE requires derivation_notes to include: '{required_note}'")

        return self
