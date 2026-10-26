"""Calibration schemas for Red Team AI.

Defines typed Pydantic models for marginal and pairwise feature calibration.
Enforces structural validity and metric/feature-type compatibility before
any statistical calculations are performed.

Design decisions:
    - FeatureType explicitly maps to the kinds of data handled.
    - MetricType maps to the supported statistical distances/correlations.
    - Strict validation ensures NUMERICAL features use numerical metrics
      (e.g., KS, Wasserstein) and CATEGORICAL features use categorical
      metrics (e.g., Jensen-Shannon).
    - Pairwise combinations require a documented reason to prevent
      calculating N^2 meaningless comparisons.
    - Threshold semantics are explicit (e.g., MAXIMUM_ALLOWED distance).
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FeatureType(str, Enum):
    """Supported feature data types."""

    NUMERICAL = "NUMERICAL"
    CATEGORICAL = "CATEGORICAL"
    TEMPORAL = "TEMPORAL"
    BOOLEAN = "BOOLEAN"


class MetricType(str, Enum):
    """Supported calibration metrics."""

    KS_STATISTIC = "KS_STATISTIC"
    WASSERSTEIN = "WASSERSTEIN"
    CATEGORICAL_FREQUENCY = "CATEGORICAL_FREQUENCY"
    JENSEN_SHANNON = "JENSEN_SHANNON"
    PEARSON = "PEARSON"
    SPEARMAN = "SPEARMAN"
    CRAMERS_V = "CRAMERS_V"
    MUTUAL_INFORMATION = "MUTUAL_INFORMATION"


class ThresholdDirection(str, Enum):
    """Semantics of the calibration threshold."""

    MAXIMUM_ALLOWED = "MAXIMUM_ALLOWED"  # Typically for distances (KS, Wasserstein)
    MINIMUM_REQUIRED = "MINIMUM_REQUIRED"  # Typically for similarity/correlation bounds


# ---------------------------------------------------------------------------
# Metric Compatibility Rules
# ---------------------------------------------------------------------------

MARGINAL_COMPATIBILITY: dict[FeatureType, set[MetricType]] = {
    FeatureType.NUMERICAL: {MetricType.KS_STATISTIC, MetricType.WASSERSTEIN},
    FeatureType.CATEGORICAL: {MetricType.CATEGORICAL_FREQUENCY, MetricType.JENSEN_SHANNON},
    FeatureType.BOOLEAN: {MetricType.CATEGORICAL_FREQUENCY, MetricType.JENSEN_SHANNON},
    FeatureType.TEMPORAL: set(),  # Temporal handled later by dedicated models
}

PAIRWISE_COMPATIBILITY: dict[tuple[FeatureType, FeatureType], set[MetricType]] = {
    # Num + Num
    (FeatureType.NUMERICAL, FeatureType.NUMERICAL): {
        MetricType.PEARSON, MetricType.SPEARMAN, MetricType.MUTUAL_INFORMATION
    },
    # Cat + Cat (and Boolean combinations)
    (FeatureType.CATEGORICAL, FeatureType.CATEGORICAL): {
        MetricType.CRAMERS_V, MetricType.MUTUAL_INFORMATION
    },
    (FeatureType.BOOLEAN, FeatureType.BOOLEAN): {
        MetricType.CRAMERS_V, MetricType.MUTUAL_INFORMATION
    },
    (FeatureType.CATEGORICAL, FeatureType.BOOLEAN): {
        MetricType.CRAMERS_V, MetricType.MUTUAL_INFORMATION
    },
    (FeatureType.BOOLEAN, FeatureType.CATEGORICAL): {
        MetricType.CRAMERS_V, MetricType.MUTUAL_INFORMATION
    },
    # Mixed Num + Cat/Bool
    (FeatureType.NUMERICAL, FeatureType.CATEGORICAL): {MetricType.MUTUAL_INFORMATION},
    (FeatureType.CATEGORICAL, FeatureType.NUMERICAL): {MetricType.MUTUAL_INFORMATION},
    (FeatureType.NUMERICAL, FeatureType.BOOLEAN): {MetricType.MUTUAL_INFORMATION},
    (FeatureType.BOOLEAN, FeatureType.NUMERICAL): {MetricType.MUTUAL_INFORMATION},
}


# ---------------------------------------------------------------------------
# Marginal Calibration Config
# ---------------------------------------------------------------------------

class MarginalCalibrationConfig(BaseModel):
    """Configuration for calibrating a single feature's distribution."""

    feature_name: str = Field(..., min_length=1, description="Feature to calibrate.")
    feature_type: FeatureType = Field(..., description="Data type of the feature.")
    metric: MetricType = Field(..., description="Statistical metric to evaluate.")
    threshold: float | None = Field(default=None, gt=0, description="Optional threshold > 0.")
    threshold_direction: ThresholdDirection = Field(
        default=ThresholdDirection.MAXIMUM_ALLOWED,
        description="How to interpret the threshold (e.g., maximum allowable distance).",
    )

    @model_validator(mode="after")
    def _validate_compatibility(self) -> MarginalCalibrationConfig:
        """Ensure the chosen metric is compatible with the feature type."""
        allowed = MARGINAL_COMPATIBILITY.get(self.feature_type, set())
        if self.metric not in allowed:
            raise ValueError(
                f"Metric {self.metric.value} is not compatible with "
                f"marginal feature type {self.feature_type.value}. "
                f"Allowed: {[m.value for m in allowed]}"
            )
        return self


# ---------------------------------------------------------------------------
# Feature Pair Calibration Config
# ---------------------------------------------------------------------------

class FeaturePairCalibration(BaseModel):
    """Configuration for calibrating the dependency between two features."""

    feature_a: str = Field(..., min_length=1, description="First feature name.")
    feature_b: str = Field(..., min_length=1, description="Second feature name.")
    feature_types: tuple[FeatureType, FeatureType] = Field(
        ..., description="Data types for (feature_a, feature_b)."
    )
    metric: MetricType = Field(..., description="Dependency metric to evaluate.")
    threshold: float | None = Field(default=None, gt=0, description="Optional threshold > 0.")
    threshold_direction: ThresholdDirection = Field(
        default=ThresholdDirection.MAXIMUM_ALLOWED,
        description="How to interpret the threshold (e.g., maximum absolute error in correlation).",
    )
    reason: str = Field(
        ..., min_length=1,
        description="Explicit reason for requiring this dependency check (prevents N^2 noise).",
    )

    @model_validator(mode="after")
    def _validate_pair(self) -> FeaturePairCalibration:
        """Ensure the pair is valid and metric is compatible."""
        if self.feature_a == self.feature_b:
            raise ValueError("Feature pairs cannot be self-referential.")

        allowed = PAIRWISE_COMPATIBILITY.get(self.feature_types, set())
        if self.metric not in allowed:
            types_str = f"({self.feature_types[0].value}, {self.feature_types[1].value})"
            raise ValueError(
                f"Metric {self.metric.value} is not compatible with "
                f"feature pair types {types_str}. Allowed: {[m.value for m in allowed]}"
            )
        return self

    @property
    def canonical_pair_name(self) -> str:
        """Returns a stable string identifier independent of ordering."""
        # Sort feature names to ensure (A,B) and (B,A) map to the same string
        a, b = sorted([self.feature_a, self.feature_b])
        return f"{a}::{b}"


# ---------------------------------------------------------------------------
# Calibration Definition
# ---------------------------------------------------------------------------

class CalibrationDefinition(BaseModel):
    """A complete calibration plan for a Normal World or dataset."""

    marginal_configs: list[MarginalCalibrationConfig] = Field(
        default_factory=list, description="All marginal feature calibrations."
    )
    pair_configs: list[FeaturePairCalibration] = Field(
        default_factory=list, description="All pairwise dependency calibrations."
    )

    @model_validator(mode="after")
    def _validate_uniqueness(self) -> CalibrationDefinition:
        """Ensure no duplicate marginals or pairs exist in the plan."""
        # Check marginal duplicates
        marginal_names = [c.feature_name for c in self.marginal_configs]
        if len(marginal_names) != len(set(marginal_names)):
            # Find the duplicate
            seen = set()
            for name in marginal_names:
                if name in seen:
                    raise ValueError(f"Duplicate marginal configuration for feature: {name}")
                seen.add(name)

        # Check pair duplicates
        pair_names = [c.canonical_pair_name for c in self.pair_configs]
        if len(pair_names) != len(set(pair_names)):
            seen = set()
            for name in pair_names:
                if name in seen:
                    # Parse canonical name back for clearer error message
                    a, b = name.split("::")
                    raise ValueError(f"Duplicate feature pair configuration for: ({a}, {b})")
                seen.add(name)

        return self
