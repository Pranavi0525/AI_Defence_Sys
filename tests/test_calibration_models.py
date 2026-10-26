"""Tests for Stage 2.5 — Calibration Metric Models."""

from typing import Any

import pytest
from pydantic import ValidationError

from red_team.schemas.calibration import (
    CalibrationDefinition,
    FeaturePairCalibration,
    FeatureType,
    MarginalCalibrationConfig,
    MetricType,
    ThresholdDirection,
)


def test_feature_type_values():
    assert FeatureType.NUMERICAL == "NUMERICAL"
    assert FeatureType.CATEGORICAL == "CATEGORICAL"
    assert FeatureType.TEMPORAL == "TEMPORAL"
    assert FeatureType.BOOLEAN == "BOOLEAN"


def test_metric_type_values():
    assert MetricType.KS_STATISTIC == "KS_STATISTIC"
    assert MetricType.WASSERSTEIN == "WASSERSTEIN"
    assert MetricType.CATEGORICAL_FREQUENCY == "CATEGORICAL_FREQUENCY"
    assert MetricType.JENSEN_SHANNON == "JENSEN_SHANNON"
    assert MetricType.PEARSON == "PEARSON"
    assert MetricType.SPEARMAN == "SPEARMAN"
    assert MetricType.CRAMERS_V == "CRAMERS_V"
    assert MetricType.MUTUAL_INFORMATION == "MUTUAL_INFORMATION"


class TestMarginalCalibrationConfig:
    @pytest.mark.parametrize(
        "ftype, metric",
        [
            (FeatureType.NUMERICAL, MetricType.KS_STATISTIC),
            (FeatureType.NUMERICAL, MetricType.WASSERSTEIN),
            (FeatureType.CATEGORICAL, MetricType.CATEGORICAL_FREQUENCY),
            (FeatureType.CATEGORICAL, MetricType.JENSEN_SHANNON),
            (FeatureType.BOOLEAN, MetricType.CATEGORICAL_FREQUENCY),
            (FeatureType.BOOLEAN, MetricType.JENSEN_SHANNON),
        ],
    )
    def test_valid_combinations(self, ftype: FeatureType, metric: MetricType):
        config = MarginalCalibrationConfig(
            feature_name="test_feature",
            feature_type=ftype,
            metric=metric,
            threshold=0.05,
            threshold_direction=ThresholdDirection.MAXIMUM_ALLOWED,
        )
        assert config.feature_type == ftype
        assert config.metric == metric

    @pytest.mark.parametrize(
        "ftype, metric",
        [
            (FeatureType.NUMERICAL, MetricType.CRAMERS_V),
            (FeatureType.CATEGORICAL, MetricType.KS_STATISTIC),
            (FeatureType.TEMPORAL, MetricType.KS_STATISTIC),  # temporal has no marginal metrics defined yet
            (FeatureType.BOOLEAN, MetricType.PEARSON),
        ],
    )
    def test_invalid_combinations(self, ftype: FeatureType, metric: MetricType):
        with pytest.raises(ValidationError, match="not compatible"):
            MarginalCalibrationConfig(
                feature_name="test_feature",
                feature_type=ftype,
                metric=metric,
            )

    def test_threshold_must_be_positive(self):
        with pytest.raises(ValidationError, match="greater_than"):
            MarginalCalibrationConfig(
                feature_name="f1",
                feature_type=FeatureType.NUMERICAL,
                metric=MetricType.KS_STATISTIC,
                threshold=0.0,
            )

    def test_serialization_round_trip(self):
        config = MarginalCalibrationConfig(
            feature_name="f1",
            feature_type=FeatureType.NUMERICAL,
            metric=MetricType.KS_STATISTIC,
            threshold=0.1,
            threshold_direction=ThresholdDirection.MAXIMUM_ALLOWED,
        )
        data = config.model_dump()
        config2 = MarginalCalibrationConfig.model_validate(data)
        assert config == config2


class TestFeaturePairCalibration:
    @pytest.mark.parametrize(
        "ftypes, metric",
        [
            ((FeatureType.NUMERICAL, FeatureType.NUMERICAL), MetricType.PEARSON),
            ((FeatureType.NUMERICAL, FeatureType.NUMERICAL), MetricType.SPEARMAN),
            ((FeatureType.NUMERICAL, FeatureType.NUMERICAL), MetricType.MUTUAL_INFORMATION),
            ((FeatureType.CATEGORICAL, FeatureType.CATEGORICAL), MetricType.CRAMERS_V),
            ((FeatureType.CATEGORICAL, FeatureType.CATEGORICAL), MetricType.MUTUAL_INFORMATION),
            ((FeatureType.BOOLEAN, FeatureType.BOOLEAN), MetricType.CRAMERS_V),
            ((FeatureType.NUMERICAL, FeatureType.CATEGORICAL), MetricType.MUTUAL_INFORMATION),
            ((FeatureType.CATEGORICAL, FeatureType.NUMERICAL), MetricType.MUTUAL_INFORMATION),
            ((FeatureType.NUMERICAL, FeatureType.BOOLEAN), MetricType.MUTUAL_INFORMATION),
        ],
    )
    def test_valid_combinations(self, ftypes: tuple[FeatureType, FeatureType], metric: MetricType):
        config = FeaturePairCalibration(
            feature_a="f1",
            feature_b="f2",
            feature_types=ftypes,
            metric=metric,
            reason="Important relationship",
        )
        assert config.feature_types == ftypes
        assert config.metric == metric

    @pytest.mark.parametrize(
        "ftypes, metric",
        [
            ((FeatureType.NUMERICAL, FeatureType.NUMERICAL), MetricType.CRAMERS_V),
            ((FeatureType.CATEGORICAL, FeatureType.CATEGORICAL), MetricType.PEARSON),
            ((FeatureType.NUMERICAL, FeatureType.CATEGORICAL), MetricType.PEARSON),
            ((FeatureType.TEMPORAL, FeatureType.NUMERICAL), MetricType.MUTUAL_INFORMATION),
        ],
    )
    def test_invalid_combinations(self, ftypes: tuple[FeatureType, FeatureType], metric: MetricType):
        with pytest.raises(ValidationError, match="not compatible"):
            FeaturePairCalibration(
                feature_a="f1",
                feature_b="f2",
                feature_types=ftypes,
                metric=metric,
                reason="Testing invalid",
            )

    def test_missing_reason_rejected(self):
        with pytest.raises(ValidationError, match="reason"):
            FeaturePairCalibration(
                feature_a="f1",
                feature_b="f2",
                feature_types=(FeatureType.NUMERICAL, FeatureType.NUMERICAL),
                metric=MetricType.PEARSON,
                reason="",
            )

    def test_self_pair_rejected(self):
        with pytest.raises(ValidationError, match="self-referential"):
            FeaturePairCalibration(
                feature_a="f1",
                feature_b="f1",
                feature_types=(FeatureType.NUMERICAL, FeatureType.NUMERICAL),
                metric=MetricType.PEARSON,
                reason="Self check",
            )

    def test_canonical_pair_name(self):
        c1 = FeaturePairCalibration(
            feature_a="apple", feature_b="banana",
            feature_types=(FeatureType.NUMERICAL, FeatureType.NUMERICAL),
            metric=MetricType.PEARSON, reason="R"
        )
        c2 = FeaturePairCalibration(
            feature_a="banana", feature_b="apple",
            feature_types=(FeatureType.NUMERICAL, FeatureType.NUMERICAL),
            metric=MetricType.PEARSON, reason="R"
        )
        assert c1.canonical_pair_name == "apple::banana"
        assert c2.canonical_pair_name == "apple::banana"
        assert c1.canonical_pair_name == c2.canonical_pair_name

    def test_serialization_round_trip(self):
        config = FeaturePairCalibration(
            feature_a="f1",
            feature_b="f2",
            feature_types=(FeatureType.NUMERICAL, FeatureType.NUMERICAL),
            metric=MetricType.PEARSON,
            threshold=0.1,
            threshold_direction=ThresholdDirection.MAXIMUM_ALLOWED,
            reason="Important",
        )
        data = config.model_dump()
        config2 = FeaturePairCalibration.model_validate(data)
        assert config == config2


class TestCalibrationDefinition:
    def test_valid_definition(self):
        defn = CalibrationDefinition(
            marginal_configs=[
                MarginalCalibrationConfig(
                    feature_name="amount",
                    feature_type=FeatureType.NUMERICAL,
                    metric=MetricType.KS_STATISTIC,
                ),
            ],
            pair_configs=[
                FeaturePairCalibration(
                    feature_a="amount",
                    feature_b="velocity",
                    feature_types=(FeatureType.NUMERICAL, FeatureType.NUMERICAL),
                    metric=MetricType.PEARSON,
                    reason="Fraud relation",
                )
            ]
        )
        assert len(defn.marginal_configs) == 1
        assert len(defn.pair_configs) == 1

    def test_duplicate_marginal_names_rejected(self):
        with pytest.raises(ValidationError, match="Duplicate marginal"):
            CalibrationDefinition(
                marginal_configs=[
                    MarginalCalibrationConfig(
                        feature_name="amount",
                        feature_type=FeatureType.NUMERICAL,
                        metric=MetricType.KS_STATISTIC,
                    ),
                    MarginalCalibrationConfig(
                        feature_name="amount",  # duplicate
                        feature_type=FeatureType.NUMERICAL,
                        metric=MetricType.WASSERSTEIN,
                    ),
                ]
            )

    def test_duplicate_pair_definitions_rejected(self):
        with pytest.raises(ValidationError, match="Duplicate feature pair"):
            CalibrationDefinition(
                pair_configs=[
                    FeaturePairCalibration(
                        feature_a="f1",
                        feature_b="f2",
                        feature_types=(FeatureType.NUMERICAL, FeatureType.NUMERICAL),
                        metric=MetricType.PEARSON,
                        reason="R1",
                    ),
                    FeaturePairCalibration(
                        feature_a="f2",  # Reversed order still triggers canonical match
                        feature_b="f1",
                        feature_types=(FeatureType.NUMERICAL, FeatureType.NUMERICAL),
                        metric=MetricType.SPEARMAN,
                        reason="R2",
                    ),
                ]
            )

    def test_invalid_nested_models_rejected(self):
        with pytest.raises(ValidationError):
            CalibrationDefinition(
                marginal_configs=[
                    {"invalid": "data"}  # type: ignore
                ]
            )

    def test_serialization_round_trip(self):
        defn = CalibrationDefinition(
            marginal_configs=[
                MarginalCalibrationConfig(
                    feature_name="amount",
                    feature_type=FeatureType.NUMERICAL,
                    metric=MetricType.KS_STATISTIC,
                ),
            ],
            pair_configs=[
                FeaturePairCalibration(
                    feature_a="amount",
                    feature_b="velocity",
                    feature_types=(FeatureType.NUMERICAL, FeatureType.NUMERICAL),
                    metric=MetricType.PEARSON,
                    reason="Fraud relation",
                )
            ]
        )
        data = defn.model_dump()
        defn2 = CalibrationDefinition.model_validate(data)
        assert defn == defn2
