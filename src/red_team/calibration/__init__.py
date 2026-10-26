"""Calibration Engine for Red Team AI.

Compares synthetic normal world data against reference statistics/data.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from red_team.schemas.calibration import MetricType


class ReferenceMode(str, Enum):
    """The mode of reference data available for calibration."""
    RAW_DATA = "RAW_DATA"
    REFERENCE_STATISTICS = "REFERENCE_STATISTICS"


class CalibrationResult(BaseModel):
    """Result of a single metric measurement."""
    feature_a: str = Field(..., description="Primary feature/metric name.")
    feature_b: Optional[str] = Field(default=None, description="Secondary feature for pairs.")
    metric: MetricType | str = Field(..., description="The metric being calculated.")
    reference_mode: ReferenceMode = Field(..., description="Mode used (raw or stats).")
    reference_source: Optional[str] = Field(default=None, description="Source of reference data/stat.")
    
    reference_value: Any = Field(default=None, description="The reference value/distribution.")
    synthetic_value: Any = Field(default=None, description="The synthetic value/distribution.")
    value: Any = Field(default=None, description="The calculated distance/metric value, or NOT_AVAILABLE.")
    
    threshold: Optional[float] = Field(default=None, description="Threshold from configuration.")
    pass_fail: str = Field(default="NOT_EVALUATED", description="PASS, FAIL, or NOT_EVALUATED.")
    
    interpretation: str = Field(default="", description="Human-readable interpretation of result.")
    notes: str = Field(default="", description="Additional context or missing-data notes.")


class CalibrationReport(BaseModel):
    """Aggregate report of all calibration metrics."""
    marginal_results: List[CalibrationResult] = Field(default_factory=list)
    dependency_results: List[CalibrationResult] = Field(default_factory=list)
    temporal_results: List[CalibrationResult] = Field(default_factory=list)
    behavioral_results: List[CalibrationResult] = Field(default_factory=list)
    graph_results: List[CalibrationResult] = Field(default_factory=list)
    structural_results: List[CalibrationResult] = Field(default_factory=list)
    
    overall_summary: str = Field(default="Calibration Complete", description="High-level summary.")

    def add_result(self, category: str, result: CalibrationResult) -> None:
        """Helper to append to correct list."""
        if category == "marginal":
            self.marginal_results.append(result)
        elif category == "dependency":
            self.dependency_results.append(result)
        elif category == "temporal":
            self.temporal_results.append(result)
        elif category == "behavioral":
            self.behavioral_results.append(result)
        elif category == "graph":
            self.graph_results.append(result)
        elif category == "structural":
            self.structural_results.append(result)


def _evaluate_threshold(value: Any, threshold: Optional[float], direction: str) -> str:
    """Helper to evaluate thresholds if numeric."""
    if value == "NOT_AVAILABLE" or value is None:
        return "NOT_EVALUATED"
    if threshold is None:
        return "NOT_EVALUATED"
        
    try:
        num_val = float(value)
        if direction == "MAXIMUM_ALLOWED":
            return "PASS" if num_val <= threshold else "FAIL"
        elif direction == "MINIMUM_REQUIRED":
            return "PASS" if num_val >= threshold else "FAIL"
    except (TypeError, ValueError):
        pass
    return "NOT_EVALUATED"

from red_team.calibration.marginal import calibrate_marginal
from red_team.calibration.dependency import calibrate_dependency
from red_team.calibration.temporal import calibrate_temporal
from red_team.calibration.behavioral import calibrate_behavioral
from red_team.calibration.graph_metrics import calibrate_graph_metrics
from red_team.calibration.structural import validate_structural

__all__ = [
    "ReferenceMode",
    "CalibrationResult",
    "CalibrationReport",
    "calibrate_marginal",
    "calibrate_dependency",
    "calibrate_temporal",
    "calibrate_behavioral",
    "calibrate_graph_metrics",
    "validate_structural",
]
