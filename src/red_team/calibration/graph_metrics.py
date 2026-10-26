"""Graph calibration module."""

from typing import Any, Dict
from red_team.calibration import CalibrationResult, ReferenceMode
from red_team.world.graph import RelationshipGraph


def calibrate_graph_metrics(
    metric_name: str,
    graph: RelationshipGraph,
    reference_statistic: Any = None,
) -> CalibrationResult:
    """Run graph metric calibration."""
    
    # We can at least extract the synthetic graph statistics here
    stats = graph.get_statistics()
    synth_val = stats.get(metric_name, "NOT_CALCULATED")
    
    if reference_statistic is None:
        return CalibrationResult(
            feature_a="graph",
            metric=metric_name,
            reference_mode=ReferenceMode.REFERENCE_STATISTICS,
            synthetic_value=synth_val,
            value="NOT_AVAILABLE",
            notes="Reference statistic missing.",
        )
        
    return CalibrationResult(
        feature_a="graph",
        metric=metric_name,
        reference_mode=ReferenceMode.REFERENCE_STATISTICS,
        reference_value=reference_statistic,
        synthetic_value=synth_val,
        value="NOT_AVAILABLE",  # Distance measure would go here
        notes="Graph metric reference vs synthetic compared.",
    )
