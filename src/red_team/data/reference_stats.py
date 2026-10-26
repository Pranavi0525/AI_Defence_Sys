"""Handling of reference statistics storage and generation."""

import json
from pathlib import Path
from typing import Dict, Any, List

from red_team.schemas.provenance import ReferenceStatistic


def save_reference_statistics(
    dataset_id: str,
    calibration_mode: str,
    stats: List[ReferenceStatistic],
    output_dir: Path,
) -> Path:
    """Save a list of ReferenceStatistics to a JSON artifact."""
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / f"{dataset_id}_statistics.json"

    payload = {
        "dataset_id": dataset_id,
        "calibration_mode": calibration_mode,
        "statistics": [s.model_dump() for s in stats],
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return file_path


def load_reference_statistics(file_path: Path) -> List[ReferenceStatistic]:
    """Load ReferenceStatistics from a saved JSON artifact."""
    with open(file_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    return [ReferenceStatistic.model_validate(s) for s in payload.get("statistics", [])]
