"""Tests for Stage 4 — Dataset Access & Preprocessing."""

import json
from pathlib import Path

from red_team.data.preprocessing import preprocess_paysim, preprocess_ieee_cis, run_all_preprocessing
from red_team.data.reference_stats import load_reference_statistics


def test_paysim_fallback_generation(tmp_path: Path):
    """Test that when raw data is missing, PaySim preprocessing uses fallback statistics."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    ref_dir = tmp_path / "reference"
    
    out_file = preprocess_paysim(raw_dir, ref_dir)
    assert out_file.exists()
    assert out_file.name == "paysim_kaggle_v1_statistics.json"
    
    # Verify the JSON structure
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert data["dataset_id"] == "paysim_kaggle_v1"
    assert data["calibration_mode"] == "REFERENCE_STATISTICS"
    assert len(data["statistics"]) == 3
    
    # Load through the proper reader
    stats = load_reference_statistics(out_file)
    assert len(stats) == 3
    
    # Verify strict provenance on the fallback
    for stat in stats:
        assert stat.externally_reported is True
        assert stat.source_dataset_id == "paysim_kaggle_v1"
        assert "Fallback Baseline" in stat.source


def test_ieee_cis_fallback_generation(tmp_path: Path):
    """Test that IEEE-CIS preprocessing uses fallback statistics when missing."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    ref_dir = tmp_path / "reference"
    
    out_file = preprocess_ieee_cis(raw_dir, ref_dir)
    assert out_file.exists()
    
    stats = load_reference_statistics(out_file)
    assert len(stats) == 2
    
    device_stat = next(s for s in stats if s.statistic_name == "device_type_frequencies")
    assert "desktop" in device_stat.value
    assert device_stat.externally_reported is True


def test_run_all_preprocessing(tmp_path: Path):
    """Test the orchestration function runs both pipelines."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    ref_dir = tmp_path / "reference"
    
    results = run_all_preprocessing(raw_dir, ref_dir)
    assert len(results) == 2
    
    assert (ref_dir / "paysim_kaggle_v1_statistics.json").exists()
    assert (ref_dir / "ieee_cis_fraud_v1_statistics.json").exists()
