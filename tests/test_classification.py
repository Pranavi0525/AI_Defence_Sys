import pytest
import numpy as np

def test_classify_entity():
    from src.red_team.ml.sequence_dataset_v2 import classify_entity
    
    # CONFIRMED_DIFFUSE: >= 50% match rate, < 60% top3
    assert classify_entity(0.60, 0.55) == "CONFIRMED_DIFFUSE"
    
    # CONFIRMED_STABLE: >= 20% match rate, >= 80% top3
    assert classify_entity(0.25, 0.85) == "CONFIRMED_STABLE"
    assert classify_entity(1.0, 1.0) == "CONFIRMED_STABLE"
    
    # INSUFFICIENT_EVIDENCE: < 20% match rate or NaNs
    assert classify_entity(0.15, 0.90) == "INSUFFICIENT_EVIDENCE"
    assert classify_entity(np.nan, 0.50) == "INSUFFICIENT_EVIDENCE"
    assert classify_entity(0.50, np.nan) == "INSUFFICIENT_EVIDENCE"
    
    # AMBIGUOUS: e.g. match rate >= 50% but top3 between 60% and 80%
    assert classify_entity(0.60, 0.70) == "AMBIGUOUS"
    
    # AMBIGUOUS: e.g. match rate between 20-50% and top3 < 80%
    assert classify_entity(0.30, 0.50) == "AMBIGUOUS"
