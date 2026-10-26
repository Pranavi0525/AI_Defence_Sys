import pytest
import pandas as pd
import numpy as np
from src.red_team.ml.proxy_addendum import compute_metrics, entropy_purity, cv_amount

def test_metrics_on_synthetic_data():
    df = pd.DataFrame({
        'TransactionID': [1, 2, 3, 4, 5, 6, 7, 8],
        'isFraud': [0, 0, 0, 0, 0, 0, 0, 0],
        'proxy': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'C'],
        'ProductCD': ['W', 'W', 'C', 'H', 'H', 'W', 'W', 'W'],
        'TransactionAmt': [100.0, 100.0, 200.0, 50.0, 50.0, 10.0, 10.0, 10.0],
        'DeviceType': ['desktop', 'desktop', 'mobile', 'mobile', 'mobile', 'd', 'd', 'd'],
        'DeviceInfo': ['dev1', 'dev2', 'dev3', 'dev4', 'dev4', 'dev1', 'dev1', 'dev1'],
        'TransactionDT': [0, 86400, 86400*2, 0, 86400, 0, 86400, 172800]
    })
    
    # We add an attribute for raw rows to mock lineage
    df.attrs['total_raw_rows'] = 10
    
    metrics = compute_metrics(df)
    
    # Recurrence
    assert metrics['recurrence']['total_entities'] == 3
    assert metrics['recurrence']['total_rows'] == 8
    assert metrics['recurrence']['max'] == 3
    
    # Baseline expected null exists
    assert 'product_cd_dist' in metrics['global_baselines']
    assert 'global_amt_cv' in metrics['global_baselines']
    
    bucket_2_5 = metrics['bucket_stats']['2-5']
    assert bucket_2_5['entity_count'] == 3
    assert 'expected_null' in bucket_2_5['purity']
    assert 'p10' in bucket_2_5['purity']
    assert 'p90' in bucket_2_5['amt_cv']
    
def test_device_stability():
    from src.red_team.ml.proxy_addendum import top_3_device_coverage
    
    # 5 devices total, top 3 should cover 4/5 = 80%
    series = pd.Series(['d1', 'd2', 'd3', 'd4', 'd1'])
    coverage = top_3_device_coverage(series)
    assert np.isclose(coverage, 0.8)
    
    # Missing devices are treated as a single category ('UNKNOWN')
    series2 = pd.Series([np.nan, np.nan, 'd1', 'd2'])
    coverage2 = top_3_device_coverage(series2)
    assert np.isclose(coverage2, 1.0) # UNKNOWN(2), d1(1), d2(1) -> top 3 cover all 4 = 1.0

def test_not_available_handling():
    # If TransactionAmt is completely missing or 0 mean
    df = pd.DataFrame({
        'TransactionID': [1, 2],
        'isFraud': [0, 0],
        'proxy': ['A', 'A'],
        'ProductCD': ['W', 'W'],
        'TransactionAmt': [0.0, 0.0],
        'DeviceType': ['desktop', 'desktop'],
        'DeviceInfo': ['dev1', 'dev2'],
        'TransactionDT': [0, 86400]
    })
    
    metrics = compute_metrics(df)
    assert metrics['bucket_stats']['2-5']['amt_cv']['mean'] == 'NOT_AVAILABLE'
