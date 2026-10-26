import pandas as pd
import numpy as np
from src.red_team.ml.sequence_dataset_v2 import apply_exclusion_rule, build_features

def test_apply_exclusion_rule():
    # Synthetic data to test exclusion rule
    # We need proxy_entity, DeviceInfo
    
    # Entity 1: CONFIRMED_STABLE (match >= 0.20, top3 >= 0.80) -> size 3, all devices are known and same
    # Entity 2: CONFIRMED_DIFFUSE (match >= 0.50, top3 < 0.60) -> size 5, 5 unique devices
    # Entity 3: AMBIGUOUS (match >= 0.50, top3 >= 0.60, < 0.80) -> size 5, 3 unique devices, top 3 = 100% so it's stable. Wait, top3 < 0.80.
    # Entity 4: INSUFFICIENT_EVIDENCE -> size 2, no devices
    
    df = pd.DataFrame({
        'TransactionID': range(1, 15),
        'proxy_entity': [
            'E1', 'E1', 'E1',
            'E2', 'E2', 'E2', 'E2', 'E2',
            'E3', 'E3', 'E3', 'E3',
            'E4', 'E4'
        ],
        'DeviceInfo': [
            'd1', 'd1', 'd1',                   # E1: match=1.0, top3=1.0 -> STABLE
            'd1', 'd2', 'd3', 'd4', 'd5',       # E2: match=1.0, top3=3/5=0.6 -> Wait, I need < 0.6 for diffuse. Let's add one more to E2.
            'd1', 'd2', 'd3', 'd4',             # E3: match=1.0, top3=3/4=0.75 -> AMBIGUOUS
            np.nan, np.nan                      # E4: match=0.0 -> INSUFFICIENT
        ]
    })
    
    # Fix E2 to be < 0.60
    # 6 devices, all unique -> top 3 = 3/6 = 0.50
    df = pd.DataFrame({
        'TransactionID': range(1, 16),
        'proxy_entity': [
            'E1', 'E1', 'E1',
            'E2', 'E2', 'E2', 'E2', 'E2', 'E2',
            'E3', 'E3', 'E3', 'E3',
            'E4', 'E4'
        ],
        'DeviceInfo': [
            'd1', 'd1', 'd1',                         # E1: STABLE
            'd1', 'd2', 'd3', 'd4', 'd5', 'd6',       # E2: DIFFUSE (match=1.0, top3=0.5)
            'd1', 'd2', 'd3', 'd4',                   # E3: AMBIGUOUS (match=1.0, top3=0.75)
            np.nan, np.nan                            # E4: INSUFFICIENT
        ]
    })
    
    filtered_df, log = apply_exclusion_rule(df)
    
    # E2 should be excluded
    assert 'E2' not in filtered_df['proxy_entity'].values
    assert 'E1' in filtered_df['proxy_entity'].values
    assert 'E3' in filtered_df['proxy_entity'].values
    assert 'E4' in filtered_df['proxy_entity'].values
    
    assert log['excluded']['entities'] == 1
    assert log['excluded']['rows'] == 6
    assert log['retained_by_category']['CONFIRMED_STABLE']['entities'] == 1
    assert log['retained_by_category']['AMBIGUOUS']['entities'] == 1
    assert log['retained_by_category']['INSUFFICIENT_EVIDENCE']['entities'] == 1

