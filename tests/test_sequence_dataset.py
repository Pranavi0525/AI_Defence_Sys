"""Tests for sequence dataset generation."""
import pytest
import pandas as pd
import numpy as np

from red_team.ml.sequence_dataset import (
    generate_proxy_entity,
    get_recurrence_stats,
    build_features,
    split_and_impute
)

def test_proxy_entity_generation():
    df = pd.DataFrame({
        'card1': [1, 2], 'card2': [1.0, 2.0], 'card3': [1.0, 2.0], 
        'card4': ['v', 'm'], 'card5': [1.0, 2.0], 'card6': ['c', 'd'],
        'addr1': [100.0, np.nan], 'P_emaildomain': ['a.com', 'b.com']
    })
    res = generate_proxy_entity(df)
    assert len(res) == 2
    assert "1-1.0-1.0-v-1.0-c_100.0_a.com" in res.values

def test_recurrence_stats():
    df = pd.DataFrame({
        'proxy_entity': ['A', 'A', 'B', 'B', 'B', 'C']
    })
    stats = get_recurrence_stats(df)
    assert stats['unique_entities'] == 3
    assert stats['total_transactions'] == 6
    assert stats['buckets']['1'] == 1
    assert stats['buckets']['2'] == 1
    assert stats['buckets']['3-4'] == 1

def test_build_features_leakage_and_chronology():
    df = pd.DataFrame({
        'proxy_entity': ['A', 'A', 'A'],
        'TransactionID': [1, 3, 2],
        'TransactionDT': [100, 300, 200],
        'TransactionAmt': [10.0, 30.0, 20.0],
        'ProductCD': ['W', 'C', 'R'],
        'DeviceType': ['D1', 'D3', 'D2'],
        'DeviceInfo': ['I1', 'I3', 'I2']
    })
    features = build_features(df)
    # The output should only include transactions 2 and 3 because the first is discarded
    assert len(features) == 2
    
    # Chronological sort means it evaluates order: TID 1 -> TID 2 -> TID 3
    f1 = features.iloc[0] # this is TID 2 predicting
    assert f1['target_ProductCD'] == 'R'
    assert f1['previous_amount'] == 10.0
    assert f1['previous_ProductCD'] == 'W'
    assert f1['transactions_seen_so_far'] == 1.0
    
    f2 = features.iloc[1] # this is TID 3 predicting
    assert f2['target_ProductCD'] == 'C'
    assert f2['previous_amount'] == 20.0
    assert f2['transactions_seen_so_far'] == 2.0
    assert f2['amount_mean_so_far'] == 15.0

def test_entity_level_split():
    df = pd.DataFrame({
        'proxy_entity': ['A', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
        'previous_amount': [1.0] * 11,
        'time_since_previous_transaction': [1.0] * 11,
        'amount_mean_so_far': [1.0] * 11,
        'amount_median_so_far': [1.0] * 11,
        'amount_std_so_far': [1.0] * 11,
        'amount_min_so_far': [1.0] * 11,
        'amount_max_so_far': [1.0] * 11,
        'transactions_seen_so_far': [1.0] * 11,
        'elapsed_time_since_first_observed': [1.0] * 11,
        'previous_ProductCD': ['W'] * 11,
        'previous_DeviceType': ['D'] * 11,
        'previous_DeviceInfo': ['I'] * 11,
    })
    
    train, val, test, params = split_and_impute(df, seed=42)
    train_ents = set(train['proxy_entity'].unique())
    val_ents = set(val['proxy_entity'].unique())
    test_ents = set(test['proxy_entity'].unique())
    
    assert train_ents.isdisjoint(val_ents)
    assert train_ents.isdisjoint(test_ents)
    assert val_ents.isdisjoint(test_ents)
