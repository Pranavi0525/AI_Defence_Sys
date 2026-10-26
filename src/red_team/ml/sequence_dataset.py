"""Stage 4.5B: Supervised Sequence Dataset Generation.

Builds a dataset predicting next transaction category (ProductCD)
from historical transaction behavior using the anonymized card_addr_email proxy.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

# Restrict to these columns to save memory
TX_COLS = ['TransactionID', 'isFraud', 'TransactionDT', 'TransactionAmt', 'ProductCD',
           'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'P_emaildomain']
ID_COLS = ['TransactionID', 'DeviceType', 'DeviceInfo']

def generate_proxy_entity(df: pd.DataFrame) -> pd.Series:
    """Recreate exactly card_addr_email."""
    card_cols = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6']
    card_composite = df[card_cols].astype(str).agg('-'.join, axis=1)
    proxy = card_composite + '_' + df['addr1'].astype(str) + '_' + df['P_emaildomain'].astype(str)
    return proxy

def load_data(tx_path: str, id_path: str) -> pd.DataFrame:
    """Load and join legit transactions."""
    df_tx = pd.read_csv(tx_path, usecols=TX_COLS)
    df_id = pd.read_csv(id_path, usecols=ID_COLS)
    
    # Filter legitimate
    df_tx = df_tx[df_tx['isFraud'] == 0].copy()
    
    # Drop isFraud immediately to prevent leakage
    df_tx.drop(columns=['isFraud'], inplace=True)
    
    # Join
    df = pd.merge(df_tx, df_id, on='TransactionID', how='left')
    
    # Create proxy
    df['proxy_entity'] = generate_proxy_entity(df)
    
    return df

def get_recurrence_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate recurrence metrics."""
    counts = df['proxy_entity'].value_counts()
    
    buckets = {
        '1': (counts == 1).sum(),
        '2': (counts == 2).sum(),
        '3-4': ((counts >= 3) & (counts <= 4)).sum(),
        '5-9': ((counts >= 5) & (counts <= 9)).sum(),
        '10-19': ((counts >= 10) & (counts <= 19)).sum(),
        '20+': (counts >= 20).sum()
    }
    
    return {
        'unique_entities': len(counts),
        'total_transactions': counts.sum(),
        'min_tx': counts.min(),
        'mean_tx': counts.mean(),
        'median_tx': counts.median(),
        'max_tx': counts.max(),
        'ge_2': (counts >= 2).mean() * 100,
        'ge_3': (counts >= 3).mean() * 100,
        'ge_5': (counts >= 5).mean() * 100,
        'ge_10': (counts >= 10).mean() * 100,
        'ge_20': (counts >= 20).mean() * 100,
        'buckets': buckets
    }

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct sequential features avoiding leakage."""
    # Sort chronologically, deterministic tie-break
    df = df.sort_values(['proxy_entity', 'TransactionDT', 'TransactionID']).reset_index(drop=True)
    
    # We want to predict current ProductCD using ONLY previous rows of the same entity.
    # Group by entity
    g = df.groupby('proxy_entity')
    
    # 1. Target
    df['target_ProductCD'] = df['ProductCD']
    
    # 2. Previous transaction features (shifted by 1)
    df['previous_amount'] = g['TransactionAmt'].shift(1)
    df['previous_ProductCD'] = g['ProductCD'].shift(1)
    df['previous_TransactionDT'] = g['TransactionDT'].shift(1)
    df['time_since_previous_transaction'] = df['TransactionDT'] - df['previous_TransactionDT']
    
    # Contextual device info from previous transaction
    df['previous_DeviceType'] = g['DeviceType'].shift(1).fillna('UNKNOWN')
    df['previous_DeviceInfo'] = g['DeviceInfo'].shift(1).fillna('UNKNOWN')
    
    # 3. Historical aggregations
    # To avoid leakage, we use expanding window on shifted amount, OR shift the expanding window.
    # Expanding on TransactionAmt but shift(1) to avoid including current.
    # We can create a shifted amount column first.
    shifted_amt = g['TransactionAmt'].shift(1)
    
    # Custom aggregations via expanding
    exp = shifted_amt.groupby(df['proxy_entity']).expanding()
    
    # This aligns with the original index!
    df['amount_mean_so_far'] = exp.mean().reset_index(level=0, drop=True)
    df['amount_median_so_far'] = exp.median().reset_index(level=0, drop=True)
    # std requires >=2
    df['amount_std_so_far'] = exp.std().reset_index(level=0, drop=True)
    df['amount_min_so_far'] = exp.min().reset_index(level=0, drop=True)
    df['amount_max_so_far'] = exp.max().reset_index(level=0, drop=True)
    
    df['transactions_seen_so_far'] = exp.count().reset_index(level=0, drop=True)
    
    first_dt = g['TransactionDT'].transform('first')
    df['elapsed_time_since_first_observed'] = df['TransactionDT'] - first_dt
    
    # Filter out the first transaction per entity since it has no history (transactions_seen_so_far == 0 or NaN)
    # Actually count() gives 0 for all-NaN.
    # Let's just drop where transactions_seen_so_far == 0 or previous_amount is null
    seq_df = df[df['transactions_seen_so_far'] > 0].copy()
    
    # Fill remaining NaNs for std (if only 1 item, std is NaN)
    seq_df['amount_std_so_far'] = seq_df['amount_std_so_far'].fillna(0.0)
    
    return seq_df

def split_and_impute(df: pd.DataFrame, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Split by entity and fit imputation on train only."""
    np.random.seed(seed)
    entities = df['proxy_entity'].unique()
    np.random.shuffle(entities)
    
    n = len(entities)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)
    
    train_ents = set(entities[:train_end])
    val_ents = set(entities[train_end:val_end])
    test_ents = set(entities[val_end:])
    
    train_df = df[df['proxy_entity'].isin(train_ents)].copy()
    val_df = df[df['proxy_entity'].isin(val_ents)].copy()
    test_df = df[df['proxy_entity'].isin(test_ents)].copy()
    
    # Imputation fitting (only numeric needed, categorical already 'UNKNOWN' or has nan handling)
    # For previous_ProductCD, missing is naturally the first, but we dropped the first. So it shouldn't be missing.
    # Let's fill any remaining missing categorical
    cat_cols = ['previous_ProductCD', 'previous_DeviceType', 'previous_DeviceInfo']
    num_cols = ['previous_amount', 'time_since_previous_transaction', 'amount_mean_so_far', 
                'amount_median_so_far', 'amount_std_so_far', 'amount_min_so_far', 'amount_max_so_far',
                'transactions_seen_so_far', 'elapsed_time_since_first_observed']
                
    impute_params = {}
    for col in num_cols:
        val = train_df[col].mean()
        impute_params[col] = val
        train_df[col] = train_df[col].fillna(val)
        val_df[col] = val_df[col].fillna(val)
        test_df[col] = test_df[col].fillna(val)
        
    for col in cat_cols:
        train_df[col] = train_df[col].fillna('UNKNOWN')
        val_df[col] = val_df[col].fillna('UNKNOWN')
        test_df[col] = test_df[col].fillna('UNKNOWN')
        impute_params[col] = 'UNKNOWN'
        
    return train_df, val_df, test_df, impute_params

def write_manifest_and_metadata(impute_params: Dict[str, Any], stats: Dict[str, Any], 
                                train_shape: tuple, val_shape: tuple, test_shape: tuple,
                                out_dir: str):
    """Write required JSON metadata."""
    os.makedirs(out_dir, exist_ok=True)
    
    manifest = [
        {
            "feature_name": "previous_amount",
            "source_fields": ["TransactionAmt"],
            "raw_or_derived": "derived",
            "entity_scope": "card_addr_email",
            "available_before_target": True,
            "missing_value_strategy": f"impute_mean_{impute_params.get('previous_amount')}",
            "leakage_risk": "low",
            "provenance": "previous transaction amount"
        },
        {
            "feature_name": "time_since_previous_transaction",
            "source_fields": ["TransactionDT"],
            "raw_or_derived": "derived",
            "entity_scope": "card_addr_email",
            "available_before_target": True,
            "missing_value_strategy": f"impute_mean_{impute_params.get('time_since_previous_transaction')}",
            "leakage_risk": "low",
            "provenance": "delta from previous TransactionDT"
        },
        {
            "feature_name": "amount_mean_so_far",
            "source_fields": ["TransactionAmt"],
            "raw_or_derived": "derived",
            "entity_scope": "card_addr_email",
            "available_before_target": True,
            "missing_value_strategy": "impute_mean",
            "leakage_risk": "low",
            "provenance": "expanding mean of past transactions"
        },
        {
            "feature_name": "amount_median_so_far",
            "source_fields": ["TransactionAmt"],
            "raw_or_derived": "derived",
            "entity_scope": "card_addr_email",
            "available_before_target": True,
            "missing_value_strategy": "impute_mean",
            "leakage_risk": "low",
            "provenance": "expanding median of past transactions"
        },
        {
            "feature_name": "amount_std_so_far",
            "source_fields": ["TransactionAmt"],
            "raw_or_derived": "derived",
            "entity_scope": "card_addr_email",
            "available_before_target": True,
            "missing_value_strategy": "fill_0_then_mean",
            "leakage_risk": "low",
            "provenance": "expanding std of past transactions"
        },
        {
            "feature_name": "amount_min_so_far",
            "source_fields": ["TransactionAmt"],
            "raw_or_derived": "derived",
            "entity_scope": "card_addr_email",
            "available_before_target": True,
            "missing_value_strategy": "impute_mean",
            "leakage_risk": "low",
            "provenance": "expanding min of past transactions"
        },
        {
            "feature_name": "amount_max_so_far",
            "source_fields": ["TransactionAmt"],
            "raw_or_derived": "derived",
            "entity_scope": "card_addr_email",
            "available_before_target": True,
            "missing_value_strategy": "impute_mean",
            "leakage_risk": "low",
            "provenance": "expanding max of past transactions"
        },
        {
            "feature_name": "transactions_seen_so_far",
            "source_fields": ["TransactionID"],
            "raw_or_derived": "derived",
            "entity_scope": "card_addr_email",
            "available_before_target": True,
            "missing_value_strategy": "impute_mean",
            "leakage_risk": "low",
            "provenance": "expanding count of past transactions"
        },
        {
            "feature_name": "elapsed_time_since_first_observed",
            "source_fields": ["TransactionDT"],
            "raw_or_derived": "derived",
            "entity_scope": "card_addr_email",
            "available_before_target": True,
            "missing_value_strategy": "impute_mean",
            "leakage_risk": "low",
            "provenance": "current TransactionDT minus first TransactionDT (strictly causal since dt is monotonic)"
        },
        {
            "feature_name": "previous_ProductCD",
            "source_fields": ["ProductCD"],
            "raw_or_derived": "derived",
            "entity_scope": "card_addr_email",
            "available_before_target": True,
            "missing_value_strategy": "UNKNOWN",
            "leakage_risk": "low",
            "provenance": "ProductCD of the immediate previous transaction"
        },
        {
            "feature_name": "previous_DeviceType",
            "source_fields": ["DeviceType"],
            "raw_or_derived": "derived",
            "entity_scope": "card_addr_email",
            "available_before_target": True,
            "missing_value_strategy": "UNKNOWN",
            "leakage_risk": "low",
            "provenance": "DeviceType of the immediate previous transaction"
        },
        {
            "feature_name": "previous_DeviceInfo",
            "source_fields": ["DeviceInfo"],
            "raw_or_derived": "derived",
            "entity_scope": "card_addr_email",
            "available_before_target": True,
            "missing_value_strategy": "UNKNOWN",
            "leakage_risk": "low",
            "provenance": "DeviceInfo of the immediate previous transaction"
        }
    ]
    
    metadata = {
        "source_dataset": "IEEE-CIS Fraud Detection",
        "proxy_definition": "card1-card6 + addr1 + P_emaildomain",
        "legitimate_filter": "isFraud == 0",
        "sequence_definition": "predict next transaction category given >=1 past transactions",
        "target_definition": "next_transaction_category (ProductCD)",
        "feature_list": [f["feature_name"] for f in manifest],
        "split_seed": 42,
        "split_method": "proxy_entity grouped (70/15/15)",
        "preprocessing_version": "1.0",
        "row_counts": {
            "train": train_shape[0],
            "validation": val_shape[0],
            "test": test_shape[0]
        },
        "entity_counts": {
            "total_pre_sequence": stats['unique_entities']
        }
    }
    
    with open(os.path.join(out_dir, 'feature_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
        
    with open(os.path.join(out_dir, 'dataset_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

def main():
    print("Loading data...")
    df = load_data('data/raw/ieee-fraud-detection/train_transaction.csv', 
                   'data/raw/ieee-fraud-detection/train_identity.csv')
                   
    print("Calculating recurrence...")
    stats = get_recurrence_stats(df)
    
    print("Building features...")
    seq_df = build_features(df)
    
    print("Splitting and imputing...")
    train_df, val_df, test_df, impute_params = split_and_impute(seq_df)
    
    print("Saving...")
    out_dir = 'data/reference/ml_sequence'
    os.makedirs(out_dir, exist_ok=True)
    
    features_to_save = [f for f in impute_params.keys()] + ['target_ProductCD', 'proxy_entity']
    
    train_df[features_to_save].to_csv(os.path.join(out_dir, 'train.csv'), index=False)
    val_df[features_to_save].to_csv(os.path.join(out_dir, 'validation.csv'), index=False)
    test_df[features_to_save].to_csv(os.path.join(out_dir, 'test.csv'), index=False)
    
    write_manifest_and_metadata(impute_params, stats, train_df.shape, val_df.shape, test_df.shape, out_dir)
    
    print("Output statistics for report:")
    print(json.dumps(stats, indent=2))
    print(f"Train rows: {len(train_df)}, entities: {train_df['proxy_entity'].nunique()}")
    print(f"Val rows: {len(val_df)}, entities: {val_df['proxy_entity'].nunique()}")
    print(f"Test rows: {len(test_df)}, entities: {test_df['proxy_entity'].nunique()}")
    print("Target distribution TRAIN:\n", train_df['target_ProductCD'].value_counts().to_dict())
    print("Target distribution VAL:\n", val_df['target_ProductCD'].value_counts().to_dict())
    print("Target distribution TEST:\n", test_df['target_ProductCD'].value_counts().to_dict())

if __name__ == '__main__':
    main()
