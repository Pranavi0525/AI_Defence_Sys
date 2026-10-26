"""Stage 4.5F: Supervised Sequence Dataset Generation with Corrected Proxy.

Builds a dataset predicting next transaction category (ProductCD)
from historical transaction behavior using the anonymized card_addr_email proxy,
filtered by the empirically-derived device stability exclusion rule.
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

def classify_entity(match_rate, top3_cov):
    if pd.isna(match_rate) or pd.isna(top3_cov):
        return "INSUFFICIENT_EVIDENCE"
    if match_rate < 0.20:
        return "INSUFFICIENT_EVIDENCE"
    if match_rate >= 0.50 and top3_cov < 0.60:
        return "CONFIRMED_DIFFUSE"
    if match_rate >= 0.20 and top3_cov >= 0.80:
        return "CONFIRMED_STABLE"
    return "AMBIGUOUS"

def top_3_known_device_coverage(series):
    known = series.dropna()
    if len(known) == 0: return np.nan
    vc = known.value_counts(normalize=True)
    return float(vc.head(3).sum())

def generate_proxy_entity(df: pd.DataFrame) -> pd.Series:
    """Recreate card_addr_email exactly as in the addendum."""
    proxy_cols = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'P_emaildomain']
    temp_df = df[proxy_cols].copy()
    for c in proxy_cols:
        temp_df[c] = temp_df[c].fillna('nan').astype(str)
    proxy = temp_df.agg('_'.join, axis=1)
    return proxy

def load_data(tx_path: str, id_path: str) -> pd.DataFrame:
    """Load and join legit transactions."""
    df_tx = pd.read_csv(tx_path, usecols=TX_COLS)
    df_id = pd.read_csv(id_path, usecols=ID_COLS)
    
    # Filter legitimate
    df_tx = df_tx[df_tx['isFraud'] == 0].copy()
    
    # Join
    df = pd.merge(df_tx, df_id, on='TransactionID', how='left')
    
    # Create proxy
    df['proxy_entity'] = generate_proxy_entity(df)
    
    # Drop isFraud immediately to prevent leakage
    df.drop(columns=['isFraud'], inplace=True)
    
    return df

def apply_exclusion_rule(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Applies the Stage 4.5E Addendum exclusion rule."""
    counts = df['proxy_entity'].value_counts()
    
    # We only compute coverage for entities with size > 1
    recur_df = df[df['proxy_entity'].isin(counts[counts > 1].index)]
    
    gb = recur_df.groupby('proxy_entity')
    stats = gb.agg(
        size=('TransactionID', 'count'),
        device_match_rate=('DeviceInfo', lambda s: float(s.notna().mean())),
        top3_known_coverage=('DeviceInfo', top_3_known_device_coverage)
    )
    
    stats['classification'] = stats.apply(lambda r: classify_entity(r['device_match_rate'], r['top3_known_coverage']), axis=1)
    
    # Add back the single-transaction entities as INSUFFICIENT_EVIDENCE
    single_entities = counts[counts == 1].index
    single_stats = pd.DataFrame(index=single_entities)
    single_stats['size'] = 1
    single_stats['classification'] = 'INSUFFICIENT_EVIDENCE'
    
    all_stats = pd.concat([stats, single_stats])
    
    # Filter out CONFIRMED_DIFFUSE
    diffuse_entities = all_stats[all_stats['classification'] == 'CONFIRMED_DIFFUSE'].index
    df_filtered = df[~df['proxy_entity'].isin(diffuse_entities)].copy()
    
    log = {
        'total_rows_before': len(df),
        'total_entities_before': len(all_stats),
        'excluded': {
            'entities': int((all_stats['classification'] == 'CONFIRMED_DIFFUSE').sum()),
            'rows': int(all_stats[all_stats['classification'] == 'CONFIRMED_DIFFUSE']['size'].sum())
        },
        'retained_by_category': {
            'CONFIRMED_STABLE': {
                'entities': int((all_stats['classification'] == 'CONFIRMED_STABLE').sum()),
                'rows': int(all_stats[all_stats['classification'] == 'CONFIRMED_STABLE']['size'].sum())
            },
            'INSUFFICIENT_EVIDENCE': {
                'entities': int((all_stats['classification'] == 'INSUFFICIENT_EVIDENCE').sum()),
                'rows': int(all_stats[all_stats['classification'] == 'INSUFFICIENT_EVIDENCE']['size'].sum())
            },
            'AMBIGUOUS': {
                'entities': int((all_stats['classification'] == 'AMBIGUOUS').sum()),
                'rows': int(all_stats[all_stats['classification'] == 'AMBIGUOUS']['size'].sum())
            }
        },
        'total_rows_after': len(df_filtered)
    }
    return df_filtered, log

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct sequential features avoiding leakage."""
    # Sort chronologically, deterministic tie-break
    df = df.sort_values(['proxy_entity', 'TransactionDT', 'TransactionID']).reset_index(drop=True)
    
    g = df.groupby('proxy_entity')
    
    df['previous_amount'] = g['TransactionAmt'].shift(1)
    df['previous_ProductCD'] = g['ProductCD'].shift(1)
    df['previous_DeviceType'] = g['DeviceType'].shift(1)
    df['previous_DeviceInfo'] = g['DeviceInfo'].shift(1)
    
    df['time_since_previous_transaction'] = df['TransactionDT'] - g['TransactionDT'].shift(1)
    
    df['amount_mean_so_far'] = g['TransactionAmt'].apply(lambda x: x.shift(1).expanding().mean()).reset_index(level=0, drop=True)
    df['amount_median_so_far'] = g['TransactionAmt'].apply(lambda x: x.shift(1).expanding().median()).reset_index(level=0, drop=True)
    df['amount_std_so_far'] = g['TransactionAmt'].apply(lambda x: x.shift(1).expanding().std()).reset_index(level=0, drop=True)
    df['amount_min_so_far'] = g['TransactionAmt'].apply(lambda x: x.shift(1).expanding().min()).reset_index(level=0, drop=True)
    df['amount_max_so_far'] = g['TransactionAmt'].apply(lambda x: x.shift(1).expanding().max()).reset_index(level=0, drop=True)
    df['transactions_seen_so_far'] = g.cumcount()
    
    first_dt = g['TransactionDT'].transform('first')
    df['elapsed_time_since_first_observed'] = df['TransactionDT'] - first_dt
    
    seq_df = df[df['transactions_seen_so_far'] > 0].copy()
    seq_df['target_ProductCD'] = seq_df['ProductCD']
    
    return seq_df

def split_and_impute(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Split by proxy_entity (70/15/15) and impute based on train."""
    entities = df['proxy_entity'].unique()
    np.random.seed(42)
    np.random.shuffle(entities)
    
    n = len(entities)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    
    train_ents = set(entities[:train_end])
    val_ents = set(entities[train_end:val_end])
    test_ents = set(entities[val_end:])
    
    train_df = df[df['proxy_entity'].isin(train_ents)].copy()
    val_df = df[df['proxy_entity'].isin(val_ents)].copy()
    test_df = df[df['proxy_entity'].isin(test_ents)].copy()
    
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

def get_recurrence_stats(df: pd.DataFrame) -> Dict[str, Any]:
    counts = df['proxy_entity'].value_counts()
    return {'unique_entities': len(counts)}

def main():
    print("Loading data...")
    df = load_data('data/raw/ieee-fraud-detection/train_transaction.csv', 
                   'data/raw/ieee-fraud-detection/train_identity.csv')
                   
    print("Applying exclusion rule...")
    df_filtered, log = apply_exclusion_rule(df)
    
    print("\n--- EXCLUSION LOG ---")
    print(json.dumps(log, indent=2))
    print("---------------------\n")
    
    print("Calculating recurrence...")
    stats = get_recurrence_stats(df_filtered)
    
    print("Building features...")
    seq_df = build_features(df_filtered)
    
    print("Splitting and imputing...")
    train_df, val_df, test_df, impute_params = split_and_impute(seq_df)
    
    print("Saving...")
    out_dir = 'data/reference/ml_sequence_v2'
    os.makedirs(out_dir, exist_ok=True)
    
    features_to_save = [f for f in impute_params.keys()] + ['target_ProductCD', 'proxy_entity']
    
    train_df[features_to_save].to_csv(os.path.join(out_dir, 'train.csv'), index=False)
    val_df[features_to_save].to_csv(os.path.join(out_dir, 'validation.csv'), index=False)
    test_df[features_to_save].to_csv(os.path.join(out_dir, 'test.csv'), index=False)
    
    print("Done generating corrected dataset.")
    print(f"Train rows: {len(train_df)}, entities: {train_df['proxy_entity'].nunique()}")
    print(f"Val rows: {len(val_df)}, entities: {val_df['proxy_entity'].nunique()}")
    print(f"Test rows: {len(test_df)}, entities: {test_df['proxy_entity'].nunique()}")

if __name__ == '__main__':
    main()
