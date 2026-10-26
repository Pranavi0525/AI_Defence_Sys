import pandas as pd
import numpy as np
import json
import os
import sys

def entropy_purity(series):
    vc = series.value_counts(normalize=True, dropna=False)
    if len(vc) <= 1:
        return 1.0
    return float(vc.max())

def cv_amount(series):
    if series.empty or series.isna().all():
        return "NOT_AVAILABLE"
    mean = series.mean()
    if mean == 0 or pd.isna(mean):
        return "NOT_AVAILABLE"
    return float(series.std() / mean)

def top_3_device_coverage(series):
    # series is DeviceInfo. We fillna so we don't drop rows.
    # actually, let's only count actual devices if possible, or treat NaN as a single unknown.
    # The simplest is treating NaN as 'UNKNOWN'.
    vc = series.fillna('UNKNOWN').value_counts(normalize=True)
    return float(vc.head(3).sum())

def compute_metrics(df):
    """Computes all required metrics for the addendum report."""
    
    # 6. Explicit independence logging
    print("--- COMPUTATION INDEPENDENCE LOG ---")
    print("Recurrence statistic is computed FRESH in this run.")
    print("Data Lineage:")
    print(f"  Input rows before filtering: {df.attrs.get('total_raw_rows', 'N/A')}")
    print(f"  Rows after filtering (isFraud==0): {len(df)}")
    
    # 1. Recurrence
    counts = df['proxy'].value_counts()
    print(f"  Entity count: {len(counts)}")
    print("------------------------------------")
    
    recurrence = {
        'median': float(counts.median()),
        'mean': float(counts.mean()),
        'std': float(counts.std()),
        'p90': float(counts.quantile(0.90)),
        'p95': float(counts.quantile(0.95)),
        'p99': float(counts.quantile(0.99)),
        'p99.9': float(counts.quantile(0.999)),
        'max': int(counts.max()),
        'total_entities': len(counts),
        'total_rows': int(counts.sum())
    }
    
    # Global baselines
    global_prod_dist = df['ProductCD'].value_counts(normalize=True)
    global_amt_cv = cv_amount(df['TransactionAmt'])
    
    median_count = recurrence['median']
    
    # Filter entities above median
    # We will compute signals only for entities with size > 1
    recur_df = df[df['proxy'].isin(counts[counts > 1].index)]
    gb = recur_df.groupby('proxy')
    
    # Compute basic stats
    entity_stats = gb.agg(
        size=('TransactionID', 'count'),
        prod_purity=('ProductCD', entropy_purity),
        amt_cv=('TransactionAmt', cv_amount),
        unique_device_types=('DeviceType', 'nunique'),
        unique_device_infos=('DeviceInfo', 'nunique'),
        top3_dev_coverage=('DeviceInfo', top_3_device_coverage),
        min_dt=('TransactionDT', 'min'),
        max_dt=('TransactionDT', 'max')
    )
    
    # Time span in days
    entity_stats['time_span_days'] = (entity_stats['max_dt'] - entity_stats['min_dt']) / 86400.0
    entity_stats['avg_gap_days'] = entity_stats['time_span_days'] / (entity_stats['size'] - 1)
    
    # Convert amt_cv to float or NaN for easy quantile computation
    entity_stats['amt_cv_numeric'] = pd.to_numeric(entity_stats['amt_cv'], errors='coerce')
    
    # Size buckets
    bins = [1, 5, 20, 100, 1000, 100000]
    labels = ['2-5', '6-20', '21-100', '101-1000', '1000+']
    entity_stats['size_bucket'] = pd.cut(entity_stats['size'], bins=bins, labels=labels)
    
    bucket_stats = {}
    
    # For Monte Carlo Expected Purity
    prod_classes = global_prod_dist.index.values
    prod_probs = global_prod_dist.values
    
    for bucket in labels:
        sub = entity_stats[entity_stats['size_bucket'] == bucket]
        if len(sub) == 0:
            continue
            
        # P10/P50/P90 for Purity
        purity_p10 = float(sub['prod_purity'].quantile(0.10))
        purity_p50 = float(sub['prod_purity'].median())
        purity_p90 = float(sub['prod_purity'].quantile(0.90))
        purity_mean = float(sub['prod_purity'].mean())
        
        # P10/P50/P90 for Amount CV
        valid_cvs = sub['amt_cv_numeric'].dropna()
        if len(valid_cvs) > 0:
            cv_p10 = float(valid_cvs.quantile(0.10))
            cv_p50 = float(valid_cvs.median())
            cv_p90 = float(valid_cvs.quantile(0.90))
            cv_mean = float(valid_cvs.mean())
        else:
            cv_p10 = cv_p50 = cv_p90 = cv_mean = "NOT_AVAILABLE"
            
        # Expected Purity via Monte Carlo
        # We sample N times with the average size of this bucket
        avg_size = int(round(sub['size'].mean()))
        n_sim = 1000
        simulated_purities = []
        for _ in range(n_sim):
            draws = np.random.choice(prod_classes, size=avg_size, p=prod_probs)
            _, counts_arr = np.unique(draws, return_counts=True)
            simulated_purities.append(counts_arr.max() / avg_size)
        expected_purity = float(np.mean(simulated_purities))
        
        bucket_stats[bucket] = {
            'entity_count': len(sub),
            'avg_size': avg_size,
            'purity': {
                'expected_null': expected_purity,
                'mean': purity_mean,
                'p10': purity_p10,
                'p50': purity_p50,
                'p90': purity_p90
            },
            'amt_cv': {
                'mean': cv_mean,
                'p10': cv_p10,
                'p50': cv_p50,
                'p90': cv_p90
            },
            'avg_unique_dev_infos': float(sub['unique_device_infos'].mean()),
            'avg_time_span_days': float(sub['time_span_days'].mean()),
            'avg_gap_days': float(sub['avg_gap_days'].mean())
        }
        
    # Mega-entity device stability
    mega_entities = entity_stats[entity_stats['size_bucket'] == '1000+']
    if len(mega_entities) > 0:
        device_stability = {
            'mean_top3_coverage': float(mega_entities['top3_dev_coverage'].mean()),
            'p10_top3_coverage': float(mega_entities['top3_dev_coverage'].quantile(0.10)),
            'p50_top3_coverage': float(mega_entities['top3_dev_coverage'].median()),
            'p90_top3_coverage': float(mega_entities['top3_dev_coverage'].quantile(0.90))
        }
    else:
        device_stability = "NOT_AVAILABLE"
    
    return {
        'global_baselines': {
            'product_cd_dist': global_prod_dist.to_dict(),
            'global_amt_cv': global_amt_cv
        },
        'recurrence': recurrence,
        'bucket_stats': bucket_stats,
        'mega_entity_device_stability': device_stability
    }

def main():
    print("Loading data...")
    cols = ['TransactionID', 'isFraud', 'TransactionDT', 'TransactionAmt', 'ProductCD',
            'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'P_emaildomain']
    
    try:
        tx_full = pd.read_csv('data/raw/ieee-fraud-detection/train_transaction.csv', usecols=cols)
        ident = pd.read_csv('data/raw/ieee-fraud-detection/train_identity.csv', usecols=['TransactionID', 'DeviceType', 'DeviceInfo'])
    except Exception as e:
        print("Data files not found. Are you running this from the correct dir?")
        sys.exit(1)
        
    df = tx_full.merge(ident, on='TransactionID', how='left')
    total_raw_rows = len(df)
    
    df = df[df['isFraud'] == 0].copy()
    
    # Store the pre-filtered count for the lineage log
    # We will pass this implicitly or just print it from compute_metrics.
    # Let's inject a dummy column to pass total_raw_rows if we want, or just let compute_metrics compute it.
    # Actually I'll just print it before calling.
    
    # Construct proxy
    proxy_cols = ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'P_emaildomain']
    for c in proxy_cols:
        df[c] = df[c].fillna('nan').astype(str)
        
    df['proxy'] = df[proxy_cols].agg('_'.join, axis=1)
    
    # Temporarily append total raw rows so compute_metrics can print it
    df.attrs['total_raw_rows'] = total_raw_rows
    
    print("Computing metrics...")
    metrics = compute_metrics(df)
    
    with open('scratch/stage_4_5E_addendum_revised.json', 'w') as f:
        # Convert NaN to string or None for JSON serialization
        def sanitize(obj):
            if isinstance(obj, float) and np.isnan(obj):
                return "NOT_AVAILABLE"
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [sanitize(v) for v in obj]
            return obj
            
        json.dump(sanitize(metrics), f, indent=2)
    print("Done.")

if __name__ == '__main__':
    main()
