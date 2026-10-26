import pandas as pd
import numpy as np
import json
import os
import sys

def main():
    print("Loading IEEE-CIS...")
    tx = pd.read_csv('data/raw/ieee-fraud-detection/train_transaction.csv', 
                     usecols=['TransactionID', 'isFraud', 'TransactionDT', 'TransactionAmt', 'ProductCD',
                              'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'P_emaildomain'])
    # Load identity for cross-field inspection
    ident = pd.read_csv('data/raw/ieee-fraud-detection/train_identity.csv',
                        usecols=['TransactionID', 'DeviceType', 'DeviceInfo'])
    
    df = tx.merge(ident, on='TransactionID', how='left')
    
    # Filter only legitimate behavior
    df = df[df['isFraud'] == 0].copy()
    print(f"Legit rows: {len(df)}")
    
    # Normalize components
    for c in ['card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'P_emaildomain']:
        df[c] = df[c].fillna('nan').astype(str)
        
    df['card_composite'] = df['card1'] + '_' + df['card2'] + '_' + df['card3'] + '_' + df['card4'] + '_' + df['card5'] + '_' + df['card6']
    
    # Candidate Definitions
    candidates = {
        'current_proxy': df['card_composite'] + '_' + df['addr1'] + '_' + df['P_emaildomain'],
        'Candidate_A_CardOnly': df['card_composite'],
        'Candidate_B_CardAddr': df['card_composite'] + '_' + df['addr1'],
        'Candidate_C_CardEmail': df['card_composite'] + '_' + df['P_emaildomain'],
        'Candidate_D_CardAddrEmail': df['card_composite'] + '_' + df['addr1'] + '_' + df['P_emaildomain']
    }
    
    stats = {}
    
    def analyze_candidate(series, name):
        vc = series.value_counts()
        total_rows = int(vc.sum())
        n = len(vc)
        stats = {
            'unique_entities': n,
            'total_transactions': total_rows,
            'min': int(vc.min()),
            'max': int(vc.max()),
            'mean': float(vc.mean()),
            'median': float(vc.median()),
            'p90': float(vc.quantile(0.90)),
            'p95': float(vc.quantile(0.95)),
            'p99': float(vc.quantile(0.99)),
            'p99.5': float(vc.quantile(0.995)),
            'p99.9': float(vc.quantile(0.999)),
            'pct_ge_2': float((vc >= 2).mean()),
            'pct_ge_3': float((vc >= 3).mean()),
            'pct_ge_5': float((vc >= 5).mean()),
            'pct_ge_10': float((vc >= 10).mean()),
            'pct_ge_20': float((vc >= 20).mean()),
            'buckets': {
                '1': int((vc == 1).sum()),
                '2': int((vc == 2).sum()),
                '3-4': int(((vc >= 3) & (vc <= 4)).sum()),
                '5-9': int(((vc >= 5) & (vc <= 9)).sum()),
                '10-19': int(((vc >= 10) & (vc <= 19)).sum()),
                '20+': int((vc >= 20).sum())
            },
            'top_0.1%_share': float(vc.head(max(1, int(n * 0.001))).sum() / total_rows),
            'top_0.5%_share': float(vc.head(max(1, int(n * 0.005))).sum() / total_rows),
            'top_1%_share': float(vc.head(max(1, int(n * 0.01))).sum() / total_rows)
        }
        return stats

    # Analyze Candidates
    for cname, cseries in candidates.items():
        print(f"Analyzing {cname}...")
        df[cname] = cseries
        stats[cname] = analyze_candidate(cseries, cname)
        
    # Analyze raw components
    print("Analyzing components...")
    stats['components'] = {
        'card_composite': analyze_candidate(df['card_composite'], 'card_composite'),
        'addr1': analyze_candidate(df['addr1'], 'addr1'),
        'P_emaildomain': analyze_candidate(df['P_emaildomain'], 'P_emaildomain')
    }
    
    # Cross-field inspection for top 50 entities of Candidate D (current proxy)
    print("Cross-field inspection...")
    top_entities = df['Candidate_D_CardAddrEmail'].value_counts().head(50).index
    cross_field_findings = []
    
    for entity in top_entities:
        sub = df[df['Candidate_D_CardAddrEmail'] == entity]
        finding = {
            'entity': entity,
            'size': len(sub),
            'distinct_dt': sub['TransactionDT'].nunique(),
            'dt_range_days': (sub['TransactionDT'].max() - sub['TransactionDT'].min()) / 86400,
            'unique_devices': sub['DeviceInfo'].nunique(),
            'unique_dev_types': sub['DeviceType'].nunique(),
            'amount_mean': float(sub['TransactionAmt'].mean()),
            'amount_std': float(sub['TransactionAmt'].std()),
            'product_cds': sub['ProductCD'].value_counts().to_dict()
        }
        cross_field_findings.append(finding)
        
    stats['cross_field_top_50'] = cross_field_findings
    
    # Save output
    with open('scratch/proxy_design_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
        
if __name__ == '__main__':
    main()
