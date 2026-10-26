import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score

CLASSES = ['C', 'H', 'R', 'S', 'W']

def evaluate(y_true, y_pred, labels=CLASSES):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'per_class_precision': precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0).tolist(),
        'per_class_recall': recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0).tolist(),
        'per_class_f1': f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0).tolist()
    }

def main():
    print("Loading data...")
    train = pd.read_csv('data/reference/ml_sequence/train.csv')
    val = pd.read_csv('data/reference/ml_sequence/validation.csv')
    test = pd.read_csv('data/reference/ml_sequence/test.csv')
    
    with open('data/reference/ml_sequence/feature_manifest.json') as f:
        manifest = json.load(f)
    features = [x['feature_name'] for x in manifest]
    
    # proxy distribution
    # Combine all to get global proxy sizes
    all_df = pd.concat([train, val, test])
    # The transactions_seen_so_far max + 1 gives the total transactions for an entity in the sequences?
    # Better: just use value_counts of proxy_entity in the sequences + 1 (since 1st tx is dropped)
    # Actually, the proxy size can just be computed from all_df
    counts = all_df['proxy_entity'].value_counts()
    
    median = counts.median()
    p90 = counts.quantile(0.90)
    p95 = counts.quantile(0.95)
    p99 = counts.quantile(0.99)
    p995 = counts.quantile(0.995)
    p999 = counts.quantile(0.999)
    maximum = counts.max()
    
    print(f"Median: {median}")
    print(f"P90: {p90}")
    print(f"P95: {p95}")
    print(f"P99: {p99}")
    print(f"P99.5: {p995}")
    print(f"P99.9: {p999}")
    print(f"Max: {maximum}")
    
    print(f"Above P99: {(counts > p99).sum()}")
    print(f"Above P99.5: {(counts > p995).sum()}")
    print(f"Above P99.9: {(counts > p999).sum()}")
    
    # Evaluate model
    model = joblib.load('models/normal_behavior/logistic_regression/model.joblib')
    test['pred'] = model.predict(test[features])
    
    # CONDITION A
    res_a = evaluate(test['target_ProductCD'], test['pred'])
    print(f"Condition A: Entities={test['proxy_entity'].nunique()}, Rows={len(test)}")
    print(f"Macro F1: {res_a['macro_f1']}")
    
    # CONDITION B - >P99 EXCLUDED
    test_counts = test['proxy_entity'].value_counts()
    # Wait, the percentiles should be based on the proxy_entity count globally or just in test? 
    # Usually global count.
    valid_entities_b = counts[counts <= p99].index
    test_b = test[test['proxy_entity'].isin(valid_entities_b)]
    res_b = evaluate(test_b['target_ProductCD'], test_b['pred'])
    print(f"Condition B: Entities={test_b['proxy_entity'].nunique()}, Rows={len(test_b)}, Removed={len(test)-len(test_b)}")
    print(f"Macro F1: {res_b['macro_f1']}, Change: {res_b['macro_f1'] - res_a['macro_f1']}")
    
    # CONDITION C - >P99.5 EXCLUDED
    valid_entities_c = counts[counts <= p995].index
    test_c = test[test['proxy_entity'].isin(valid_entities_c)]
    res_c = evaluate(test_c['target_ProductCD'], test_c['pred'])
    print(f"Condition C: Entities={test_c['proxy_entity'].nunique()}, Rows={len(test_c)}")
    print(f"Macro F1: {res_c['macro_f1']}, Change: {res_c['macro_f1'] - res_a['macro_f1']}")
    
    # CONDITION D - >P99.9 EXCLUDED
    valid_entities_d = counts[counts <= p999].index
    test_d = test[test['proxy_entity'].isin(valid_entities_d)]
    res_d = evaluate(test_d['target_ProductCD'], test_d['pred'])
    print(f"Condition D: Entities={test_d['proxy_entity'].nunique()}, Rows={len(test_d)}")
    print(f"Macro F1: {res_d['macro_f1']}, Change: {res_d['macro_f1'] - res_a['macro_f1']}")
    
    # Per-Class Sensitivity (Baseline vs Condition D)
    print("\nPer-class (Baseline vs D):")
    for i, cls in enumerate(CLASSES):
        base_rec = res_a['per_class_recall'][i]
        filt_rec = res_d['per_class_recall'][i]
        base_f1 = res_a['per_class_f1'][i]
        filt_f1 = res_d['per_class_f1'][i]
        print(f"{cls}: Rec={base_rec:.4f}->{filt_rec:.4f} ({filt_rec-base_rec:.4f}), F1={base_f1:.4f}->{filt_f1:.4f} ({filt_f1-base_f1:.4f})")
        
    # Split composition
    def get_row_share(df, counts, pct):
        top_n = int(len(counts) * pct)
        top_entities = counts.head(top_n).index
        return len(df[df['proxy_entity'].isin(top_entities)]) / len(df) * 100
        
    for name, df in [("TRAIN", train), ("VALIDATION", val), ("TEST", test)]:
        df_counts = df['proxy_entity'].value_counts()
        print(f"{name}:")
        print(f"  Top 0.1%: {get_row_share(df, df_counts, 0.001):.2f}%")
        print(f"  Top 0.5%: {get_row_share(df, df_counts, 0.005):.2f}%")
        print(f"  Top 1%:   {get_row_share(df, df_counts, 0.01):.2f}%")
        
    # History Depth on TEST
    h = test['transactions_seen_so_far']
    print("History Depth:")
    print("1:", (h == 1).sum())
    print("2:", (h == 2).sum())
    print("3-4:", ((h >= 3) & (h <= 4)).sum())
    print("5-9:", ((h >= 5) & (h <= 9)).sum())
    print("10-19:", ((h >= 10) & (h <= 19)).sum())
    print("20+:", (h >= 20).sum())
    
    # Full dump for copy-paste
    import os
    os.makedirs('data/reference/ml_sequence/sensitivity_results', exist_ok=True)
    with open('data/reference/ml_sequence/sensitivity_results/res_a.json', 'w') as f: json.dump(res_a, f)
    with open('data/reference/ml_sequence/sensitivity_results/res_b.json', 'w') as f: json.dump(res_b, f)
    with open('data/reference/ml_sequence/sensitivity_results/res_c.json', 'w') as f: json.dump(res_c, f)
    with open('data/reference/ml_sequence/sensitivity_results/res_d.json', 'w') as f: json.dump(res_d, f)
    
if __name__ == '__main__':
    main()
