import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
import json

def evaluate(y_true, y_pred, labels=['C', 'H', 'R', 'S', 'W']):
    return {
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
    }

def main():
    test = pd.read_csv('data/reference/ml_sequence_v2/test.csv')
    
    y_true = test['target_ProductCD']
    
    # Rule 1: Predict the previous_ProductCD. 
    # If it's UNKNOWN, predict 'W' (the global majority).
    y_pred_rule = test['previous_ProductCD'].replace('UNKNOWN', 'W')
    
    res = evaluate(y_true, y_pred_rule)
    
    print("NAIVE RULE (Predict previous_ProductCD):")
    print(json.dumps(res, indent=2))

if __name__ == '__main__':
    main()
