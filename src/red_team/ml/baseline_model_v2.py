"""Stage 4.5F: Train Corrected Logistic Regression Model.

Retrains the identical LogisticRegression configuration on the
filtered, corrected ml_sequence_v2 dataset.
"""

import os
import json
import time
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, 
    recall_score, f1_score, confusion_matrix
)

# Forbidden columns
FORBIDDEN = {'proxy_entity', 'card_addr_email', 'TransactionID', 'isFraud', 'target_ProductCD', 'TransactionDT', 'TransactionAmt'}
CLASSES = ['C', 'H', 'R', 'S', 'W']

def evaluate(y_true, y_pred, labels=CLASSES) -> Dict[str, Any]:
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        'per_class_precision': precision_score(y_true, y_pred, average=None, labels=labels, zero_division=0).tolist(),
        'per_class_recall': recall_score(y_true, y_pred, average=None, labels=labels, zero_division=0).tolist(),
        'per_class_f1': f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0).tolist()
    }

def main():
    print("Loading data...")
    train = pd.read_csv('data/reference/ml_sequence_v2/train.csv')
    val = pd.read_csv('data/reference/ml_sequence_v2/validation.csv')
    test = pd.read_csv('data/reference/ml_sequence_v2/test.csv')
    
    with open('data/reference/ml_sequence_v2/feature_manifest.json') as f:
        manifest = json.load(f)
        
    features = [x['feature_name'] for x in manifest]
    
    # 3. FEATURE CONSISTENCY
    for f_col in features:
        assert f_col in train.columns and f_col in val.columns and f_col in test.columns, f"Missing feature {f_col}"
    assert 'target_ProductCD' in train.columns
    
    # Identify X and y
    X_train = train[features]
    y_train = train['target_ProductCD']
    
    X_val = val[features]
    y_val = val['target_ProductCD']
    
    X_test = test[features]
    y_test = test['target_ProductCD']
    
    # Verify no leakage
    for f in features:
        if f in FORBIDDEN:
            raise ValueError(f"Leakage detected: {f} is forbidden.")
            
    # Verify disjoint splits
    train_ents = set(train['proxy_entity'])
    val_ents = set(val['proxy_entity'])
    test_ents = set(test['proxy_entity'])
    assert train_ents.isdisjoint(val_ents)
    assert train_ents.isdisjoint(test_ents)
    assert val_ents.isdisjoint(test_ents)
    
    # 4. MAJORITY BASELINE
    print("Majority Baseline...")
    majority_class = y_train.mode()[0]
    
    y_test_maj = [majority_class] * len(y_test)
    test_maj_res = evaluate(y_test, y_test_maj)
    
    # 5. ML BASELINE: Logistic Regression Preprocessing
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), num_cols),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
            ('ohe', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_cols)
    ])
    
    print("Training Logistic Regression model...")
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            C=0.1, class_weight=None,
            solver='lbfgs', max_iter=1000, random_state=42
        ))
    ])
    pipe.fit(X_train, y_train)
    
    print("\nEvaluating model on TEST...")
    y_test_pred = pipe.predict(X_test)
    y_test_proba = pipe.predict_proba(X_test)
    
    assert np.allclose(y_test_proba.sum(axis=1), 1.0), "Probas do not sum to 1"
    
    test_lr_res = evaluate(y_test, y_test_pred)
    
    print(f"TEST Majority Baseline Macro F1: {test_maj_res['macro_f1']:.4f}")
    print(f"TEST Logistic Regres Macro F1: {test_lr_res['macro_f1']:.4f}")
    print(f"TEST Logistic Regres Balanced Acc: {test_lr_res['balanced_accuracy']:.4f}")
    
    # Save Model
    out_dir = 'models/normal_behavior/logistic_regression_v2'
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, 'logreg_v2_corrected.joblib')
    joblib.dump(pipe, model_path)
    
    meta = {
        'dataset_identifier': 'ml_sequence_v2',
        'target': 'target_ProductCD',
        'feature_list': features,
        'class_list': pipe.named_steps['classifier'].classes_.tolist(),
        'C': 0.1,
        'class_weight': None,
        'solver': 'lbfgs',
        'max_iter': 1000,
        'random_state': 42,
        'training_row_count': len(train),
        'validation_row_count': len(val),
        'test_row_count': len(test),
        'model_creation_timestamp': time.time(),
        'model_size_bytes': os.path.getsize(model_path)
    }
    with open(os.path.join(out_dir, 'metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
        
    print("\n--- FINAL REPORT DATA ---")
    print(json.dumps({
        'test_majority': test_maj_res,
        'test_lr': test_lr_res,
        'model_metadata': meta
    }, indent=2))

if __name__ == '__main__':
    main()
