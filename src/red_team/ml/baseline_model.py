"""Stage 4.5C: First ML Baseline Model."""
import os
import json
import time
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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
    train = pd.read_csv('data/reference/ml_sequence/train.csv')
    val = pd.read_csv('data/reference/ml_sequence/validation.csv')
    test = pd.read_csv('data/reference/ml_sequence/test.csv')
    
    with open('data/reference/ml_sequence/feature_manifest.json') as f:
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
    y_val_maj = [majority_class] * len(y_val)
    val_maj_res = evaluate(y_val, y_val_maj)
    
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
    
    configs = [
        {'C': 0.1, 'class_weight': None},
        {'C': 1.0, 'class_weight': None},
        {'C': 10.0, 'class_weight': None},
        {'C': 0.1, 'class_weight': 'balanced'},
        {'C': 1.0, 'class_weight': 'balanced'},
        {'C': 10.0, 'class_weight': 'balanced'},
    ]
    
    results = []
    best_f1 = -1
    best_pipe = None
    best_config = None
    best_res = None
    
    print("Training Logistic Regression models...")
    for cfg in configs:
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                C=cfg['C'], class_weight=cfg['class_weight'],
                solver='lbfgs', max_iter=1000, random_state=42
            ))
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_val)
        res = evaluate(y_val, y_pred)
        
        print(f"Config: {cfg} -> Macro F1: {res['macro_f1']:.4f}, Balanced Acc: {res['balanced_accuracy']:.4f}")
        
        if res['macro_f1'] > best_f1:
            best_f1 = res['macro_f1']
            best_pipe = pipe
            best_config = cfg
            best_res = res
            
    print(f"\nBest configuration selected on VAL: {best_config}")
    
    # Freeze model and evaluate on TEST
    print("\nEvaluating selected model on TEST...")
    y_test_pred = best_pipe.predict(X_test)
    y_test_proba = best_pipe.predict_proba(X_test)
    
    assert np.allclose(y_test_proba.sum(axis=1), 1.0), "Probas do not sum to 1"
    
    test_lr_res = evaluate(y_test, y_test_pred)
    
    print(f"TEST Majority Baseline Macro F1: {test_maj_res['macro_f1']:.4f}")
    print(f"TEST Logistic Regres Macro F1: {test_lr_res['macro_f1']:.4f}")
    
    # Interpretability
    cat_enc = best_pipe.named_steps['preprocessor'].named_transformers_['cat'].named_steps['ohe']
    cat_names = cat_enc.get_feature_names_out(cat_cols)
    feature_names = num_cols + list(cat_names)
    
    coefs = best_pipe.named_steps['classifier'].coef_
    classes_ = best_pipe.named_steps['classifier'].classes_
    
    associations = {}
    for i, cls in enumerate(classes_):
        class_coefs = coefs[i]
        top_pos_idx = np.argsort(class_coefs)[-3:][::-1]
        top_neg_idx = np.argsort(class_coefs)[:3]
        associations[cls] = {
            'positive': [(feature_names[j], class_coefs[j]) for j in top_pos_idx],
            'negative': [(feature_names[j], class_coefs[j]) for j in top_neg_idx]
        }
    
    # Error Analysis
    cm = np.array(test_lr_res['confusion_matrix'])
    np.fill_diagonal(cm, 0)
    idx_max = np.argmax(cm)
    row, col = np.unravel_index(idx_max, cm.shape)
    common_conf = (CLASSES[row], CLASSES[col], cm[row, col])
    
    lowest_recall_idx = np.argmin(test_lr_res['per_class_recall'])
    lowest_f1_idx = np.argmin(test_lr_res['per_class_f1'])
    
    error_analysis = {
        'most_common_confusion': f"True {common_conf[0]} predicted as {common_conf[1]} ({common_conf[2]} times)",
        'lowest_recall_class': CLASSES[lowest_recall_idx],
        'lowest_f1_class': CLASSES[lowest_f1_idx]
    }
    
    # Save Model
    out_dir = 'models/normal_behavior/logistic_regression'
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, 'model.joblib')
    joblib.dump(best_pipe, model_path)
    
    meta = {
        'dataset_identifier': 'ml_sequence',
        'target': 'target_ProductCD',
        'feature_list': features,
        'class_list': classes_.tolist(),
        'C': best_config['C'],
        'class_weight': best_config['class_weight'],
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
        
    # Output final report format data to console
    print("\n--- FINAL REPORT DATA ---")
    print(json.dumps({
        'test_majority': test_maj_res,
        'test_lr': test_lr_res,
        'associations': associations,
        'error_analysis': error_analysis,
        'model_metadata': meta
    }, indent=2))

if __name__ == '__main__':
    main()
