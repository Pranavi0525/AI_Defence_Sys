"""Tests for ML Baseline Model."""
import pytest
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Given that the model script trains on actual data and saves artifacts,
# we can just write simple structural tests for the expected output constraints.

def test_no_leakage_in_forbidden_list():
    from red_team.ml.baseline_model import FORBIDDEN
    assert 'isFraud' in FORBIDDEN
    assert 'target_ProductCD' in FORBIDDEN
    assert 'TransactionID' in FORBIDDEN
    assert 'proxy_entity' in FORBIDDEN

def test_evaluate_metrics():
    from red_team.ml.baseline_model import evaluate
    y_true = ['W', 'C', 'W']
    y_pred = ['W', 'W', 'W']
    res = evaluate(y_true, y_pred, labels=['W', 'C'])
    assert 'accuracy' in res
    assert 'macro_f1' in res
    assert 'confusion_matrix' in res
    assert res['confusion_matrix'] == [[2, 0], [1, 0]]
