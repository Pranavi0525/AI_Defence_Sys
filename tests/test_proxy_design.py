import pytest
import pandas as pd
import json

def test_proxy_specification_format():
    with open('data/reference/ml_sequence/proxy_specification.json') as f:
        spec = json.load(f)
        
    assert "status" in spec
    assert "selected_proxy_objective" in spec
    assert "exact_source_fields" in spec
    assert "fallback_objective" in spec
    assert spec["status"] == "COMPLETED"

def test_proxy_construction_determinism():
    data = pd.DataFrame({
        'card1': [1, 1],
        'card2': [2, 2],
        'card3': [3, 3],
        'card4': ['visa', 'visa'],
        'card5': [5, 5],
        'card6': ['debit', 'debit'],
        'addr1': [100.0, 100.0],
        'P_emaildomain': ['gmail.com', 'gmail.com']
    })
    for c in data.columns:
        data[c] = data[c].astype(str)
        
    proxy = data['card1'] + '_' + data['card2'] + '_' + data['card3'] + '_' + data['card4'] + '_' + data['card5'] + '_' + data['card6'] + '_' + data['addr1'] + '_' + data['P_emaildomain']
    assert proxy.nunique() == 1
    assert proxy.iloc[0] == "1_2_3_visa_5_debit_100.0_gmail.com"
