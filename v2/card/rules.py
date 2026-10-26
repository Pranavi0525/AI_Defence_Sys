"""v2/card/rules.py -- the rule baseline: what an issuer rule engine typically does (impossible travel, velocity, limits).
Thresholds are my assumptions of ordinary rule values, NOT Pine Labs' or any bank's rules."""
from __future__ import annotations
import numpy as np
import pandas as pd

def rules_cp(f: pd.DataFrame) -> pd.Series:
    """Refined rule set: impossible travel only between card-present transactions."""
    r = ((f["speed_cp_kmh"] > 900) & (f["dist_cp_km"] > 100) & (f["gap_cp_s"] < 7200)) | (f["cnt_10m"] >= 5) | (f["amount"] > 25000) \
        | (f["atm_1h"] >= 3) | (f["dec_10m"] >= 3)
    return r.astype(int)

def rules_naive(f: pd.DataFrame) -> pd.Series:
    """Naive impossible travel on ALL channels (online purchases carry the merchant's city, so genuine customers trigger it)."""
    r = ((f["speed_prev_kmh"] > 900) & (f["dist_prev_km"] > 100) & (f["gap_s"] < 7200)) | (f["cnt_10m"] >= 5) | (f["amount"] > 25000) \
        | (f["atm_1h"] >= 3) | (f["dec_10m"] >= 3)
    return r.astype(int)
