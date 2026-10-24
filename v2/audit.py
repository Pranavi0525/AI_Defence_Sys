"""v2/audit.py -- artifact audits: raw-field separability, feature separability, and simulator distinguishability."""
from __future__ import annotations

import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestClassifier


def _fmt(v):
    if v is None:
        return "None"
    s = str(v)
    if re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", s):
        return "<uuid>"
    return re.sub(r"[0-9a-f]+", "#", s)[:18]


FIELDS = [("device_id", ("SESSION_LOGIN", "DEVICE_REGISTRATION")), ("fingerprint", ("DEVICE_REGISTRATION",)),
          ("channel", ("TRANSACTION",)), ("transaction_status", ("TRANSACTION",)), ("merchant_category", ("TRANSACTION",)),
          ("relationship_type", ("BENEFICIARY_ADDITION",)), ("is_verified", ("BENEFICIARY_ADDITION",)),
          ("auth_method", ("SESSION_LOGIN",))]


def raw_field_audit(records: list[dict]) -> list[dict]:
    """For each categorical raw field: value set in legit vs fraud traces, and whether they are DISJOINT
    (a disjoint field is a perfect label proxy that any model reading raw fields would exploit)."""
    rows = []
    for field, etypes in FIELDS:
        vals = {0: Counter(), 1: Counter()}
        for r in records:
            for e in r["events"]:
                if e["event_type"] in etypes and field in e:
                    vals[r["fraud"]][_fmt(e[field]) if field in ("device_id", "fingerprint") else str(e[field])] += 1
        l, f = set(vals[0]), set(vals[1])
        rows.append({"field": field, "legit_values": dict(vals[0].most_common(4)), "fraud_values": dict(vals[1].most_common(4)),
                     "disjoint_value_sets": bool(l and f and not (l & f))})
    return rows


def feature_auc_audit(dfw: pd.DataFrame, feats: list, top=8) -> list[dict]:
    rows = []
    for c in feats:
        x = dfw[c].fillna(-999)
        if x.nunique() > 1:
            a = roc_auc_score(dfw.fraud, x)
            rows.append({"feature": c, "auc": float(a), "strength": float(abs(a - 0.5))})
    return sorted(rows, key=lambda r: -r["strength"])[:top]


def simulator_distinguishability(df: pd.DataFrame, feats: list) -> float:
    """Among LEGIT traces only, can a classifier tell which simulator (v1 Normal World vs v2 generator) made it?
    AUC near 0.5 = same-looking; near 1.0 = big domain gap (so cross-world transfer numbers are confounded)."""
    leg = df[df.fraud == 0]
    X = leg[feats].fillna(-999).values
    y = (leg.source == "v2_gen").astype(int).values
    p = cross_val_predict(RandomForestClassifier(200, random_state=0, n_jobs=1), X, y, cv=5, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))
