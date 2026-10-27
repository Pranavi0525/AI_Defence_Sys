"""v2/hist_features.py -- per-customer deviation features, computed ONLY from the customer's own earlier events (trace['history']).
Cold start (no history): every feature is NaN except hist_n_sessions = 0."""
from __future__ import annotations

import numpy as np

from v2.history import _hour, _parse

HIST = ["hist_n_sessions", "hist_known_device", "hist_amt_z", "hist_amt_ratio", "hist_merchant_known_share",
        "hist_payee_known_share", "hist_hour_dev"]


def _circ_mean_hour(hours):
    a = np.array(hours) / 24 * 2 * np.pi
    return (np.arctan2(np.sin(a).mean(), np.cos(a).mean()) / (2 * np.pi) * 24) % 24


def hist_features(rec: dict) -> dict:
    out = {k: np.nan for k in HIST}
    h = rec.get("history") or []
    t0 = min(_parse(e["timestamp"]) for e in rec["events"])
    h = [e for e in h if _parse(e["timestamp"]) < t0]                 # causality guard: nothing at or after the trace start
    logins = [e for e in h if e["event_type"] == "SESSION_LOGIN"]
    out["hist_n_sessions"] = float(len(logins))
    if not logins:
        return out
    cur_login = next((e for e in rec["events"] if e["event_type"] == "SESSION_LOGIN"), None)
    if cur_login is not None:
        out["hist_known_device"] = float(cur_login["device_id"] in {e["device_id"] for e in logins})
        mh = _circ_mean_hour([_hour(_parse(e["timestamp"])) for e in logins])
        d = abs(_hour(_parse(cur_login["timestamp"])) - mh)
        out["hist_hour_dev"] = float(min(d, 24 - d))
    ht = [float(e["amount"]) for e in h if e["event_type"] == "TRANSACTION"]
    ct = [e for e in rec["events"] if e["event_type"] == "TRANSACTION"]
    if len(ht) >= 3 and ct:
        lg = np.log(np.maximum(ht, 1.0))
        out["hist_amt_z"] = float((np.log(max(float(e["amount"]) for e in ct)) - lg.mean()) / max(lg.std(), 0.3))
        out["hist_amt_ratio"] = float(max(float(e["amount"]) for e in ct) / max(ht))
    hm = {e["merchant_id"] for e in h if e.get("merchant_id")}
    hp = {e["beneficiary_id"] for e in h if e.get("beneficiary_id")}
    cm = [e["merchant_id"] for e in ct if e.get("merchant_id")]
    cp = [e["beneficiary_id"] for e in ct if e.get("beneficiary_id")]
    if cm:
        out["hist_merchant_known_share"] = float(np.mean([m in hm for m in cm]))
    if cp:
        out["hist_payee_known_share"] = float(np.mean([p in hp for p in cp]))
    return out
