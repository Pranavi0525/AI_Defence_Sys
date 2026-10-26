"""
v2/serving/core_features.py -- THE feature extractor, shared by training and serving.

Pure python + numpy (no sklearn, pandas, pydantic or blue_team_pipeline import) so the deployed API can run it.
It reproduces blue_team_pipeline.extract_features (same keys, same maths) with the v2 fixes:
  * failure = status in {failed, declined, rejected, error}  (v1 counted 'pending' as failed)
  * extra fields for v2 traces: account age, account opening, KYC telemetry, merchant age.
tests/test_v2_serving.py checks equality against the v1 extractor on real traces.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime

import numpy as np

FAILED = {"failed", "declined", "rejected", "error"}
NaN = float("nan")


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def normalize_record(rec: dict) -> dict:
    """Guarantee sorted events and an observation_window (first/last event) so scoring never crashes on a missing one."""
    ev = sorted(rec["events"], key=lambda e: e["timestamp"])
    if not ev:
        raise ValueError("trace has no events")
    out = dict(rec)
    out["events"] = ev
    if not rec.get("observation_window"):
        out["observation_window"] = [ev[0]["timestamp"], ev[-1]["timestamp"]]
    return out


def extract_features_core(rec: dict) -> dict:
    rec = normalize_record(rec)
    events = rec["events"]
    f: dict = {"trace_id": rec.get("trace_id", "")}
    counts = defaultdict(int)
    for e in events:
        counts[e["event_type"]] += 1
    for t in ("TRANSACTION", "SESSION_LOGIN", "DEVICE_REGISTRATION", "BENEFICIARY_ADDITION"):
        f[f"count_{t.lower()}"] = counts.get(t, 0)
    f["total_events"] = len(events)

    start, end = _dt(rec["observation_window"][0]), _dt(rec["observation_window"][1])
    window = max((end - start).total_seconds(), 1.0)
    f["window_seconds"] = window

    txns = [e for e in events if e["event_type"] == "TRANSACTION"]
    logins = [e for e in events if e["event_type"] == "SESSION_LOGIN"]
    benefs = [e for e in events if e["event_type"] == "BENEFICIARY_ADDITION"]
    devices = [e for e in events if e["event_type"] == "DEVICE_REGISTRATION"]

    f["transactions_per_hour"] = len(txns) / (window / 3600.0)
    if len(txns) >= 2:
        tt = [_dt(t["timestamp"]) for t in txns]
        gaps = [(tt[i + 1] - tt[i]).total_seconds() for i in range(len(tt) - 1)]
        f["mean_time_between_transactions"] = float(np.mean(gaps))
        f["min_time_between_transactions"] = float(np.min(gaps))
    else:
        f["mean_time_between_transactions"] = window
        f["min_time_between_transactions"] = window

    f["transactions_per_session"] = len(txns) / max(len(logins), 1)
    if logins and txns:
        f["time_login_to_first_transaction"] = max((_dt(txns[0]["timestamp"]) - _dt(logins[0]["timestamp"])).total_seconds(), 0.0)
    else:
        f["time_login_to_first_transaction"] = -1.0
    f["login_attempt_count_max"] = max((l.get("login_attempt_count", 1) for l in logins), default=0)
    f["auth_failure_present"] = int(any(l.get("auth_success") is False for l in logins))

    f["new_device_present"] = int(len(devices) > 0)
    f["time_device_registration_to_transaction"] = -1.0
    if devices and txns:
        d0 = _dt(devices[0]["timestamp"])
        after = [_dt(t["timestamp"]) for t in txns if _dt(t["timestamp"]) >= d0]
        if after:
            f["time_device_registration_to_transaction"] = (min(after) - d0).total_seconds()

    f["beneficiary_added_before_transaction"] = 0
    f["time_from_beneficiary_add_to_transaction"] = -1.0
    if benefs and txns:
        b0 = _dt(benefs[0]["timestamp"])
        later = [_dt(t["timestamp"]) for t in txns if _dt(t["timestamp"]) >= b0]
        if later:
            f["beneficiary_added_before_transaction"] = 1
            f["time_from_beneficiary_add_to_transaction"] = (min(later) - b0).total_seconds()

    st = [t.get("transaction_status", "") for t in txns]
    am = [float(t["amount"]) for t in txns]
    f["failed_transaction_count"] = sum(s in FAILED for s in st)
    f["failed_then_completed"] = int(any(st[i] in FAILED and st[i + 1] not in FAILED for i in range(len(st) - 1)))
    if am:
        f["amount_mean"], f["amount_max"], f["amount_min"] = float(np.mean(am)), float(np.max(am)), float(np.min(am))
        f["amount_std"] = float(np.std(am))
        f["amount_cv"] = f["amount_std"] / max(f["amount_mean"], 1e-6)
        if len(am) >= 2:
            f["amount_change_after_failure"] = int(any(st[i] in FAILED and am[i + 1] < am[i] for i in range(len(am) - 1)))
            f["amount_trend"] = float(np.sign(am[-1] - am[0]))
        else:
            f["amount_change_after_failure"], f["amount_trend"] = 0, 0.0
    else:
        f.update({"amount_mean": 0.0, "amount_max": 0.0, "amount_min": 0.0, "amount_std": 0.0, "amount_cv": 0.0,
                  "amount_change_after_failure": 0, "amount_trend": 0.0})
    f["distinct_channels"] = len({t.get("channel") for t in txns if t.get("channel")})

    # ---- v2 schema fields (NaN when the trace does not carry them)
    ages = [e["account_age_days"] for e in logins if e.get("account_age_days") is not None]
    f["account_age_days"] = float(ages[0]) if ages else NaN
    f["account_opening_present"] = int(any(e["event_type"] == "ACCOUNT_OPENING" for e in events))
    kyc = [e for e in events if e["event_type"] == "KYC_VERIFICATION"]
    f["kyc_present"] = int(bool(kyc))
    k = kyc[0] if kyc else None
    for name, key in (("kyc_liveness", "liveness_score"), ("kyc_doc", "doc_score"), ("kyc_face", "face_match_score"),
                      ("kyc_retries", "retries"), ("kyc_duration_s", "duration_s")):
        f[name] = float(k[key]) if k else NaN
    f["kyc_virtual_camera"] = float(k["virtual_camera"]) if k else NaN
    mages = [t["merchant_age_days"] for t in txns if t.get("merchant_age_days") is not None]
    f["merchant_age_min"] = float(min(mages)) if mages else NaN
    f["merchant_young_txn_share"] = float(np.mean([m < 60 for m in mages])) if mages else NaN
    return f
