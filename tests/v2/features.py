"""
v2/features.py -- feature extraction + graph features + pooled dataset builder.

Differences from v1 features (and why):
  * FAILURE features: v1 counted status 'pending' (every Normal World legit transaction) as 'failed'.
    v2 counts only status in {failed, declined, rejected, error}.
  * STRUCTURAL features (window_seconds, transactions_per_hour, count_session_login, transactions_per_session)
    are NOT in the v2 model features: they describe how a trace was cut, not how a customer behaved.
  * New observable fields (account age, KYC scores, merchant age) exist only for v2-generated traces; elsewhere NaN.
    XGBoost handles NaN natively. v2 legit and v2 fraud both carry them, so NaN-ness alone does not reveal the label.
  * GRAPH features count how many OTHER customers share a device/payee/merchant/address/phone with this trace.
    They use no labels. (They are computed over the whole pooled dataset, i.e. transductively; in production the
    graph would be built from history before the scoring time. Noted as a limitation.)
"""
from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import blue_team_pipeline as btp
from v2.gen import build_v2_records

FAILED = {"failed", "declined", "rejected", "error"}

BEHAVIOUR = [
    "count_transaction", "count_beneficiary_addition", "count_device_registration", "total_events",
    "mean_time_between_transactions", "min_time_between_transactions", "time_login_to_first_transaction",
    "login_attempt_count_max", "auth_failure_present", "new_device_present",
    "time_device_registration_to_transaction", "beneficiary_added_before_transaction",
    "time_from_beneficiary_add_to_transaction", "failed_transaction_count", "failed_then_completed",
    "amount_mean", "amount_max", "amount_min", "amount_std", "amount_cv", "amount_change_after_failure",
    "amount_trend", "distinct_channels",
]
NEW_SCHEMA = [
    "account_age_days", "account_opening_present", "kyc_present", "kyc_liveness", "kyc_doc", "kyc_face",
    "kyc_virtual_camera", "kyc_retries", "kyc_duration_s", "merchant_age_min", "merchant_young_txn_share",
]
GRAPH = [
    "g_device_shared_customers", "g_payee_shared_customers", "g_merchant_payers", "g_address_shared_customers",
    "g_phone_shared_customers", "g_component_size",
]
COMMON_BEHAVIOUR = BEHAVIOUR   # features available for every trace in every family


def extract_features_v2(rec: dict) -> dict:
    f = btp.extract_features(rec)
    ev = sorted(rec["events"], key=lambda e: e["timestamp"])
    txns = [e for e in ev if e["event_type"] == "TRANSACTION"]
    st = [t.get("transaction_status", "") for t in txns]
    am = [float(t["amount"]) for t in txns]
    f["failed_transaction_count"] = sum(s in FAILED for s in st)
    f["failed_then_completed"] = int(any(st[i] in FAILED and st[i + 1] not in FAILED for i in range(len(st) - 1)))
    f["amount_change_after_failure"] = int(any(st[i] in FAILED and am[i + 1] < am[i] for i in range(len(am) - 1)))

    nan = np.nan
    logins = [e for e in ev if e["event_type"] == "SESSION_LOGIN"]
    ages = [e["account_age_days"] for e in logins if e.get("account_age_days") is not None]
    f["account_age_days"] = float(ages[0]) if ages else nan
    f["account_opening_present"] = int(any(e["event_type"] == "ACCOUNT_OPENING" for e in ev))
    kyc = [e for e in ev if e["event_type"] == "KYC_VERIFICATION"]
    f["kyc_present"] = int(bool(kyc))
    k = kyc[0] if kyc else None
    for name, key in [("kyc_liveness", "liveness_score"), ("kyc_doc", "doc_score"), ("kyc_face", "face_match_score"),
                      ("kyc_retries", "retries"), ("kyc_duration_s", "duration_s")]:
        f[name] = float(k[key]) if k else nan
    f["kyc_virtual_camera"] = float(k["virtual_camera"]) if k else nan
    mages = [t["merchant_age_days"] for t in txns if t.get("merchant_age_days") is not None]
    f["merchant_age_min"] = float(min(mages)) if mages else nan
    f["merchant_young_txn_share"] = float(np.mean([m < 60 for m in mages])) if mages else nan
    return f


def _entities(rec):
    dev, payee, merch, addr, phone = set(), set(), set(), set(), set()
    for e in rec["events"]:
        t = e["event_type"]
        if t in ("SESSION_LOGIN", "DEVICE_REGISTRATION") and e.get("device_id"):
            dev.add(e["device_id"])
        if t in ("BENEFICIARY_ADDITION", "TRANSACTION") and e.get("beneficiary_id"):
            payee.add(e["beneficiary_id"])
        if t == "TRANSACTION" and e.get("merchant_id"):
            merch.add(e["merchant_id"])
        if t == "ACCOUNT_OPENING":
            addr.add(e["address_hash"]); phone.add(e["phone_hash"])
    return dev, payee, merch, addr, phone


def graph_features(records: list[dict]) -> pd.DataFrame:
    """Unlabeled customer<->entity graph features. See module docstring for the transductive caveat."""
    owners = [defaultdict(set) for _ in range(5)]
    ents = []
    for r in records:
        es = _entities(r)
        ents.append(es)
        for i, s in enumerate(es):
            for x in s:
                owners[i][x].add(r["customer_id"])
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    # Connectivity uses device / payee / address / phone only. Merchants are HUBS (thousands of customers each) and would
    # glue every shopper into one giant component; customers who never shop at merchants would then look "isolated".
    for i in (0, 1, 3, 4):
        for x, cs in owners[i].items():
            cs = list(cs)
            for c in cs[1:]:
                parent[find(c)] = find(cs[0])
    for r in records:
        find(r["customer_id"])
    comp = defaultdict(int)
    for c in list(parent):
        comp[find(c)] += 1
    rows = []
    for r, es in zip(records, ents):
        def deg(i, own_is_payer=False):
            vals = [len(owners[i][x]) - (0 if own_is_payer else 1) for x in es[i]]
            return max(vals) if vals else 0
        rows.append({"trace_id": r["trace_id"],
                     "g_device_shared_customers": deg(0), "g_payee_shared_customers": deg(1),
                     "g_merchant_payers": deg(2, own_is_payer=True), "g_address_shared_customers": deg(3),
                     "g_phone_shared_customers": deg(4), "g_component_size": comp[find(r["customer_id"])]})
    return pd.DataFrame(rows)


def load_v1_records(root: Path) -> list[dict]:
    cache = root / "reports" / "v2" / "_records_v1_cache.pkl"
    recs = []
    recs += btp.load_attack_corpus(root / "reports/ato_corpus_raw.json", "ATO")
    recs += btp.load_attack_corpus(root / "reports/app_corpus_raw.json", "APP")
    recs += btp.load_attack_corpus(root / "reports/mule_corpus_raw.json", "MULE")
    recs += btp.build_legitimate_traces(btp.CONFIG)
    for r in recs:
        r["source"] = "v1"
    return recs


def build_pooled(root: Path, v2_seed=2026, hard_share=0.45, n_legit_v2=900, n_per_family=90,
                 include_v1_legit=True, **gen_kw) -> pd.DataFrame:
    v1 = load_v1_records(root)
    v2 = build_v2_records(v2_seed, n_legit_v2, hard_share, n_per_family, **gen_kw)
    recs = [r for r in v1 if include_v1_legit or r["fraud"] == 1] + v2
    feats = pd.DataFrame([extract_features_v2(r) for r in recs])
    meta = pd.DataFrame([{"trace_id": r["trace_id"], "customer_id": r["customer_id"], "fraud": r["fraud"],
                          "attack_family": r["attack_family"], "source": r["source"]} for r in recs])
    meta["family"] = np.where(meta.fraud == 0, "legitimate", meta.attack_family)
    meta["legit_kind"] = np.where(meta.fraud == 0, meta.attack_family, "")
    df = meta.merge(feats, on="trace_id").merge(graph_features(recs), on="trace_id")
    assert len(df) == len(recs) and df.trace_id.is_unique
    return df.reset_index(drop=True)


WORLD_B_HARD = dict(stealth=0.5, onboarding_share=0.20, young_share=0.12)   # default evaluation world (see PHASE notes)


def build_world_b(seed=2026, hard_share=0.45, n_legit=900, n_per_family=90, **gen_kw) -> pd.DataFrame:
    """World B only (v2 generator): used for sweeps, no v1 loading needed."""
    recs = build_v2_records(seed, n_legit, hard_share, n_per_family, **gen_kw)
    feats = pd.DataFrame([extract_features_v2(r) for r in recs])
    meta = pd.DataFrame([{"trace_id": r["trace_id"], "customer_id": r["customer_id"], "fraud": r["fraud"],
                          "attack_family": r["attack_family"], "source": r["source"]} for r in recs])
    meta["family"] = np.where(meta.fraud == 0, "legitimate", meta.attack_family)
    meta["legit_kind"] = np.where(meta.fraud == 0, meta.attack_family, "")
    return meta.merge(feats, on="trace_id").merge(graph_features(recs), on="trace_id").reset_index(drop=True)
