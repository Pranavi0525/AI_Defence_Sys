"""
v2/features.py -- feature extraction + graph features + pooled dataset builder.

Differences from v1 features (and why):
  * FAILURE features: v1 counted status 'pending' (every Normal World legit transaction) as 'failed'.
    v2 counts only status in {failed, declined, rejected, error}.
  * STRUCTURAL features (window_seconds, transactions_per_hour, count_session_login, transactions_per_session)
    are NOT in the v2 model features: they describe how a trace was cut, not how a customer behaved.
  * New observable fields (account age, KYC scores, merchant age) exist only for v2-generated traces; elsewhere NaN.
    XGBoost handles NaN natively. v2 legit and v2 fraud both carry them, so NaN-ness alone does not reveal the label.
  * GRAPH features count how many OTHER customers share a device/payee/merchant/address/phone with this trace,
    using ONLY traces that occurred before it (time-ordered replay through v2/serving/store.py, the same class the API
    uses online). No labels, no future information, no train/serve skew.
"""
from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import blue_team_pipeline as btp
from v2.gen import build_v2_records
from v2.serving.core_features import extract_features_core
from v2.serving.store import replay_features

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
    """Thin wrapper: the ONE extractor lives in v2/serving/core_features.py (shared with the API)."""
    return extract_features_core(rec)


def graph_features(records: list[dict]) -> pd.DataFrame:
    """History-only graph features: records are replayed in time order through the same EntityStore the API uses,
    so each trace sees only traces that happened BEFORE it."""
    rows, _ = replay_features(records)
    df = pd.DataFrame(rows)
    df.insert(0, "trace_id", [r["trace_id"] for r in records])
    return df


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


def build_world_b(seed=2026, hard_share=0.45, n_legit=900, n_per_family=90, return_records=False, **gen_kw):
    """World B only (v2 generator): used for sweeps and for the served model, no v1 loading needed."""
    recs = build_v2_records(seed, n_legit, hard_share, n_per_family, **gen_kw)
    feats = pd.DataFrame([extract_features_v2(r) for r in recs])
    meta = pd.DataFrame([{"trace_id": r["trace_id"], "customer_id": r["customer_id"], "fraud": r["fraud"],
                          "attack_family": r["attack_family"], "source": r["source"]} for r in recs])
    meta["family"] = np.where(meta.fraud == 0, "legitimate", meta.attack_family)
    meta["legit_kind"] = np.where(meta.fraud == 0, meta.attack_family, "")
    df = meta.merge(feats, on="trace_id").merge(graph_features(recs), on="trace_id").reset_index(drop=True)
    return (df, recs) if return_records else df
