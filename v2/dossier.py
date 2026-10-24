"""
v2/dossier.py -- Phase 7a: a readable case dossier for every REVIEW / BLOCK decision of the v2 cascade.

A dossier is what an analyst would open. It has five parts and says where each part comes from:
  1. DECISION       decision, score, thresholds, distance to each threshold          (model output)
  2. MODEL VIEW     XGBoost probability, novelty percentiles, stacked score          (model output)
  3. MODEL DRIVERS  top XGBoost feature contributions in plain words                 (model output, log-odds)
  4. EVIDENCE       fixed, readable checks on the raw trace (new payee paid fast,    (rule-of-thumb flags: NOT model
                    virtual camera, young merchant, shared entities, ...)             outputs, cut-offs are not tuned)
  5. ACTING PARTY   who bears the loss / who should act, for a FAMILY                (a policy mapping, not statistics)

Honest limits (also written into every dossier):
  * At scoring time the fraud family is UNKNOWN. The acting-party block is filled only when an analyst (or an evaluation
    script that knows the label) supplies `family`. Without it the dossier says so instead of guessing.
  * Liability for ACCOUNT_TAKEOVER / AUTHORIZED_PUSH_PAYMENT / MULE_NETWORK is the v1 mapping (tests/test_v2_dossier.py checks
    it still equals decision_policy.liable_side). For the four new families it is MY ASSUMPTION, not a regulatory reading.
  * Everything is synthetic-data; scores are not real fraud probabilities.

Runtime needs no scikit-learn / pandas: it only post-processes what V2Scorer already returns.
"""
from __future__ import annotations

import math

# ---------------------------------------------------------------------------------------------- acting party
# (liable_side, acting_side, basis, note).  basis "v1_policy" = copied from decision_policy.py; "ASSUMED" = author's assumption.
LIABILITY_V2 = {
    "ACCOUNT_TAKEOVER": ("SENDING", "SENDING", "v1_policy",
                         "Customer did not authorise the payment; the customer's own bank refunds it."),
    "AUTHORIZED_PUSH_PAYMENT": ("SHARED_50_50", "BOTH", "v1_policy",
                                "Customer was tricked into paying; cost is split between sending and receiving institution."),
    "MULE_NETWORK": ("RECEIVING", "RECEIVING", "v1_policy",
                     "The fraud is the receiving (mule) account; the institution that onboarded it is responsible."),
    "AI_SCRIPTED_APP": ("SHARED_50_50", "BOTH", "ASSUMED",
                        "Same mechanics as an authorised push payment (customer pushes money), so the APP split is assumed."),
    "SYNTHETIC_IDENTITY": ("ONBOARDING_INSTITUTION", "THIS_INSTITUTION", "ASSUMED",
                           "The 'customer' never existed; the institution that opened the account and extended access carries the bust-out loss."),
    "DEEPFAKE_KYC": ("ONBOARDING_INSTITUTION", "THIS_INSTITUTION", "ASSUMED",
                     "Identity checks were passed by a forged or injected identity at onboarding; the onboarding institution carries the loss."),
    "FAKE_MERCHANT": ("SHARED_CARD_OR_ACQUIRER", "BOTH", "ASSUMED",
                      "Customer paid a fraudulent merchant; who refunds depends on scheme / acquirer rules that this project does not model."),
}

CAVEATS = [
    "Synthetic-data prototype: the score is not a real-world fraud probability.",
    "Evidence flags are fixed, readable checks on the raw trace, not model outputs; their cut-offs were not tuned.",
    "The fraud family is unknown at scoring time; the acting-party block applies only if a family was supplied.",
]

# ---------------------------------------------------------------------------------------------- plain-language labels
LABELS = {
    "count_transaction": "number of transactions", "count_beneficiary_addition": "number of payees added",
    "count_device_registration": "number of devices registered", "total_events": "total events in the trace",
    "mean_time_between_transactions": "average gap between transactions (s)", "min_time_between_transactions": "shortest gap between transactions (s)",
    "time_login_to_first_transaction": "time from login to first transaction (s)", "login_attempt_count_max": "most login attempts in a session",
    "auth_failure_present": "a login authentication failed", "new_device_present": "a new device was registered",
    "time_device_registration_to_transaction": "time from device registration to a transaction (s)",
    "beneficiary_added_before_transaction": "a payee was added before paying", "time_from_beneficiary_add_to_transaction": "time from adding a payee to paying (s)",
    "failed_transaction_count": "number of failed transactions", "failed_then_completed": "a failed payment was followed by a completed one",
    "amount_mean": "average amount", "amount_max": "largest amount", "amount_min": "smallest amount", "amount_std": "spread of amounts",
    "amount_cv": "relative spread of amounts", "amount_change_after_failure": "amount lowered after a failure", "amount_trend": "amounts rising (+1) or falling (-1)",
    "distinct_channels": "number of channels used", "account_age_days": "account age (days)", "account_opening_present": "an account was opened in this trace",
    "kyc_present": "an identity (KYC) check is in the trace", "kyc_liveness": "KYC liveness score", "kyc_doc": "KYC document score",
    "kyc_face": "KYC face-match score", "kyc_virtual_camera": "KYC used a virtual camera", "kyc_retries": "KYC retries", "kyc_duration_s": "KYC duration (s)",
    "merchant_age_min": "age of the youngest merchant paid (days)", "merchant_young_txn_share": "share of payments to merchants under 60 days old",
    "g_device_shared_customers": "other customers sharing this device", "g_payee_shared_customers": "other customers who also paid this payee",
    "g_merchant_payers": "customers who have paid this merchant", "g_address_shared_customers": "other customers sharing this address",
    "g_phone_shared_customers": "other customers sharing this phone", "g_component_size": "size of the linked customer group",
}


def label(name: str) -> str:
    return LABELS.get(name, name.replace("_", " "))


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "missing"
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return f"{v:.3g}" if isinstance(v, float) else str(v)


# ---------------------------------------------------------------------------------------------- evidence (readable checks)
FAST_PAY_SECONDS = 3600          # a payee added and paid within an hour
YOUNG_ACCOUNT_DAYS = 90
YOUNG_MERCHANT_DAYS = 60
LOW_LIVENESS = 0.6


def _nn(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def evidence_flags(f: dict) -> list[dict]:
    """Readable observations on the raw feature values. Each item: {flag, detail}. No model involved."""
    out = []

    def add(flag, detail):
        out.append({"flag": flag, "detail": detail})

    if f.get("beneficiary_added_before_transaction") == 1 and 0 <= f.get("time_from_beneficiary_add_to_transaction", -1) <= FAST_PAY_SECONDS:
        add("new_payee_paid_quickly", f"A payee was added and paid {_fmt(f['time_from_beneficiary_add_to_transaction'])} s later.")
    if f.get("new_device_present") == 1:
        add("new_device", "A new device was registered in this trace.")
    if _nn(f.get("kyc_virtual_camera")) and f["kyc_virtual_camera"] >= 1:
        add("kyc_virtual_camera", "The identity check was done through a virtual camera.")
    if _nn(f.get("kyc_liveness")) and f["kyc_liveness"] < LOW_LIVENESS:
        add("kyc_low_liveness", f"KYC liveness score {_fmt(f['kyc_liveness'])} is below {LOW_LIVENESS}.")
    if _nn(f.get("kyc_retries")) and f["kyc_retries"] >= 3:
        add("kyc_many_retries", f"KYC was retried {_fmt(f['kyc_retries'])} times.")
    if _nn(f.get("account_age_days")) and f["account_age_days"] < YOUNG_ACCOUNT_DAYS:
        add("young_account", f"The account is {_fmt(f['account_age_days'])} day(s) old.")
    if _nn(f.get("merchant_age_min")) and f["merchant_age_min"] < YOUNG_MERCHANT_DAYS:
        add("young_merchant", f"A payment went to a merchant only {_fmt(f['merchant_age_min'])} day(s) old.")
    if f.get("failed_then_completed") == 1:
        add("retry_after_failure", "A failed payment was followed by a completed one.")
    if f.get("count_transaction", 0) >= 4 and _nn(f.get("min_time_between_transactions")) and f["min_time_between_transactions"] < 120:
        add("rapid_payments", f"{_fmt(f['count_transaction'])} payments with the shortest gap {_fmt(f['min_time_between_transactions'])} s.")
    return out


def graph_evidence(g: dict) -> list[str]:
    """Plain sentences about shared entities, from HISTORY-ONLY graph features (earlier traces only)."""
    out = []
    for key, text in (("g_device_shared_customers", "device"), ("g_payee_shared_customers", "payee"),
                      ("g_address_shared_customers", "address"), ("g_phone_shared_customers", "phone number")):
        n = g.get(key, 0) or 0
        if n >= 1:
            out.append(f"This {text} was already used by {int(n)} other customer(s) in earlier traces.")
    if (g.get("g_merchant_payers", 0) or 0) >= 25:
        out.append(f"The merchant paid has {int(g['g_merchant_payers'])} distinct paying customers (a hub, so weak evidence on its own).")
    if (g.get("g_component_size", 1) or 1) >= 3:
        out.append(f"This customer is linked to a group of {int(g['g_component_size'])} customers through shared devices, payees, addresses or phones.")
    if not out:
        out.append("No links to other customers were found in earlier traces (the graph stage adds nothing here).")
    return out


SUGGESTED_CHECKS = {
    "new_payee_paid_quickly": "Confirm with the customer that they know and intended to pay the new payee.",
    "new_device": "Confirm the new device through a second channel.",
    "kyc_virtual_camera": "Re-run identity verification with a live, in-person or supervised check.",
    "kyc_low_liveness": "Re-run identity verification; compare against document and face-match scores.",
    "young_account": "Look at what else this young account has done; compare with onboarding-day behaviour.",
    "young_merchant": "Check the merchant's registration and complaint history before settling.",
    "retry_after_failure": "Check whether the retry used the same payee and a lowered amount.",
}


# ---------------------------------------------------------------------------------------------- dossier
def acting_party(family: str | None) -> dict:
    if not family:
        return {"family": None, "note": "Fraud family unknown at scoring time; supply it after analyst triage to get the liability view."}
    if family not in LIABILITY_V2:
        return {"family": family, "note": "No liability mapping for this family."}
    liable, acting, basis, note = LIABILITY_V2[family]
    return {"family": family, "liable_side": liable, "acting_side": acting, "basis": basis, "note": note}


def _money(record: dict) -> dict:
    amts = [float(e["amount"]) for e in record["events"] if e.get("event_type") == "TRANSACTION" and e.get("amount") is not None]
    return {"transactions": len(amts), "total_amount": round(sum(amts), 2), "largest_amount": round(max(amts), 2) if amts else 0.0}


def build_dossier(scorer, record: dict, family: str | None = None, include_allow: bool = False) -> dict | None:
    """Score `record` (read-only, commit=False) and return a dossier, or None for ALLOW (unless include_allow)."""
    res = scorer.score(record, commit=False)
    if res["decision"] == "ALLOW" and not include_allow:
        return None
    # features again (cheap) for the evidence block; same extractor + history-only graph as the scorer used
    from v2.serving.core_features import extract_features_core, normalize_record
    rec = normalize_record(record)
    f = extract_features_core(rec)
    f.update(res["graph_features"])
    t_r, t_b, s = res["thresholds"]["t_review"], res["thresholds"]["t_block"], res["stacked_probability"]
    drivers = [{"feature": d["feature"], "meaning": label(d["feature"]), "value": _fmt(d["value"]),
                "contribution_log_odds": round(d["contribution_log_odds"], 3),
                "direction": "raises risk" if d["contribution_log_odds"] > 0 else "lowers risk"} for d in res["top_features"]]
    ev = evidence_flags(f)
    headline = (f"{res['decision']}: stacked score {s:.3f} vs review {t_r:.3f} / block {t_b:.3f}. "
                + (f"Strongest driver: {drivers[0]['meaning']} ({drivers[0]['value']}) {drivers[0]['direction']}. " if drivers else "")
                + (f"{len(ev)} evidence flag(s)." if ev else "No evidence flags fired (the model is reacting to a combination of features)."))
    return {
        "trace_id": record.get("trace_id"), "customer_id": record.get("customer_id"), "model_version": res["model_version"],
        "decision": res["decision"], "headline": headline,
        "decision_detail": {"risk_score": s, "t_review": t_r, "t_block": t_b,
                            "margin_to_review": round(s - t_r, 4), "margin_to_block": round(s - t_b, 4)},
        "model_view": {"xgboost_probability": res["xgb_probability"], "isolation_forest_percentile": res["iforest_percentile"],
                       "autoencoder_percentile": res["autoencoder_percentile"], "novelty_note": res["novelty_note"]},
        "model_drivers": drivers, "evidence_flags": ev, "graph_evidence": graph_evidence(res["graph_features"]),
        "money": _money(rec), "acting_party": acting_party(family),
        "suggested_checks": [SUGGESTED_CHECKS[e["flag"]] for e in ev if e["flag"] in SUGGESTED_CHECKS],
        "caveats": CAVEATS,
    }


def render_markdown(d: dict) -> str:
    dd, mv = d["decision_detail"], d["model_view"]
    L = [f"## Case {d['trace_id']} -- {d['decision']}", "", d["headline"], "",
         f"**Customer:** {d['customer_id']}  |  **Money in trace:** {d['money']['total_amount']} across {d['money']['transactions']} payment(s)  |  **Model:** {d['model_version']}", "",
         "### 1. Decision",
         f"- risk score {dd['risk_score']:.3f}; review at {dd['t_review']:.3f} (margin {dd['margin_to_review']:+.3f}), block at {dd['t_block']:.3f} (margin {dd['margin_to_block']:+.3f})", "",
         "### 2. Model view",
         f"- XGBoost (known fraud patterns): {mv['xgboost_probability']:.3f}",
         f"- Isolation Forest, unusual vs normal traffic: percentile {mv['isolation_forest_percentile']:.3f}",
         f"- Autoencoder, unusual vs normal traffic: percentile {mv['autoencoder_percentile']:.3f}"]
    if mv["novelty_note"]:
        L.append(f"- Note: {mv['novelty_note']}")
    L += ["", "### 3. What pushed the model (XGBoost, log-odds)"]
    L += [f"- {x['meaning']} = {x['value']}: {x['direction']} ({x['contribution_log_odds']:+.2f})" for x in d["model_drivers"]] or ["- (no contribution above noise)"]
    if any(x["value"] == "missing" and x["direction"] == "raises risk" for x in d["model_drivers"]):
        L.append("- Note: a MISSING value is itself a signal the model uses (e.g. no merchant purchase raises risk). That is partly a simulator artifact, so treat it as weak evidence.")
    L += ["", "### 4. Evidence on the raw trace (readable checks, not model outputs)"]
    L += [f"- {e['detail']}" for e in d["evidence_flags"]] or ["- none fired"]
    L += ["", "### 5. Graph evidence (earlier traces only)"] + [f"- {s}" for s in d["graph_evidence"]]
    ap = d["acting_party"]
    L += ["", "### 6. Acting party"]
    if ap.get("liable_side"):
        L.append(f"- family {ap['family']}: liable {ap['liable_side']}, should act {ap['acting_side']} (basis: {ap['basis']}). {ap['note']}")
    else:
        L.append(f"- {ap['note']}")
    if d["suggested_checks"]:
        L += ["", "### Suggested checks"] + [f"- {c}" for c in d["suggested_checks"]]
    L += ["", "_" + " ".join(d["caveats"]) + "_", ""]
    return "\n".join(L)
