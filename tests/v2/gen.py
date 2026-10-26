"""
v2/gen.py  --  lightweight trace generators for the v2 evaluation layer.

WHAT THIS IS (and is not)
-------------------------
* It produces observable-trace dicts in the SAME JSON shape as the Red Team corpora, so the same feature
  extractor can read them. It is NOT run through the Red Team's pydantic validators/realism checks.
  Porting these into src/red_team/attacks/ as proper signatures is a later step.
* Every distribution below is an ASSUMPTION chosen by me (documented inline), not an empirical estimate
  from real payment data. Results on this data are stress-test results, never real-world performance.
* All entity ids are random UUIDs for every class (no 'dev_' / 'atk_fp_' style prefixes), so id text can't leak the label.

LEGIT = routine sessions + "hard negatives": benign behavior that LOOKS like fraud on some feature
(bursts of bill payments, new payee then pay, new phone, retries, large one-off transfer, genuine onboarding,
shared household device, popular shared payee). hard_share controls how many.

NEW FRAUD FAMILIES (GenAI-era, from the hackathon brief):
  SYNTHETIC_IDENTITY, FAKE_MERCHANT, DEEPFAKE_KYC, AI_SCRIPTED_APP
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta

import numpy as np

BASE = datetime(2025, 3, 1)
FMT = "%Y-%m-%dT%H:%M:%S"
NEW_FAMILIES = ["SYNTHETIC_IDENTITY", "FAKE_MERCHANT", "DEEPFAKE_KYC", "AI_SCRIPTED_APP"]


class G:
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def uid(self) -> str:
        r = self.rng
        return str(uuid.UUID(int=(int(r.integers(0, 2**62)) << 64) | int(r.integers(0, 2**62)), version=4))

    def ln(self, median, sigma, size=None):
        return self.rng.lognormal(np.log(median), sigma, size)

    def pick(self, seq):
        return seq[int(self.rng.integers(0, len(seq)))]


# ---------------------------------------------------------------- event builders
def _ts(t):  # datetime -> iso
    return t.strftime(FMT)


def e_login(g, t, cust, acct, sess, dev, age, attempts=1, ok=True):
    return {"event_type": "SESSION_LOGIN", "event_id": g.uid(), "timestamp": _ts(t), "customer_id": cust,
            "account_id": acct, "session_id": sess, "device_id": dev, "auth_method": "password",
            "auth_success": ok, "login_attempt_count": attempts, "account_age_days": age}


def e_txn(g, t, cust, acct, amount, ttype, status, channel, merchant_id=None, merchant_age=None,
          beneficiary_id=None):
    return {"event_type": "TRANSACTION", "event_id": g.uid(), "timestamp": _ts(t), "customer_id": cust,
            "account_id": acct, "amount": f"{amount:.2f}", "currency": "USD", "transaction_type": ttype,
            "channel": channel, "transaction_status": status, "merchant_id": merchant_id,
            "merchant_age_days": merchant_age, "beneficiary_id": beneficiary_id}


def e_benef(g, t, cust, benef, verified):
    return {"event_type": "BENEFICIARY_ADDITION", "event_id": g.uid(), "timestamp": _ts(t), "customer_id": cust,
            "beneficiary_id": benef, "relationship_type": "other", "is_verified": bool(verified), "action": "add"}


def e_devreg(g, t, cust, dev):
    return {"event_type": "DEVICE_REGISTRATION", "event_id": g.uid(), "timestamp": _ts(t), "customer_id": cust,
            "device_id": dev, "device_type": "mobile", "action": "register"}


def e_open(g, t, cust, phone, addr):
    return {"event_type": "ACCOUNT_OPENING", "event_id": g.uid(), "timestamp": _ts(t), "customer_id": cust,
            "phone_hash": phone, "address_hash": addr}


def e_kyc(g, t, cust, liveness, doc, face, vcam, retries, duration):
    return {"event_type": "KYC_VERIFICATION", "event_id": g.uid(), "timestamp": _ts(t), "customer_id": cust,
            "liveness_score": float(np.clip(liveness, 0, 1)), "doc_score": float(np.clip(doc, 0, 1)),
            "face_match_score": float(np.clip(face, 0, 1)), "virtual_camera": bool(vcam),
            "retries": int(retries), "duration_s": float(duration)}


def _trace(prefix, family, fraud, cust, events, difficulty="n/a"):
    events = sorted(events, key=lambda e: e["timestamp"])
    return {"trace_id": f"{prefix}_{events[0]['event_id'][:12]}", "customer_id": cust, "events": events,
            "observation_window": [events[0]["timestamp"], events[-1]["timestamp"]],
            "fraud": fraud, "attack_family": family, "attack_difficulty": difficulty, "source": "v2_gen"}


# ---------------------------------------------------------------- shared pools
class Pools:
    """Entities shared across traces: needed so graph features have something real to find."""
    def __init__(self, g: G, young_share: float = 0.0):
        self.young_share = young_share
        self.popular_payees = [g.uid() for _ in range(15)]                 # legit shared payees (landlords, utilities)
        self.legit_merchants = [(g.uid(), int(g.ln(900, 0.9)) + 20) for _ in range(60)]   # (id, age_days)
        self.young_legit_merchants = [(g.uid(), int(g.rng.integers(5, 150))) for _ in range(12)]
        self.fake_merchants = [(g.uid(), int(g.rng.integers(2, 30))) for _ in range(6)]
        self.scam_payees = [g.uid() for _ in range(40)]       # many payees => per-payee degree overlaps legit popular payees
        self.offices = []      # (device, address, remaining_seats): shared devices/addresses among legit customers
        self.households = []   # (device, address) shared by 2-4 legit customers


def _routine_txns(g, t0, cust, acct, n, pools, gap_med=1200, amt_med=900, channel=None):
    """Routine legit activity: ~70% merchant purchases, ~30% ordinary person-to-person transfers (no merchant), so that
    'this trace has no merchant' does not by itself indicate fraud (v1 legit is ~25% transfers too)."""
    ev, t = [], t0
    for _ in range(n):
        t = t + timedelta(seconds=float(g.ln(gap_med, 1.0)))
        status = "failed" if g.rng.random() < 0.03 else "completed"
        if g.rng.random() < 0.30:
            payee = g.pick(pools.popular_payees) if g.rng.random() < 0.2 else g.uid()
            ev.append(e_txn(g, t, cust, acct, g.ln(amt_med * 1.2, 0.9), "transfer", status,
                            channel or g.pick(["mobile", "online"]), beneficiary_id=payee))
            continue
        mid, mage = g.pick(pools.legit_merchants)
        if g.rng.random() < 0.15:
            mid, mage = g.pick(pools.young_legit_merchants)
        ev.append(e_txn(g, t, cust, acct, g.ln(amt_med, 0.8), "purchase", status,
                        channel or g.pick(["mobile", "online"]), merchant_id=mid, merchant_age=mage))
    return ev, t


def legit_trace(g: G, pools: Pools, kind: str, k: int):
    cust, acct, sess, dev = g.uid(), g.uid(), g.uid(), g.uid()
    age = int(g.ln(700, 0.9)) + 30
    if g.rng.random() < pools.young_share:           # genuinely new (but real) customers
        age = int(g.rng.integers(5, 60))
    t0 = BASE + timedelta(seconds=float(g.rng.uniform(0, 30 * 86400)))
    ev = [e_login(g, t0, cust, acct, sess, dev, age)]
    if kind == "routine":
        txs, _ = _routine_txns(g, t0, cust, acct, int(g.rng.integers(1, 5)), pools)
        ev += txs
    elif kind == "H1_bill_burst":
        t = t0 + timedelta(seconds=float(g.ln(120, 0.6)))
        for _ in range(int(g.rng.integers(3, 7))):
            t += timedelta(seconds=float(g.ln(45, 0.7)))
            ev.append(e_txn(g, t, cust, acct, g.ln(700, 0.6), "purchase", "completed", g.pick(["mobile", "online"]),
                            *g.pick(pools.legit_merchants)))
    elif kind == "H2_new_payee_then_pay":
        b = g.pick(pools.popular_payees) if g.rng.random() < 0.3 else g.uid()
        t = t0 + timedelta(seconds=float(g.ln(150, 0.7)))
        ev.append(e_benef(g, t, cust, b, g.rng.random() < 0.7))
        t += timedelta(seconds=float(g.ln(480, 1.0)))
        for _ in range(int(g.rng.integers(1, 3))):
            ev.append(e_txn(g, t, cust, acct, g.ln(1500, 0.8), "transfer", "completed", g.pick(["mobile", "online"]),
                            beneficiary_id=b))
            t += timedelta(seconds=float(g.ln(300, 0.7)))
    elif kind == "H3_new_phone":
        t = t0 - timedelta(seconds=float(g.ln(60, 0.5)))
        ev.append(e_devreg(g, t, cust, dev))
        txs, _ = _routine_txns(g, t0 + timedelta(seconds=float(g.ln(900, 0.9))), cust, acct,
                               int(g.rng.integers(1, 4)), pools, gap_med=600)
        ev += txs
    elif kind == "H4_retry":
        t, amt = t0 + timedelta(seconds=float(g.ln(150, 0.6))), g.ln(1200, 0.7)
        for _ in range(int(g.rng.integers(1, 4))):
            ev.append(e_txn(g, t, cust, acct, amt, "purchase", "failed", g.pick(["mobile", "online"]),
                            *g.pick(pools.legit_merchants)))
            t += timedelta(seconds=float(g.ln(30, 0.6)))
            amt *= float(g.rng.choice([1.0, 1.0, 0.8]))
        ev.append(e_txn(g, t, cust, acct, amt, "purchase", "completed", g.pick(["mobile", "online"]),
                        *g.pick(pools.legit_merchants)))
    elif kind == "H5_large_oneoff":
        t = t0 + timedelta(seconds=float(g.ln(200, 0.8)))
        ev.append(e_txn(g, t, cust, acct, g.ln(9000, 0.5), "transfer", "completed", g.pick(["mobile", "online"]),
                        beneficiary_id=g.uid()))
    elif kind == "H6_genuine_onboarding":
        age = int(g.rng.integers(0, 2))
        t_open = t0 - timedelta(seconds=float(g.ln(600, 0.5)))
        ev.append(e_open(g, t_open, cust, g.uid(), g.uid()))
        ev.append(e_kyc(g, t_open + timedelta(seconds=30), cust, g.rng.normal(0.92, 0.04), g.rng.normal(0.93, 0.04),
                        g.rng.normal(0.90, 0.05), g.rng.random() < 0.02, g.rng.choice([0, 0, 1, 2]),
                        g.ln(150, 0.4)))
        ev.append(e_devreg(g, t_open, cust, dev))
        txs, _ = _routine_txns(g, t0 + timedelta(seconds=float(g.ln(1200, 0.8))), cust, acct, int(g.rng.integers(1, 3)), pools, amt_med=700)
        ev += txs
        ev[0] = e_login(g, t0, cust, acct, sess, dev, age)
    elif kind == "H7_household":
        if not pools.households or g.rng.random() < 0.25:
            pools.households.append((g.uid(), g.uid()))
        hdev, haddr = pools.households[-1] if g.rng.random() < 0.7 else g.pick(pools.households)
        ev[0] = e_login(g, t0, cust, acct, sess, hdev, age)
        ev.append(e_open(g, t0 - timedelta(seconds=5), cust, g.uid(), haddr))   # same address as housemates
        txs, _ = _routine_txns(g, t0, cust, acct, int(g.rng.integers(1, 4)), pools)
        ev += txs
    elif kind == "H9_shared_office":
        if not pools.offices or pools.offices[-1][2] == 0:
            pools.offices.append([g.uid(), g.uid(), int(g.rng.integers(3, 9))])
        office = pools.offices[-1]
        office[2] -= 1
        ev[0] = e_login(g, t0, cust, acct, sess, office[0], age)
        ev.append(e_open(g, t0 - timedelta(seconds=5), cust, g.uid(), office[1]))
        txs, _ = _routine_txns(g, t0, cust, acct, int(g.rng.integers(1, 4)), pools)
        ev += txs
    elif kind == "H10_p2p_spree":
        t = t0 + timedelta(seconds=float(g.ln(200, 0.7)))
        for _ in range(int(g.rng.integers(2, 5))):
            t += timedelta(seconds=float(g.ln(300, 0.7)))
            ev.append(e_txn(g, t, cust, acct, g.ln(900, 0.7), "transfer", "failed" if g.rng.random() < 0.05 else "completed",
                            g.pick(["mobile", "online"]), beneficiary_id=g.uid()))
    elif kind == "H8_popular_payee":
        b = g.pick(pools.popular_payees)
        t = t0 + timedelta(seconds=float(g.ln(100, 0.6)))
        ev.append(e_benef(g, t, cust, b, True))
        t += timedelta(seconds=float(g.ln(200, 0.8)))
        ev.append(e_txn(g, t, cust, acct, g.ln(1100, 0.4), "transfer", "completed", g.pick(["mobile", "online"]),
                        beneficiary_id=b))
    else:
        raise ValueError(kind)
    return _trace("v2legit", "legitimate_v2" if kind == "routine" else f"legitimate_v2_{kind}", 0, cust, ev)


HARD_KINDS = ["H1_bill_burst", "H2_new_payee_then_pay", "H3_new_phone", "H4_retry", "H5_large_oneoff",
              "H7_household", "H8_popular_payee", "H9_shared_office", "H10_p2p_spree"]
HARD_WEIGHTS = np.array([8, 8, 5, 6, 4, 4, 4, 6, 8], float)


def gen_legit(g, pools, n, hard_share, onboarding_share=0.053):
    """onboarding_share: share of ALL legit traces that are genuine new-customer onboardings (KYC + account opening).
    In real life every new account has KYC, so 'has a KYC event' must not by itself point to fraud."""
    out = []
    for _ in range(n):
        if g.rng.random() < onboarding_share:
            out.append(legit_trace(g, pools, "H6_genuine_onboarding", 0)); continue
        if g.rng.random() < hard_share:
            kind = HARD_KINDS[int(g.rng.choice(len(HARD_KINDS), p=HARD_WEIGHTS / HARD_WEIGHTS.sum()))]
        else:
            kind = "routine"
        out.append(legit_trace(g, pools, kind, 0))
    return out


# ---------------------------------------------------------------- new fraud families
# stealth in [0,1] = probability that an individual attacker ADAPTS to look legitimate on the features that gave the
# family away (account age, shared entities, merchant age, KYC quality, payee reuse, timing, amounts). The core fraud
# behaviour (money moved out quickly, to a payee the customer does not know) is kept.
def gen_synthetic_identity(g, pools, n, stealth=0.0):
    out, farm = [], None
    for i in range(n):
        s = g.rng.random() < stealth
        if farm is None or farm["left"] == 0:
            farm = {"left": int(g.rng.integers(3, 8)), "dev": [g.uid() for _ in range(int(g.rng.integers(1, 3)))], "addr": g.uid()}
        farm["left"] -= 1
        cust, acct, sess = g.uid(), g.uid(), g.uid()
        dev = g.uid() if s else (g.pick(farm["dev"]) if g.rng.random() < 0.85 else g.uid())
        addr = g.uid() if s else (farm["addr"] if g.rng.random() < 0.9 else g.uid())
        t0 = BASE + timedelta(seconds=float(g.rng.uniform(0, 30 * 86400)))
        age = int(g.ln(300, 0.7)) + 40 if s else int(g.rng.integers(5, 45))      # "aged" identity when stealthy
        ev = [e_login(g, t0, cust, acct, sess, dev, age)]
        if not s:     # an identity that has been "aged" opened its account long before this episode (not in the trace)
            ev.append(e_open(g, t0 - timedelta(seconds=20), cust, g.uid(), addr))
        t = t0
        n_tx = int(g.rng.integers(2, 5)) if s else int(g.rng.integers(3, 8))
        gap, amt = (400, 1800) if s else (70, 3000)
        for _ in range(n_tx):
            t += timedelta(seconds=float(g.ln(gap, 0.7)))
            ev.append(e_txn(g, t, cust, acct, g.ln(amt, 0.6), "transfer", "failed" if g.rng.random() < 0.08 else "completed",
                            g.pick(["mobile", "online"]), beneficiary_id=g.uid()))
        out.append(_trace("v2syn", "SYNTHETIC_IDENTITY", 1, cust, ev, "stealth" if s else "plain"))
    return out


def gen_fake_merchant(g, pools, n, stealth=0.0):
    out = []
    for _ in range(n):
        s = g.rng.random() < stealth
        cust, acct, sess, dev = g.uid(), g.uid(), g.uid(), g.uid()
        age = int(g.ln(700, 0.9)) + 30
        t0 = BASE + timedelta(seconds=float(g.rng.uniform(0, 30 * 86400)))
        if s:   # storefront operating through a compromised / established merchant id
            mid, mage = g.pick(pools.legit_merchants)
        else:
            mid, mage = g.pick(pools.fake_merchants)
        ev = [e_login(g, t0, cust, acct, sess, dev, age)]
        t = t0
        for _ in range(int(g.rng.integers(1, 4))):
            t += timedelta(seconds=float(g.ln(180 if not s else 600, 0.7)))
            ev.append(e_txn(g, t, cust, acct, g.ln(700, 0.6), "purchase", "failed" if g.rng.random() < 0.10 else "completed",
                            g.pick(["mobile", "online"]) if s else "online", mid, mage))
        out.append(_trace("v2fm", "FAKE_MERCHANT", 1, cust, ev, "stealth" if s else "plain"))
    return out


def gen_deepfake_kyc(g, pools, n, stealth=0.0):
    out = []
    for _ in range(n):
        s = g.rng.random() < stealth
        cust, acct, sess, dev = g.uid(), g.uid(), g.uid(), g.uid()
        t0 = BASE + timedelta(seconds=float(g.rng.uniform(0, 30 * 86400)))
        t_open = t0 - timedelta(seconds=float(g.ln(400, 0.5)))
        if s:   # high-quality deepfake: KYC telemetry indistinguishable from a genuine applicant
            kyc = e_kyc(g, t_open + timedelta(seconds=30), cust, g.rng.normal(0.92, 0.04), g.rng.normal(0.93, 0.04),
                        g.rng.normal(0.90, 0.05), g.rng.random() < 0.02, g.rng.choice([0, 0, 1, 2]), g.ln(150, 0.4))
            delay = 1200
        else:
            kyc = e_kyc(g, t_open + timedelta(seconds=20), cust, g.rng.normal(0.84, 0.05), g.rng.normal(0.88, 0.05),
                        g.rng.normal(0.83, 0.04), g.rng.random() < 0.35, g.rng.choice([1, 2, 3, 4]), g.ln(60, 0.5))
            delay = 400
        ev = [e_open(g, t_open, cust, g.uid(), g.uid()), kyc, e_devreg(g, t_open, cust, dev),
              e_login(g, t0, cust, acct, sess, dev, int(g.rng.integers(0, 2)))]
        t = t0
        for _ in range(int(g.rng.integers(1, 4))):
            t += timedelta(seconds=float(g.ln(delay, 0.9)))
            ev.append(e_txn(g, t, cust, acct, g.ln(3500 if not s else 1500, 0.5), "transfer", "completed",
                            g.pick(["mobile", "online"]), beneficiary_id=g.uid()))
        out.append(_trace("v2dk", "DEEPFAKE_KYC", 1, cust, ev, "stealth" if s else "plain"))
    return out


def gen_ai_scripted_app(g, pools, n, stealth=0.0):
    out = []
    for _ in range(n):
        s = g.rng.random() < stealth
        cust, acct, sess, dev = g.uid(), g.uid(), g.uid(), g.uid()
        age = int(g.ln(700, 0.9)) + 30
        t0 = BASE + timedelta(seconds=float(g.rng.uniform(0, 30 * 86400)))
        b = g.uid() if s else g.pick(pools.scam_payees)        # stealth: fresh mule account per victim (no payee reuse)
        ev = [e_login(g, t0, cust, acct, sess, dev, age)]
        t = t0 + timedelta(seconds=float(g.ln(150, 0.7)))
        ev.append(e_benef(g, t, cust, b, g.rng.random() < (0.6 if s else 0.3)))
        t += timedelta(seconds=float(g.ln(600, 0.9)))
        amt = g.ln(2000 if s else 4000, 0.5)
        for _ in range(int(g.rng.integers(1, 3) if s else g.rng.integers(2, 6))):
            failed = g.rng.random() < 0.25
            ev.append(e_txn(g, t, cust, acct, amt, "transfer", "failed" if failed else "completed", g.pick(["mobile", "online"]),
                            beneficiary_id=b))
            if failed:
                amt *= 0.8
            t += timedelta(seconds=float(g.ln(240, 0.8)))
        out.append(_trace("v2aia", "AI_SCRIPTED_APP", 1, cust, ev, "stealth" if s else "plain"))
    return out


def build_v2_records(seed=2026, n_legit=900, hard_share=0.45, n_per_family=90, stealth=0.0,
                     onboarding_share=0.053, young_share=0.0):
    """Defaults reproduce the FIRST v2 world (stealth 0). Harder world used for evaluation:
    stealth=0.5, onboarding_share=0.20, young_share=0.12 (see features.WORLD_B_HARD)."""
    g = G(seed)
    pools = Pools(g, young_share)
    recs = gen_legit(g, pools, n_legit, hard_share, onboarding_share)
    recs += gen_synthetic_identity(g, pools, n_per_family, stealth)
    recs += gen_fake_merchant(g, pools, n_per_family, stealth)
    recs += gen_deepfake_kyc(g, pools, n_per_family, stealth)
    recs += gen_ai_scripted_app(g, pools, n_per_family, stealth)
    return recs
