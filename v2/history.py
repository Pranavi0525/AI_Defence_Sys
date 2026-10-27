"""v2/history.py -- Step 5: give World B customers a PAST, so a per-customer baseline can be tested.

WHY: v2/gen.py makes every trace a brand-new customer, so "unusual for THIS customer" cannot be measured.
This module is a post-processing layer (v2/gen.py is untouched): it adds a `history` list of earlier events to each trace that
has one, and makes genuine customers' current behaviour consistent with their own habits.

EVERY CHOICE BELOW IS MY ASSUMPTION, not an empirical estimate. `strength` in [0,1] scales how much customers differ from each other:
  strength = 0 -> history carries NO information (amount scale, usual hour, repeat merchants/payees all uninformative). Control run.
  strength = 1 -> reference setting.
Assumptions (reference setting):
  * Habit scale: each customer has an amount multiplier s_c ~ lognormal(0, 0.6*strength); genuine sessions use it, history uses it.
  * Usual hour: login hour ~ N(h_c, 12-10*strength hours) for genuine sessions (strength 1 -> sd 2h); h_c uniform 8..22.
  * Repeat behaviour: each merchant in a genuine trace is also in that customer's history with prob 0.6*strength, each payee with 0.35*strength.
  * Attackers: scam amounts are chosen by the attacker, not tied to the victim's habit; victims never paid the fake merchant / scam payee before.
    STEALTH attackers (attack_difficulty == "stealth") adapt: they match the victim's amount scale and usual hour.
  * Cold start: a customer's number of past sessions ~ account_age_days/15 (cap 12). New accounts (deepfake KYC, genuine onboarding,
    plain synthetic identity) have none or almost none: a per-customer baseline CANNOT help them, by construction.
  * Stealth synthetic identities are "aged" identities and get a normal-looking history (an attacker can fake that).
History events are strictly BEFORE the trace's first event (checked in tests).
"""
from __future__ import annotations

import copy
import hashlib
from datetime import datetime, timedelta

import numpy as np

from v2.gen import FMT, G, e_login, e_txn

HOUR0, HOUR1 = 8.0, 22.0


def _rng_for(seed: int, key: str) -> np.random.Generator:
    h = int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:12], 16)
    return np.random.default_rng(h)


def _parse(ts: str) -> datetime:
    return datetime.strptime(ts, FMT)


def _hour(dt: datetime) -> float:
    return dt.hour + dt.minute / 60 + dt.second / 3600


def _shift_to_hour(events, target_h: float):
    """Shift ALL events by one constant so the first login's time-of-day equals target_h (smallest rotation, within +-12h)."""
    login = next(e for e in events if e["event_type"] == "SESSION_LOGIN")
    cur = _hour(_parse(login["timestamp"]))
    d = ((target_h - cur + 12) % 24) - 12
    for e in events:
        e["timestamp"] = (_parse(e["timestamp"]) + timedelta(hours=d)).strftime(FMT)


def _n_hist(age_days: float) -> int:
    return int(min(12, max(0, age_days) // 15))


def add_history(records: list[dict], seed: int = 7, strength: float = 1.0) -> list[dict]:
    """Return NEW records (inputs untouched). Adds trace['history'] (list of events) and adapts genuine/stealth behaviour."""
    recs = copy.deepcopy(records)
    g = G(seed)
    merchants = sorted({e["merchant_id"] for r in recs if r["fraud"] == 0 for e in r["events"] if e.get("merchant_id")})
    for r in recs:
        ev = r["events"]
        login = next((e for e in ev if e["event_type"] == "SESSION_LOGIN"), None)
        if login is None:
            r["history"] = []
            continue
        rng = _rng_for(seed, r["customer_id"])
        s_c = float(np.exp(rng.normal(0.0, 0.6 * strength)))
        h_c = float(rng.uniform(HOUR0, HOUR1))
        hour_sd = 12.0 - 10.0 * strength
        genuine = r["fraud"] == 0
        stealth = r.get("attack_difficulty") == "stealth"
        victim = r.get("attack_family") in ("FAKE_MERCHANT", "AI_SCRIPTED_APP")
        age = float(login.get("account_age_days") or 0)
        fam = r.get("attack_family")
        # which traces behave "in character" (their current session matches the customer's own habits)?
        in_character = genuine or (stealth and (victim or fam == "SYNTHETIC_IDENTITY"))
        # --- the usual device: the one used now, unless this very trace registers it as new
        reg_devs = {e["device_id"] for e in ev if e["event_type"] == "DEVICE_REGISTRATION"}
        usual_dev = login["device_id"] if (login["device_id"] not in reg_devs or fam == "SYNTHETIC_IDENTITY") else g.uid()
        # --- current session adaptation
        if in_character:
            for e in ev:
                if e["event_type"] == "TRANSACTION":
                    e["amount"] = f"{float(e['amount']) * s_c:.2f}"
            _shift_to_hour(ev, float((h_c + rng.normal(0, hour_sd)) % 24))
            r["events"] = ev = sorted(ev, key=lambda e: e["timestamp"])
            r["observation_window"] = [ev[0]["timestamp"], ev[-1]["timestamp"]]
        # --- history
        n = _n_hist(age)
        if fam == "SYNTHETIC_IDENTITY" and not stealth:
            n = min(n, 2)                                   # thin, not "aged"
        t_start = _parse(ev[0]["timestamp"])
        usual_m = list(rng.choice(merchants, size=min(4, len(merchants)), replace=False)) if merchants else []
        if genuine or (stealth and fam == "SYNTHETIC_IDENTITY"):
            for e in ev:                                    # repeat behaviour: current merchants/payees sometimes already known
                if e["event_type"] == "TRANSACTION":
                    if e.get("merchant_id") and rng.random() < 0.6 * strength:
                        usual_m.append(e["merchant_id"])
        known_payees = []
        if genuine:
            for e in ev:
                if e["event_type"] == "TRANSACTION" and e.get("beneficiary_id") and rng.random() < 0.35 * strength:
                    known_payees.append(e["beneficiary_id"])
        hist = []
        days = sorted(rng.uniform(1.0, 60.0, size=n), reverse=True)
        for d in days:
            day0 = (t_start - timedelta(days=float(d))).replace(hour=0, minute=0, second=0)
            hh = float((h_c + rng.normal(0, hour_sd)) % 24)
            t = day0 + timedelta(hours=hh)
            hist.append(e_login(g, t, r["customer_id"], login["account_id"], g.uid(), usual_dev, int(max(0, age - d)) , 1, True))
            for _ in range(int(rng.integers(1, 5))):
                t = t + timedelta(seconds=float(rng.lognormal(np.log(600), 0.8)))
                amt = float(rng.lognormal(np.log(900), 0.8)) * s_c
                if known_payees and rng.random() < 0.3:
                    hist.append(e_txn(g, t, r["customer_id"], login["account_id"], amt, "transfer", "completed", "mobile",
                                      beneficiary_id=known_payees[int(rng.integers(len(known_payees)))]))
                elif rng.random() < 0.3:
                    hist.append(e_txn(g, t, r["customer_id"], login["account_id"], amt, "transfer", "completed", "mobile",
                                      beneficiary_id=g.uid()))
                else:
                    m = usual_m[int(rng.integers(len(usual_m)))] if usual_m else g.uid()
                    hist.append(e_txn(g, t, r["customer_id"], login["account_id"], amt, "purchase", "completed", "mobile",
                                      merchant_id=m, merchant_age=int(rng.integers(30, 2000))))
        r["history"] = sorted([h for h in hist if _parse(h["timestamp"]) < t_start], key=lambda e: e["timestamp"])
    return recs
