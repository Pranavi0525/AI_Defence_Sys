"""v2/card/sim.py -- synthetic card-transaction simulator (SYNTHETIC; all patterns and rates are my assumptions).

Each row is one authorisation: card, time, amount, MCC, channel (POS/ECOM/ATM), location, merchant, status.
For ECOM the location is the MERCHANT's location (as in real card-not-present data), which is what makes a naive
"two cities in 10 minutes" rule fire on genuine customers.
Fraud patterns (6). Two of them (SLOW_DRIP, LIMIT_SPLIT) are written to stay under the thresholds of ordinary rules,
by my design: read results with that in mind.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

CITIES = {"Chennai": (13.08, 80.27), "Bengaluru": (12.97, 77.59), "Mumbai": (19.08, 72.88), "Delhi": (28.61, 77.21),
          "Hyderabad": (17.39, 78.49), "Kolkata": (22.57, 88.36), "Pune": (18.52, 73.86), "Ahmedabad": (23.02, 72.57),
          "Jaipur": (26.91, 75.79), "Kochi": (9.93, 76.27), "Lucknow": (26.85, 80.95), "Surat": (21.17, 72.83)}
CITY_NAMES = list(CITIES)
MCCS = [5411, 5812, 5541, 5311, 5651, 4111, 5912, 5814, 5732, 5947, 5815, 5944, 7011, 5200]   # grocery..hardware
RISKY_MCC = [5732, 5947, 5815, 5944]                                                           # electronics, gift cards, digital goods, jewellery
PATTERNS = ["CLONED_CARD_POS", "CARD_TESTING", "CNP_SPREE", "ATM_CASHOUT", "SLOW_DRIP", "LIMIT_SPLIT"]
# footprints deliberately DIFFERENT from the six above: familiar merchant/category, home city, no new-merchant signal
NOVEL = ["DORMANT_REACTIVATION", "FAMILIAR_MERCHANT_DRAIN", "ODD_HOURS_QUIET"]
DAY = 86400.0


def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(a))


def _far_city(rng, home, min_km=500):
    h = CITIES[CITY_NAMES[home]]
    ok = [i for i, c in enumerate(CITY_NAMES) if haversine_km(h[0], h[1], *CITIES[c]) > min_km]
    return int(rng.choice(ok))


def _loc(rng, city):
    la, lo = CITIES[CITY_NAMES[city]]
    return la + rng.normal(0, 0.05), lo + rng.normal(0, 0.05)


def simulate(n_cards=1500, days=60, seed=0, events_per_pattern=40, novel=False):
    rng = np.random.default_rng(seed)
    hours_w = np.array([1, 1, 1, 1, 1, 2, 3, 5, 6, 7, 7, 7, 8, 8, 7, 7, 7, 8, 9, 9, 8, 6, 3, 2], float)
    hours_w /= hours_w.sum()
    rows = []

    def add(card, ts, amt, mcc, ch, city, lat, lon, merch, declined, pat):
        rows.append((card, ts, amt, mcc, ch, city, lat, lon, merch, int(declined), pat))

    cards = []
    for c in range(n_cards):
        home = int(rng.integers(len(CITY_NAMES)))
        prof = dict(home=home, mu=rng.normal(6.4, 0.5), sd=rng.uniform(0.4, 0.8), rate=float(np.clip(rng.lognormal(0.1, 0.5), 0.3, 4)),
                    mccs=rng.choice(MCCS, 5, replace=False), mw=rng.dirichlet(np.ones(5)),
                    ch=rng.dirichlet([5.5, 3.5, 1.0]), merch=rng.integers(0, 40, 6),
                    trip=(int(rng.integers(len(CITY_NAMES))), float(rng.uniform(10, days - 5))) if rng.random() < 0.25 else None)
        cards.append(prof)
        n = rng.poisson(prof["rate"] * days)
        for _ in range(n):
            day = rng.uniform(0, days)
            hour = rng.choice(24, p=hours_w)
            ts = int(day) * DAY + hour * 3600 + rng.uniform(0, 3600)
            ch = ["POS", "ECOM", "ATM"][rng.choice(3, p=prof["ch"])]
            mcc = int(rng.choice(prof["mccs"], p=prof["mw"])) if rng.random() > 0.07 else int(rng.choice(MCCS))
            merch = f"M{mcc}_{prof['merch'][rng.integers(6)] if rng.random() > 0.2 else rng.integers(0, 40)}"
            city = prof["home"]
            if prof["trip"] and prof["trip"][1] <= ts / DAY <= prof["trip"][1] + 3 and ch != "ECOM":
                city = prof["trip"][0]
            if ch == "ECOM" and rng.random() < 0.35:
                city = int(rng.integers(len(CITY_NAMES)))
            lat, lon = _loc(rng, city)
            amt = float(np.round(np.exp(rng.normal(prof["mu"], prof["sd"])), 2))
            if ch == "ATM":
                amt = float(rng.choice([500, 1000, 2000, 3000, 5000, 10000], p=[.1, .15, .25, .2, .2, .1]))
            add(c, ts, amt, mcc, ch, city, lat, lon, merch, rng.random() < 0.02, "LEGIT")

    legit = pd.DataFrame(rows, columns=["card", "ts", "amount", "mcc", "channel", "city", "lat", "lon", "merchant", "declined", "pattern"])
    legit = legit.sort_values(["card", "ts"]).reset_index(drop=True)
    by_card = {c: g for c, g in legit.groupby("card")}
    rows.clear()

    ev_id = 0
    event_of = []
    dormant = []                        # (card, from_ts, to_ts): legit activity removed to make the card genuinely dormant
    for pat in PATTERNS + (NOVEL if novel else []):
        for _ in range(events_per_pattern):
            c = int(rng.integers(n_cards))
            g = by_card.get(c)
            if g is None or len(g) < 6:
                continue
            p = cards[c]
            if pat == "DORMANT_REACTIVATION":
                t0 = float(rng.uniform(25, days - 1) * DAY); dormant.append((c, t0 - 15 * DAY, t0))
            base_i = int(rng.integers(3, len(g)))
            t0 = float(g["ts"].iloc[base_i]) + rng.uniform(600, 1800) if pat != "DORMANT_REACTIVATION" else t0
            start = len(rows)
            if pat == "CLONED_CARD_POS":
                city = _far_city(rng, p["home"])
                t = t0
                for _ in range(int(rng.integers(3, 7))):
                    lat, lon = _loc(rng, city)
                    add(c, t, float(np.round(np.exp(rng.normal(7.8, 0.5)), 2)), int(rng.choice(RISKY_MCC + [5541])), "POS", city, lat, lon,
                        f"M{rng.integers(0, 40)}_fx{rng.integers(0, 15)}", rng.random() < 0.1, pat)
                    t += rng.uniform(180, 1200)
            elif pat == "CARD_TESTING":
                city = int(rng.integers(len(CITY_NAMES)))
                t = t0
                for _ in range(int(rng.integers(4, 11))):
                    lat, lon = _loc(rng, city)
                    add(c, t, float(rng.uniform(1, 20)), int(rng.choice(MCCS)), "ECOM", city, lat, lon, f"M0_t{rng.integers(0, 6)}", rng.random() < 0.5, pat)
                    t += rng.uniform(10, 90)
                for _ in range(int(rng.integers(1, 3))):
                    lat, lon = _loc(rng, city)
                    add(c, t, float(rng.uniform(3000, 9000)), int(rng.choice(RISKY_MCC)), "ECOM", city, lat, lon, f"M0_t{rng.integers(0, 6)}", False, pat)
                    t += rng.uniform(30, 300)
            elif pat == "CNP_SPREE":
                city = int(rng.integers(len(CITY_NAMES)))
                t = t0
                for _ in range(int(rng.integers(3, 8))):
                    lat, lon = _loc(rng, city)
                    add(c, t, float(np.round(np.exp(rng.normal(8.3, 0.4)), 2)), int(rng.choice(RISKY_MCC)), "ECOM", city, lat, lon,
                        f"M{rng.integers(0, 40)}_fx{rng.integers(0, 15)}", rng.random() < 0.08, pat)
                    t += rng.uniform(300, 2400)
            elif pat == "ATM_CASHOUT":
                city = _far_city(rng, p["home"])
                t = t0
                for _ in range(int(rng.integers(3, 7))):
                    lat, lon = _loc(rng, city)
                    add(c, t, 10000.0, 6011, "ATM", city, lat, lon, f"ATM_fx{rng.integers(0, 10)}", rng.random() < 0.1, pat)
                    t += rng.uniform(300, 900)
            elif pat == "SLOW_DRIP":
                t = t0
                for _ in range(int(rng.integers(3, 6))):
                    lat, lon = _loc(rng, p["home"])
                    add(c, t, float(np.round(np.exp(rng.normal(7.4, 0.3)), 2)), int(rng.choice(RISKY_MCC)), "ECOM", p["home"], lat, lon,
                        f"M{rng.integers(0, 40)}_fx{rng.integers(0, 15)}", rng.random() < 0.05, pat)
                    t += rng.uniform(3 * 3600, 8 * 3600)
            elif pat == "LIMIT_SPLIT":
                t = t0
                mid = f"M{rng.integers(0, 40)}_fx{rng.integers(0, 15)}"
                mc = int(rng.choice(RISKY_MCC))
                for _ in range(int(rng.integers(4, 9))):
                    lat, lon = _loc(rng, p["home"])
                    add(c, t, float(rng.uniform(4800, 4990)), mc, str(rng.choice(["POS", "ECOM"])), p["home"], lat, lon, mid, rng.random() < 0.05, pat)
                    t += rng.uniform(1200, 5400)
            elif pat == "DORMANT_REACTIVATION":
                for _ in range(int(rng.integers(1, 3))):
                    lat, lon = _loc(rng, p["home"])
                    add(c, t0, float(np.round(np.exp(rng.normal(p["mu"] + 1.8, 0.3)), 2)), int(rng.choice(p["mccs"])), "POS", p["home"], lat, lon,
                        f"M{int(p['mccs'][0])}_{p['merch'][0]}", rng.random() < 0.03, pat)
                    t0 += rng.uniform(600, 3600)
            elif pat == "FAMILIAR_MERCHANT_DRAIN":
                t = t0
                k = int(rng.integers(0, 6)); mc = int(p["mccs"][k % 5]); mid = f"M{mc}_{p['merch'][k]}"
                for _ in range(int(rng.integers(4, 8))):
                    lat, lon = _loc(rng, p["home"])
                    add(c, t, float(np.round(np.exp(rng.normal(p["mu"] + 0.9, 0.2)), 2)), mc, str(rng.choice(["POS", "ECOM"])), p["home"], lat, lon, mid,
                        rng.random() < 0.03, pat)
                    t += rng.uniform(3600, 6 * 3600)
            elif pat == "ODD_HOURS_QUIET":
                night = int(t0 // DAY) * DAY + 2 * 3600 + rng.uniform(0, 1800)
                t = night
                for _ in range(int(rng.integers(4, 7))):
                    lat, lon = _loc(rng, p["home"]); k = int(rng.integers(5))
                    add(c, t, float(np.round(np.exp(rng.normal(p["mu"], 0.3)), 2)), int(p["mccs"][k]), "POS", p["home"], lat, lon,
                        f"M{int(p['mccs'][k])}_{p['merch'][k]}", rng.random() < 0.03, pat)
                    t += rng.uniform(900, 2400)
            for _ in range(len(rows) - start):
                event_of.append(ev_id)
            ev_id += 1
    for c_, a_, b_ in dormant:
        legit = legit[~((legit.card == c_) & (legit.ts >= a_) & (legit.ts < b_))]
    fr = pd.DataFrame(rows, columns=legit.columns)
    fr["event"] = event_of
    legit["event"] = -1
    df = pd.concat([legit, fr], ignore_index=True)
    df = df[df["ts"] < days * DAY].sort_values(["card", "ts"]).reset_index(drop=True)
    df["is_fraud"] = (df["pattern"] != "LEGIT").astype(int)
    return df
