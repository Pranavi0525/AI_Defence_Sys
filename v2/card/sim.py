"""v2/card/sim.py -- synthetic card-transaction simulator (SYNTHETIC; all patterns and rates are my assumptions).

Each row is one authorisation: card, time, amount, MCC, channel (POS/ECOM/ATM), location, merchant, status.
For ECOM the location is the MERCHANT's location (as in real card-not-present data), which is what makes a naive
"two cities in 10 minutes" rule fire on genuine customers.
Fraud patterns (6). Two of them (SLOW_DRIP, LIMIT_SPLIT) are written to stay under the thresholds of ordinary rules,
by my design: read results with that in mind.

EXTENDED MODE (`simulate(..., extended=True)`, used by the served-model training pipeline)
    The original simulator never produced a transaction above ~18,000 and had almost no legitimate large purchases, so a tree
    model trained on it could not respond to large amounts at all (every amount above ~10,000 scored identically; see
    reports/card/large_amount_investigation.md). Extended mode adds, from a SEPARATE random stream so that `extended=False`
    stays bit-identical to the original:
      * LEGIT big-ticket purchases (electronics, jewellery, travel, furniture ...), including a few "premium" cards that
        legitimately spend much more; amounts are heavy-tailed up to `max_amount` (INR).
      * "new cards": some cards start mid-window, so rows with no / very little history exist in both classes.
      * NO injection-timing artifact. The original code starts every fraud event 10-30 minutes after one of the card's own
        transactions; 91% of fraud events then begin with a gap in that window versus 2.9% of legitimate rows, a shortcut worth
        AUC 0.946 on its own. Extended mode starts events after a gap drawn from the card's own inter-arrival distribution instead.
        (Bursts inside an event, e.g. card testing, keep their designed short gaps: that is the velocity signal, not the artifact.)
        Which patterns are re-timed: the six base patterns and FAMILIAR_MERCHANT_DRAIN (see RETIMED_PATTERNS), all through one helper,
        `_retime_event`. The other two novel patterns are NOT re-timed on purpose, because they never had the anchor + 10-30 min gap and
        their timing is part of what they are: DORMANT_REACTIVATION starts at an absolute time after a deleted 15-day window, and
        ODD_HOURS_QUIET starts at 02:00-02:30 clock time (a shift would move it out of the odd hours). The EXTENDED patterns are built
        with an organic gap from the start (`_extended_fraud`).
      * six more fraud scenarios (EXTENDED below). Their amounts deliberately OVERLAP the legitimate big-ticket range, and
        some of them (SPEND_SPIKE_FAMILIAR) happen in an otherwise perfectly ordinary context, so no single feature
        separates the classes.
    All amounts are INR. Every rate, ratio and scale here is the author's ASSUMPTION about what such behaviour looks like;
    none of it is measured from real transactions.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .limits import DEFAULT_SIM_MAX_AMOUNT

CITIES = {"Chennai": (13.08, 80.27), "Bengaluru": (12.97, 77.59), "Mumbai": (19.08, 72.88), "Delhi": (28.61, 77.21),
          "Hyderabad": (17.39, 78.49), "Kolkata": (22.57, 88.36), "Pune": (18.52, 73.86), "Ahmedabad": (23.02, 72.57),
          "Jaipur": (26.91, 75.79), "Kochi": (9.93, 76.27), "Lucknow": (26.85, 80.95), "Surat": (21.17, 72.83)}
CITY_NAMES = list(CITIES)
MCCS = [5411, 5812, 5541, 5311, 5651, 4111, 5912, 5814, 5732, 5947, 5815, 5944, 7011, 5200]   # grocery..hardware
RISKY_MCC = [5732, 5947, 5815, 5944]                                                           # electronics, gift cards, digital goods, jewellery
PATTERNS = ["CLONED_CARD_POS", "CARD_TESTING", "CNP_SPREE", "ATM_CASHOUT", "SLOW_DRIP", "LIMIT_SPLIT"]
# footprints deliberately DIFFERENT from the six above: familiar merchant/category, home city, no new-merchant signal
NOVEL = ["DORMANT_REACTIVATION", "FAMILIAR_MERCHANT_DRAIN", "ODD_HOURS_QUIET"]
# Patterns whose first transaction is placed at (one of the card's own transactions) + U(600, 1800) s, i.e. the injection artifact that
# `extended=True` removes. NOT in here, deliberately: DORMANT_REACTIVATION (absolute start time + deleted dormant window) and
# ODD_HOURS_QUIET (02:00-02:30 clock time); neither has the 10-30 minute gap and a shift would break what the pattern means.
RETIMED_PATTERNS = tuple(PATTERNS) + ("FAMILIAR_MERCHANT_DRAIN",)
# extended-mode fraud scenarios (see module docstring)
EXTENDED = ["HIGH_VALUE_BUSTOUT", "SPEND_SPIKE_FAMILIAR", "RAPID_REPEAT", "UNUSUAL_MERCHANT", "DECLINE_THEN_APPROVE", "NEW_CARD_FIRST_USE"]
BIG_TICKET_MCC = [5732, 5944, 5311, 7011, 4511, 5712, 5200]      # electronics, jewellery, dept store, lodging, airlines, furniture, hardware
UNUSUAL_MCC = [5733, 5999, 4814, 5942, 5993, 7995, 5967]         # categories a given card has usually never used
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



def _organic_gap(rng2, card_profile):
    """Time from the anchor transaction to the start of an injected event, drawn like a genuine inter-arrival gap for this card
    (exponential with the card's own rate), so the gap itself carries no information about fraud."""
    return float(max(60.0, rng2.exponential(DAY / card_profile["rate"])))


def _retime_event(rows, start, anchor_ts, gap):
    """Extended mode: move the event stored in rows[start:] so that its FIRST transaction happens `gap` seconds after `anchor_ts` (the
    timestamp of the card's own transaction the event was anchored to). Every row moves by the same amount, so the event's internal
    gaps (the designed velocity signal), amounts, channels, merchants and labels are untouched. This is the only place the
    original 10-30 minute injection gap is replaced."""
    shift = (anchor_ts + gap) - rows[start][1]
    rows[start:] = [(r[0], r[1] + shift) + tuple(r[2:]) for r in rows[start:]]


def _ext_cap(a, max_amount):
    return float(np.round(min(max(a, 1.0), max_amount), 2))


def _extended_legit(legit, cards, rng2, days, hours_w, max_amount):
    """Extended mode: add legitimate big-ticket purchases and turn ~8% of cards into 'new cards' (activity starts mid-window).
    Draws ONLY from rng2. Returns (legit_frame, {card: first_activity_ts}) for the new cards."""
    n_cards = len(cards)
    premium = rng2.random(n_cards) < 0.06                      # a few cards that legitimately spend a lot
    bt_rate = rng2.gamma(2.0, 0.025, n_cards) * np.where(premium, 1.5, 1.0)        # big-ticket purchases per day (~3 per card per 60 days)
    new_card = rng2.random(n_cards) < 0.08
    new_start = rng2.uniform(3, max(days - 10, 4), n_cards) * DAY
    extra = []
    for c, p in enumerate(cards):
        for _ in range(int(rng2.poisson(bt_rate[c] * days))):
            ts = int(rng2.uniform(0, days)) * DAY + int(rng2.choice(24, p=hours_w)) * 3600 + rng2.uniform(0, 3600)
            ch = "POS" if rng2.random() < 0.55 else "ECOM"
            mcc = int(rng2.choice(BIG_TICKET_MCC))
            # log-scale step above this card's usual purchase: mostly moderate (3x-11x), sometimes large (11x-37x), premium cards far larger
            off = rng2.uniform(3.4, 5.0) if premium[c] else (rng2.uniform(1.2, 2.4) if rng2.random() < 0.7 else rng2.uniform(2.4, 3.6))
            amt = _ext_cap(np.exp(rng2.normal(p["mu"] + off, 0.5)), max_amount)
            merch = f"M{mcc}_{p['merch'][rng2.integers(6)] if rng2.random() < 0.35 else rng2.integers(0, 40)}"
            city = p["home"]
            if ch == "ECOM" and rng2.random() < 0.35:
                city = int(rng2.integers(len(CITY_NAMES)))
            lat, lon = _loc(rng2, city)
            extra.append((c, ts, amt, mcc, ch, city, lat, lon, merch, int(rng2.random() < 0.02), "LEGIT"))
    if extra:
        legit = pd.concat([legit, pd.DataFrame(extra, columns=legit.columns)], ignore_index=True)
    starts = {c: float(new_start[c]) for c in range(n_cards) if new_card[c]}
    if starts:
        drop = np.zeros(len(legit), bool)
        for c, t in starts.items():
            drop |= ((legit["card"] == c) & (legit["ts"] < t)).to_numpy()
        legit = legit[~drop]
    legit = legit.sort_values(["card", "ts"]).reset_index(drop=True)
    return legit, starts


def _extended_fraud(rows, event_of, ev_id, by_card, cards, new_starts, rng2, n_cards, events_per_pattern, max_amount):
    """Extended mode: append the EXTENDED fraud scenarios to `rows` / `event_of`. Draws ONLY from rng2. Returns the next event id."""
    def add(card, ts, amt, mcc, ch, city, lat, lon, merch, declined, pat):
        rows.append((card, ts, amt, mcc, ch, city, lat, lon, merch, int(declined), pat))

    def fx(rng):
        return f"M{rng.integers(0, 40)}_fx{rng.integers(0, 15)}"

    for pat in EXTENDED:
        for _ in range(events_per_pattern):
            if pat == "NEW_CARD_FIRST_USE":
                pool = [c for c in new_starts if c in by_card]
                if not pool:
                    continue
                c = int(rng2.choice(pool)); g = by_card[c]
                t0 = max(new_starts[c], float(g["ts"].iloc[0]) + rng2.uniform(-4 * 3600, 30 * 3600))
            else:
                c = int(rng2.integers(n_cards)); g = by_card.get(c)
                if g is None or len(g) < 6:
                    continue
                base_i = int(rng2.integers(3, len(g)))
                t0 = float(g["ts"].iloc[base_i]) + _organic_gap(rng2, cards[c])
            p = cards[c]; typ = float(np.exp(p["mu"]))             # this card's usual purchase size (INR)
            start = len(rows); t = t0
            far = _far_city(rng2, p["home"])
            if pat == "HIGH_VALUE_BUSTOUT":
                city = p["home"] if rng2.random() < 0.4 else far
                goods_driven = rng2.random() < 0.5      # half: sized by the goods being bought (absolute); half: sized by the card's available limit,
                for _ in range(int(rng2.integers(1, 4))):     # which scales with its usual spend (relative). Both are ASSUMPTIONS.
                    lat, lon = _loc(rng2, city)
                    raw = np.exp(rng2.normal(np.log(40000), 0.9)) if goods_driven else typ * np.exp(rng2.uniform(np.log(20), np.log(400)))
                    amt = _ext_cap(max(raw, 8000), max_amount)
                    add(c, t, amt, int(rng2.choice(RISKY_MCC + BIG_TICKET_MCC)), "ECOM" if rng2.random() < 0.7 else "POS", city, lat, lon, fx(rng2), rng2.random() < 0.08, pat)
                    t += rng2.uniform(300, 3600)
            elif pat == "SPEND_SPIKE_FAMILIAR":
                prior = g.iloc[:base_i]; prior = prior[prior["channel"] != "ATM"]
                src = prior.iloc[int(rng2.integers(len(prior)))] if len(prior) else g.iloc[0]
                for _ in range(int(rng2.integers(1, 3))):
                    lat, lon = _loc(rng2, p["home"])
                    amt = _ext_cap(typ * np.exp(rng2.uniform(np.log(12), np.log(600))), max_amount)   # 12x .. 600x the usual purchase
                    add(c, t, amt, int(src["mcc"]), str(src["channel"]) if src["channel"] != "ATM" else "POS", p["home"], lat, lon, str(src["merchant"]), rng2.random() < 0.03, pat)
                    t += rng2.uniform(900, 7200)
            elif pat == "RAPID_REPEAT":
                city = p["home"] if rng2.random() < 0.6 else far
                mid, mcc, base = fx(rng2), int(rng2.choice(MCCS)), float(np.exp(rng2.uniform(np.log(800), np.log(6000))))
                for _ in range(int(rng2.integers(4, 10))):
                    lat, lon = _loc(rng2, city)
                    add(c, t, _ext_cap(base * rng2.uniform(0.9, 1.1), max_amount), mcc, "ECOM", city, lat, lon, mid, rng2.random() < 0.06, pat)
                    t += rng2.uniform(20, 150)
            elif pat == "UNUSUAL_MERCHANT":
                for _ in range(int(rng2.integers(1, 4))):
                    mcc = int(rng2.choice(UNUSUAL_MCC)); lat, lon = _loc(rng2, p["home"])
                    add(c, t, _ext_cap(typ * rng2.uniform(1.5, 6), max_amount), mcc, "POS" if rng2.random() < 0.5 else "ECOM", p["home"], lat, lon,
                        f"M{mcc}_u{rng2.integers(0, 15)}", rng2.random() < 0.05, pat)
                    t += rng2.uniform(600, 5400)
            elif pat == "DECLINE_THEN_APPROVE":
                city = p["home"] if rng2.random() < 0.5 else far
                mid, mcc = fx(rng2), int(rng2.choice(RISKY_MCC))
                for i in range(int(rng2.integers(3, 7))):
                    lat, lon = _loc(rng2, city)
                    add(c, t, _ext_cap(rng2.uniform(500, 3000) * (1 + 0.3 * i), max_amount), mcc, "ECOM", city, lat, lon, mid, True, pat)
                    t += rng2.uniform(20, 120)
                lat, lon = _loc(rng2, city)
                add(c, t, _ext_cap(np.exp(rng2.uniform(np.log(5000), np.log(80000))), max_amount), mcc, "ECOM", city, lat, lon, mid, False, pat)
            elif pat == "NEW_CARD_FIRST_USE":
                city = p["home"] if rng2.random() < 0.5 else far
                for _ in range(int(rng2.integers(1, 4))):
                    lat, lon = _loc(rng2, city)
                    amt = _ext_cap(max(np.exp(rng2.normal(np.log(25000), 0.9)), 5000), max_amount)
                    add(c, t, amt, int(rng2.choice(RISKY_MCC)), "ECOM", city, lat, lon, fx(rng2), rng2.random() < 0.1, pat)
                    t += rng2.uniform(300, 3600)
            event_of.extend([ev_id] * (len(rows) - start))
            ev_id += 1
    return ev_id


def simulate(n_cards=1500, days=60, seed=0, events_per_pattern=40, novel=False, extended=False, max_amount=None):
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
    rng2 = np.random.default_rng([seed, 20261027])          # separate stream: extended=False output is bit-identical to the original
    rng_novel = np.random.default_rng([seed, 20261027, 1])  # gaps for re-timed NOVEL patterns only, so novel=True moves no draw of any other pattern
    max_amount = float(DEFAULT_SIM_MAX_AMOUNT if max_amount is None else max_amount)
    new_starts = {}
    if extended:
        legit, new_starts = _extended_legit(legit, cards, rng2, days, hours_w, max_amount)
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
            if pat == "DORMANT_REACTIVATION":                       # absolute start time; the 15 days before it are emptied at the end. Never re-timed.
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
            elif pat == "ODD_HOURS_QUIET":                          # clock-time pattern (02:00-02:30 of the anchor's day). Never re-timed.
                night = int(t0 // DAY) * DAY + 2 * 3600 + rng.uniform(0, 1800)
                t = night
                for _ in range(int(rng.integers(4, 7))):
                    lat, lon = _loc(rng, p["home"]); k = int(rng.integers(5))
                    add(c, t, float(np.round(np.exp(rng.normal(p["mu"], 0.3)), 2)), int(p["mccs"][k]), "POS", p["home"], lat, lon,
                        f"M{int(p['mccs'][k])}_{p['merch'][k]}", rng.random() < 0.03, pat)
                    t += rng.uniform(900, 2400)
            if extended and pat in RETIMED_PATTERNS:               # remove the 10-30 minute injection artifact (see module docstring)
                _retime_event(rows, start, float(g["ts"].iloc[base_i]), _organic_gap(rng2 if pat in PATTERNS else rng_novel, p))
            for _ in range(len(rows) - start):
                event_of.append(ev_id)
            ev_id += 1
    if extended:
        ev_id = _extended_fraud(rows, event_of, ev_id, by_card, cards, new_starts, rng2, n_cards, events_per_pattern, max_amount)
    for c_, a_, b_ in dormant:
        legit = legit[~((legit.card == c_) & (legit.ts >= a_) & (legit.ts < b_))]
    fr = pd.DataFrame(rows, columns=legit.columns)
    fr["event"] = event_of
    legit["event"] = -1
    df = pd.concat([legit, fr], ignore_index=True)
    df = df[df["ts"] < days * DAY].sort_values(["card", "ts"]).reset_index(drop=True)
    df["is_fraud"] = (df["pattern"] != "LEGIT").astype(int)
    return df
