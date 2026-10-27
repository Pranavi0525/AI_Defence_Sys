"""v2/card/features.py -- CAUSAL per-transaction features: current transaction + that card's PRIOR history only.

Never reads `pattern`, `event`, `is_fraud`, or the current transaction's own `declined` flag (the outcome of the very
decision being made). Prior declines are known at scoring time and are used.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .sim import haversine_km

FEATURES = ["amount", "log_amount", "is_pos", "is_ecom", "is_atm", "hour_sin", "hour_cos", "hist_len", "gap_s", "dist_prev_km", "speed_prev_kmh",
            "dist_cp_km", "gap_cp_s", "speed_cp_kmh", "cnt_10m", "cnt_1h", "cnt_24h", "sum_24h", "dec_10m", "dec_1h", "atm_1h", "same_merch_24h",
            "amt_z", "amt_ratio_max", "new_mcc", "new_merchant", "dist_home_km"]
# card-relative subset for the per-card novelty model (what is unusual FOR THIS CARD)
RELATIVE = ["amt_z", "amt_ratio_max", "new_mcc", "new_merchant", "dist_home_km", "speed_cp_kmh", "gap_s", "cnt_10m", "cnt_1h", "cnt_24h",
            "dec_10m", "dec_1h", "atm_1h", "same_merch_24h", "is_ecom", "is_atm", "log_amount"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["card", "ts"]).reset_index(drop=True)
    out = {k: np.full(len(df), np.nan) for k in FEATURES}
    pos = 0
    for _, g in df.groupby("card", sort=False):
        n = len(g)
        ts = g["ts"].to_numpy(dtype=np.float64)
        amt = g["amount"].to_numpy(dtype=np.float64)
        la = g["lat"].to_numpy(dtype=np.float64)
        lo = g["lon"].to_numpy(dtype=np.float64)
        ch = g["channel"].to_numpy(); mcc = g["mcc"].to_numpy(); mer = g["merchant"].to_numpy(); dec = g["declined"].to_numpy()
        dcum = np.concatenate([[0], np.cumsum(dec)])               # exclusive prefix sums
        acum = np.concatenate([[0], np.cumsum(amt)])
        atm = (ch == "ATM").astype(int); atmc = np.concatenate([[0], np.cumsum(atm)])
        la_ = np.log1p(amt); s1 = s2 = 0.0
        seen_mcc, seen_mer, mer_ts = set(), set(), {}
        cp_lat = cp_lon = 0.0; cp_n = 0; last_cp = None; mx = 0.0
        for i in range(n):
            r = pos + i
            out["amount"][r] = amt[i]; out["log_amount"][r] = la_[i]
            out["is_pos"][r] = ch[i] == "POS"; out["is_ecom"][r] = ch[i] == "ECOM"; out["is_atm"][r] = ch[i] == "ATM"
            h = (ts[i] % 86400) / 3600
            out["hour_sin"][r] = np.sin(2 * np.pi * h / 24); out["hour_cos"][r] = np.cos(2 * np.pi * h / 24)
            out["hist_len"][r] = i
            if i > 0:
                gap = ts[i] - ts[i - 1]; d = haversine_km(la[i - 1], lo[i - 1], la[i], lo[i])
                out["gap_s"][r] = gap; out["dist_prev_km"][r] = d; out["speed_prev_kmh"][r] = min(d / max(gap, 60) * 3600, 5000)
            else:
                out["gap_s"][r] = 3e6; out["dist_prev_km"][r] = 0; out["speed_prev_kmh"][r] = 0
            if ch[i] != "ECOM" and last_cp is not None:                   # card-present: compare with last card-present txn
                gap = ts[i] - ts[last_cp]; d = haversine_km(la[last_cp], lo[last_cp], la[i], lo[i])
                out["dist_cp_km"][r] = d; out["gap_cp_s"][r] = gap; out["speed_cp_kmh"][r] = min(d / max(gap, 60) * 3600, 5000)
            else:
                out["dist_cp_km"][r] = 0; out["gap_cp_s"][r] = 3e6; out["speed_cp_kmh"][r] = 0
            for W, name in ((600, "10m"), (3600, "1h"), (86400, "24h")):
                lo_i = int(np.searchsorted(ts, ts[i] - W, "left")); lo_i = min(lo_i, i)
                out["cnt_" + name][r] = i - lo_i
                if name == "24h": out["sum_24h"][r] = acum[i] - acum[lo_i]
                if name in ("10m", "1h"): out["dec_" + name][r] = dcum[i] - dcum[lo_i]
                if name == "1h": out["atm_1h"][r] = atmc[i] - atmc[lo_i]
            out["same_merch_24h"][r] = sum(1 for t in mer_ts.get(mer[i], []) if ts[i] - t <= 86400)
            if i >= 5:
                m = s1 / i; sd = np.sqrt(max(s2 / i - m * m, 1e-6))
                out["amt_z"][r] = (la_[i] - m) / sd; out["amt_ratio_max"][r] = amt[i] / max(mx, 1)
            out["new_mcc"][r] = mcc[i] not in seen_mcc if i > 0 else 0
            out["new_merchant"][r] = mer[i] not in seen_mer if i > 0 else 0
            out["dist_home_km"][r] = haversine_km(cp_lat / cp_n, cp_lon / cp_n, la[i], lo[i]) if cp_n else 0
            # update history AFTER computing features
            s1 += la_[i]; s2 += la_[i] ** 2; mx = max(mx, amt[i]); seen_mcc.add(mcc[i]); seen_mer.add(mer[i]); mer_ts.setdefault(mer[i], []).append(ts[i])
            if ch[i] != "ECOM":
                cp_lat += la[i]; cp_lon += lo[i]; cp_n += 1; last_cp = i
        pos += n
    feat = pd.DataFrame(out)
    return pd.concat([df, feat.drop(columns=["amount"])], axis=1)
