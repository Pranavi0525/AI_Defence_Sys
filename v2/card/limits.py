"""v2/card/limits.py -- shared, documented limits for the card module.  ALL AMOUNTS ARE INR (Indian rupees) PER TRANSACTION.

Amount boundaries and policies:
1. SYNTHETIC SIMULATOR CAP (`DEFAULT_SIM_MAX_AMOUNT` = 1,000,000.0 INR):
   Upper ceiling applied during synthetic transaction simulation (v2/card/sim.py).
2. ACTUAL TRAINING-DATA SUPPORT (`policy.amount_support.train_max_amount` = 465,290.34 INR):
   Realized maximum amount in the training partition of the synthetic dataset.
   Tree models cannot extrapolate beyond the splits learned from training data; amounts above
   training support are scored, but flagged with out-of-training-range warnings.
3. SERVING INPUT-VALIDATION LIMIT (`DEFAULT_MAX_TXN_AMOUNT` = 1,000,000.0 INR, env `CARD_MAX_TXN_AMOUNT`):
   Input-validation guard at serving time. Requests strictly above it are rejected with
   HTTP 422 before reaching the model; they are never scored. The serving limit is separate
   from the simulator cap and is not lowered to training support (e.g. ₹4 lakh).
4. ISSUER/BUSINESS TRANSACTION POLICY:
   Issuer-specific policies (e.g. daily limits, cardholder credit limits, velocity caps).
   NOTE: Issuer-specific policy engines are NOT currently implemented in this module.
"""
from __future__ import annotations

import math
import os

CURRENCY = "INR"
DEFAULT_SIM_MAX_AMOUNT = 1_000_000.0          # INR 10 lakh per transaction (simulator generation ceiling)
DEFAULT_MAX_TXN_AMOUNT = 1_000_000.0          # INR 10 lakh per transaction (serving input-validation default; override with CARD_MAX_TXN_AMOUNT)
ENV_MAX_TXN_AMOUNT = "CARD_MAX_TXN_AMOUNT"
MAX_TS_SECONDS = 4_102_444_800.0              # 2100-01-01 UTC as epoch seconds; the simulator uses small offsets, both are fine
MCC_MAX = 9999


class LimitsConfigError(RuntimeError):
    """CARD_MAX_TXN_AMOUNT is set to something that is not a positive finite number."""


def max_txn_amount() -> float:
    raw = os.environ.get(ENV_MAX_TXN_AMOUNT)
    if raw is None or raw.strip() == "":
        return DEFAULT_MAX_TXN_AMOUNT
    raw_str = raw.strip()
    try:
        v = float(raw_str)
    except ValueError as exc:
        raise LimitsConfigError(f"{ENV_MAX_TXN_AMOUNT}={raw!r} is not a number (expected INR, e.g. 1000000)") from exc
    if not math.isfinite(v) or v <= 0:
        raise LimitsConfigError(f"{ENV_MAX_TXN_AMOUNT}={raw!r} must be a positive finite number (INR)")
    return v
