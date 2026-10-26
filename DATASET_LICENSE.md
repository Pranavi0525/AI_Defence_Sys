# Dataset Provenance & Licensing

This document records the provenance, licensing, access requirements, and
redistribution rights for all reference datasets used by Red Team AI Module 1.

**Rule**: Raw third-party datasets must NOT be committed to this repository
unless redistribution is explicitly permitted. Only derived statistics with
proper citation may be stored.

---

## 1. PaySim Dataset (Kaggle ealaxi/paysim1)

| Field | Value |
|-------|-------|
| **ID** | `paysim_kaggle_v1` |
| **Source Type** | `SYNTHETIC_RESEARCH_DATASET` |
| **URL** | https://www.kaggle.com/datasets/ealaxi/paysim1/data |
| **Access** | Kaggle CSV Download |
| **License** | CC BY-SA 4.0 |
| **Redistribution** | Derived statistics only — raw CSV not committed |
| **Commercial/Hackathon Use** | Permitted under CC BY-SA 4.0 with attribution |
| **Citation Required** | Yes |
| **Citation** | Lopez-Rojas, Elmir, Axelsson. "PaySim: A financial mobile money simulator for fraud detection" (2016) |
| **Used For** | Transaction amount distributions, transaction type frequencies, temporal patterns, balance dynamics (Filtered for LEGITIMATE non-fraud transactions to calibrate baseline). |
| **Known Limitations** | Synthetic origin — does not capture real-world tail behavior, merchant diversity, multi-currency, or real geographic patterns. Agent-based model may produce artifacts not present in real payment systems. Limited transaction types. No device/session/channel data. |

---

## 2. IEEE-CIS Fraud Detection (Kaggle Competition)

| Field | Value |
|-------|-------|
| **ID** | `ieee_cis_fraud_v1` |
| **Source Type** | `REAL_WORLD_PRODUCTION` (anonymized/transformed real transactions) |
| **URL** | https://www.kaggle.com/c/ieee-fraud-detection/data |
| **Access** | Kaggle Competition Agreement (Authentication Required) |
| **License** | Subject to Kaggle Competition Rules |
| **Redistribution** | Derived statistics only — raw data NOT committed |
| **Commercial/Hackathon Use** | Competition use per rules |
| **Citation Required** | Yes |
| **Citation** | IEEE Computational Intelligence Society + Vesta Corporation (2019) |
| **Used For** | Device categorical attributes, relative timedelta features |
| **Known Limitations** | TransactionDT is relative, not a wall-clock timestamp. Heavily anonymized — many V-features have no semantic meaning. USD only. US-centric. Competition data may have selection bias. No session/login data. No beneficiary data. |

---

## 3. Domain Knowledge (ATO Typologies)

| Field | Value |
|-------|-------|
| **ID** | `domain_ato_typologies` |
| **Source Type** | `DOMAIN_KNOWLEDGE` |
| **Sources** | FATF typologies, EMVCo fraud classification, UK Finance fraud reports, Javelin identity fraud studies, APWG phishing reports |
| **Access** | Public advisory documents |
| **License** | Public domain / fair use for research summaries |
| **Redistribution** | Summaries and references only |
| **Used For** | ATO phase definitions, transition plausibility, observable consequence definitions |
| **Known Limitations** | Qualitative typologies — no precise probability distributions. Publication bias toward detected attacks. |

---

## Calibration Mode Labeling

All calibration outputs explicitly record:

```
calibration_mode: RAW_DATA | REFERENCE_STATISTICS
```

- **RAW_DATA**: Raw reference dataset is available locally for direct statistical comparison.
- **REFERENCE_STATISTICS**: Only published/independently documented summary statistics are available.

Reference-statistics calibration is NOT equivalent to raw-data validation.
Every hardcoded statistic must include: source, citation, statistic definition,
derivation notes, and whether derivation was independently verified.
