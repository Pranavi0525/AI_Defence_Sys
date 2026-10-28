# AI Defence System

## AI Risk Manager for Payment Fraud

An explainable, layered payment-risk research and demonstration platform combining behavioral rules, machine learning, graph signals, anomaly detection, risk fusion, operational decision policy, and a separate credit-card fraud detection system.

---

# 1. What We Built

Payment fraud detection is not only a classification problem.

A robust risk-management system needs to answer several questions:

- Is this behavior suspicious?
- How confident are we?
- Should the transaction be allowed, reviewed, or blocked?
- Can we explain why?
- Which operational party should act?
- What happens when a fraud case is missed?
- Can the system be evaluated without leaking hidden attack information into the model?

AI Defence System addresses these questions through two independent fraud-detection workflows:

1. **Digital Payment Fraud Detection** — a layered payment-security research pipeline involving attack simulation, behavioral detection, risk analysis, and decision support.
2. **Credit Card Fraud Detection (Phase 10)** — an independent transaction-scoring system with its own simulator, feature engineering pipeline, XGBoost model, serving layer, API endpoints, and dashboard.

Both workflows are accessible through one web dashboard while maintaining separate models, scoring logic, and API routes.

---

# 2. Live Application

| Component | URL |
|---|---|
| Web Dashboard | https://ai-defence-sys.vercel.app/ |
| Backend API | https://ai-defense-api.onrender.com |
| API Documentation | https://ai-defense-api.onrender.com/docs |
| Health Check | https://ai-defense-api.onrender.com/healthz |
| Readiness Check | https://ai-defense-api.onrender.com/readyz |
| Credit Card Model Status | https://ai-defense-api.onrender.com/api/card/status |

### Dashboard Modules

The deployed dashboard provides two independent modules:

- Digital Payment Fraud Detection
- Credit Card Fraud Detection

The frontend is hosted on Vercel, while the backend API and model-serving infrastructure run on Render.

---

# 3. Product Architecture

The system separates the two fraud-detection workflows.

```text
                    AI DEFENCE SYSTEM
                           |
                  Vercel Web Dashboard
                           |
             +-------------+-------------+
             |                           |
             ▼                           ▼
    Digital Payment Fraud        Credit Card Fraud
        Detection                    Detection
             |                           |
             ▼                           ▼
       Payment API                  Card API
             |                           |
             ▼                           ▼
     Payment Risk Pipeline          CardScorer
             |                           |
     +-------+--------+                  |
     |       |        |                  ▼
   Rules   XGBoost   Graph            XGBoost
     |       |        |                  |
     |   Autoencoder  |                  |
     |       |        |                  |
     +-------+--------+                  |
             |                           |
             ▼                           ▼
        Risk Fusion                Card Model Bundle
             |                           |
             ▼                           ▼
      Decision Policy             Transaction Score
             |                           |
             ▼                           ▼
       ALLOW / REVIEW / BLOCK      FRAUD / LEGITIMATE
```

The credit-card module communicates exclusively with `/api/card/*` routes and does not reuse the digital-payment model.

---

# 4. Digital Payment Fraud Detection

The digital-payment workstream focuses on a closed-loop payment-security research workflow.

## Core Components

1. Behavioral rule filtering
2. Calibrated XGBoost fraud scoring
3. Graph-based relationship analysis
4. Autoencoder-based novelty detection
5. Risk fusion
6. Cost-aware decision policy
7. SHAP-based explainability
8. Miss collection and hard-example analysis
9. API and dashboard integration

The system evaluates controlled synthetic payment-security scenarios representing:

- `ACCOUNT_TAKEOVER` (ATO)
- `AUTHORIZED_PUSH_PAYMENT` (APP)
- `MULE_NETWORK`

The data is synthetic and does not interact with real payment systems, accounts, or payment networks.

## 4.1 Core Design Principle: No Ground-Truth Leakage

The Red Team produces two conceptually different pieces of information:

```text
AttackRecord
├── observable_trace
└── ground_truth
```

The model receives only the observable trace.

Ground truth is reserved for:

- Labels
- Stratification
- Evaluation
- Error analysis
- Attack-family comparison
- Difficulty-specific analysis

It is not used as a predictive feature.

This preserves the distinction between:

**What the attacker actually did**

and

**What a payment-security system could observe.**

## 4.2 Attack Scenarios

### ACCOUNT_TAKEOVER

ATO represents unauthorized access to a customer's account.

Observable behavioral signals can include:

- New-device registration
- Login and session activity
- Device changes
- Transaction frequency
- Transaction fragmentation
- Failed transaction attempts
- Timing patterns
- Transaction progression

### AUTHORIZED_PUSH_PAYMENT

APP represents a legitimate customer being socially engineered into authorizing a payment.

Observable signals can include:

- Trusted-device usage
- Session continuity
- Beneficiary activity
- Hesitation gaps
- Failed and retried payments
- Amount reduction after failures
- Transaction timing

### MULE_NETWORK

MULE_NETWORK models coordinated movement of funds through related entities.

Graph relationships can include:

```text
Customer
   |
   +---- Device
   |
   +---- Session
   |
   +---- Beneficiary
   |
   +---- Transaction
```

Graph connectivity is used only where an observable relationship exists.

## 4.3 Risk Pipeline

### Stage 1 — Behavioral Rule Filter

Fast deterministic checks identify suspicious behavioral patterns.

```text
Low-risk behavior
      |
      +----> May be auto-cleared

Suspicious / uncertain behavior
      |
      +----> Escalated to ML scoring
```

### Stage 2 — Calibrated XGBoost

The primary tabular ML detector uses engineered behavioral and event-sequence features.

Representative features include:

- Transaction counts
- Session and login counts
- Device registrations
- Beneficiary additions
- Total events
- Transactions per hour
- Transactions per session
- Authentication failures
- Failed transactions
- Amount statistics
- Channel diversity
- Event timing

### Stage 3 — Graph Escalation

The graph detector models relationships among:

- Customers
- Devices
- Sessions
- Beneficiaries
- Transactions

It is particularly useful when suspicious activity is connected through observable cross-customer relationships.

### Stage 4 — Autoencoder Novelty Detection

The autoencoder provides an independent anomaly signal based on reconstruction error.

It is designed to identify behavior that differs from the learned normal behavioral representation.

Unusual does not necessarily mean fraudulent, so anomaly scores are not treated as fraud truth by themselves.

### Stage 5 — Risk Fusion

Multiple signals can be combined into a unified risk score.

```text
Stage 1 / Stage 2 score
          +
Graph signal
          +
Autoencoder signal
          |
          ▼
       Risk Fusion
          |
          ▼
      Unified Risk
```

Additional signals must be evaluated against both their detection benefit and operational cost.

---

# 5. Decision Policy

The unified risk score is mapped to an operational action.

```text
                    Risk Score
                        |
          +-------------+-------------+
          |             |             |
          ▼             ▼             ▼
        ALLOW         REVIEW         BLOCK
```

The decision policy is cost-aware rather than optimized solely for classification accuracy.

The evaluation considers:

- Fraud prevalence
- Review operations cost
- Legitimate-user friction
- Liability exposure
- Fraud caught through review

The project evaluates the decision policy under different assumptions rather than treating a model score as an automatic operational decision.

## Liability-Aware Decisions

The system distinguishes operational responsibility across fraud families.

```text
ACCOUNT_TAKEOVER
liable side:   SENDING
acting side:   SENDING

AUTHORIZED_PUSH_PAYMENT
liable side:   SHARED_50_50
acting side:   BOTH

MULE_NETWORK
liable side:   RECEIVING
acting side:   RECEIVING
```

This connects detection to an operational response rather than stopping at a fraud classification.

---

# 6. Explainability and Evaluation

The system provides global and case-level explanations.

## Global Explanations

SHAP is used to understand features driving the Stage 2 model.

Representative features include:

- `min_time_between_transactions`
- `failed_transaction_count`
- `mean_time_between_transactions`
- `count_beneficiary_addition`
- `total_events`
- `count_device_registration`
- `transactions_per_hour`

## Evaluation Metrics

The evaluation considers:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion matrix
- False-positive rate
- False-negative rate
- Family-level performance

The project also examines:

- Stage-specific misses
- Graph-stage recovery
- Fusion performance
- Decision-policy sensitivity
- Hard-example generalization

All synthetic evaluation metrics describe the evaluated synthetic population. They are not claims of real-world payment-fraud performance.

---

# 7. Credit Card Fraud Detection — Phase 10

Phase 10 introduces an independent credit-card fraud detection system.

Unlike the digital-payment workflow, this module is designed around individual card transactions and eligible transaction history.

## 7.1 Features

- Individual transaction scoring
- CSV batch prediction
- Transaction-history-based behavioral features
- Velocity analysis
- Merchant and transaction novelty analysis
- XGBoost classification
- Training-support and reliability checks
- Model status and evaluation information
- Separate dashboard interface

## 7.2 Model Architecture

The card module uses 27 engineered features and a separately trained XGBoost model.

```text
Card Transaction
       |
       ▼
Input Validation
       |
       ▼
History Filtering
       |
       ▼
Feature Engineering
       |
       ▼
27 Engineered Features
       |
       ▼
XGBoost Model
       |
       ▼
Threshold Comparison
       |
       ▼
Prediction + Reliability Information
```

Model artifacts:

```text
models/card/
├── policy.json
└── xgb.json
```

Deployed model version:

```text
card-xgb-synthetic-seed0-c0a47d31
```

The serving layer loads the model bundle and associated policy rather than hardcoding the model threshold in the frontend.

## 7.3 Transaction History

The system uses eligible earlier transactions belonging to the same card.

It excludes:

- Transactions belonging to another card
- Future-dated transactions
- Duplicate transactions
- The transaction currently being scored

History-dependent features may be unavailable when insufficient history is supplied.

The API reports the number of supplied and used history rows, filtering information, and relevant warnings.

## 7.4 Amount Boundaries

The implementation distinguishes the following limits:

| Boundary | Amount |
|---|---:|
| Synthetic simulator ceiling | ₹10,00,000 |
| Observed training-data maximum | ₹4,65,290.34 |
| Default serving input-validation limit | ₹10,00,000 |

The serving limit is an input-validation boundary, not an issuer authorisation policy.

Transactions above the observed training range may still be scored when within the configured serving limit. However, the API reports reliability warnings and out-of-range features.

A transaction being below the serving limit does not guarantee that the model has sufficient learned resolution for that transaction amount.

## 7.5 Single Transaction Prediction

Endpoint:

```text
POST /api/card/predict
```

Example request:

```json
{
  "transaction": {
    "card": "card_001",
    "ts": 1793102400,
    "amount": 1000,
    "mcc": "5411",
    "channel": "POS",
    "lat": 13.0827,
    "lon": 80.2707,
    "merchant": "merchant_001",
    "declined": 0
  },
  "history": []
}
```

The response includes the prediction, ranking score, threshold, model version, history information, and reliability details.

## 7.6 Batch CSV Prediction

Endpoint:

```text
POST /api/card/predict-batch
```

The batch workflow supports CSV transaction uploads and returns predictions with relevant scoring and reliability information.

The batch interface includes:

- CSV upload
- Input validation
- Transaction scoring
- Fraud and legitimate classifications
- Score and threshold information
- Reliability warnings
- Downloadable prediction results

## 7.7 Measured Phase 10 Results

The following figures come from the **extended synthetic card-simulator evaluation**. They are not measurements from live bank or issuer transactions.

### Held-out synthetic test results

| Metric | Result |
|---|---:|
| Precision | **69.9%** |
| Recall | **85.2%** |
| F1-score | **0.768** |
| False-positive rate | **0.652%** |
| Fraud transactions detected | **53 / 61** |
| High-ticket legitimate transactions incorrectly flagged | **25 / 215** |
| Fraud-event recall | **92.0% across 150 simulated fraud events** |
| Selected decision threshold | **0.044527** |

The threshold was selected on validation legitimate transactions at a target false-positive rate of 1%. The 0.652% false-positive rate above is the measured result on the held-out synthetic test set.

These results describe this simulator, data split, and model bundle only. They are not estimates of production performance.

### Earlier bundle vs. extended-simulator bundle

| Metric | Earlier bundle | Extended-simulator bundle |
|---|---:|---:|
| Precision | 54.0% | **69.9%** |
| Recall | 80.9% | **85.2%** |
| F1-score | 0.648 | **0.768** |
| False-positive rate | 1.225% | **0.652%** |
| High-ticket legitimate transactions flagged | 40 / 215 | **25 / 215** |
| Fraud transactions detected | 44 / 61 | **53 / 61** |

The extended simulator introduced:

- Legitimate high-value purchases
- Greater amount overlap between legitimate and fraudulent transactions
- New-card behaviour
- More organic fraud timing
- Additional fraud scenarios

The comparison shows improvement on this synthetic evaluation, while the remaining high-ticket false positives show that the issue is not fully solved.

### Evaluation and test evidence

- Temporal evaluation split: training before day 30, validation from day 30 to day 42, and test data from day 42 onward.
- Model bundle: `card-xgb-synthetic-seed0-c0a47d31`.
- Feature count: **27**.
- Recorded full project test run: **681 passed**.
- Recorded card/API-focused test run: **56 passed**.

Test counts are software test results, not model-performance metrics.

### Known score-resolution limitation

Amount-range checks identify transactions outside observed training support, but do not guarantee useful score resolution throughout the supported range.

The XGBoost score can become relatively flat above approximately ₹50,000 in some contexts because tree models do not smoothly extrapolate beyond learned split boundaries.

This remains an open limitation. An in-range amount or a returned score is not proof of reliable ranking.

---

# 8. Engineering Challenges and How They Were Addressed

The following table records concrete development issues and the engineering responses. Where a mitigation did not fully solve the issue, the remaining limitation is stated explicitly.

| Challenge | What we changed | Remaining limitation |
|---|---|---|
| Card and digital-payment transactions have different schemas and scoring requirements | Kept Phase 10 separate, with its own 27-feature pipeline, model bundle, serving class, `/api/card/*` routes, and dashboard module | The two workflows are not interchangeable models |
| The original synthetic card data underrepresented legitimate large purchases | Extended the simulator with legitimate high-ticket purchases and more overlap between legitimate and fraudulent amounts | 25 of 215 high-ticket legitimate transactions were still flagged in the reported held-out synthetic test |
| Fraud timing was too concentrated and could become a shortcut feature | Used multi-seed timing diagnostics and retimed the extended simulator; the 10–30 minute concentration fell from roughly 97.6–98.5% to roughly 3–4% | This is a simulator diagnostic, not evidence about real-world fraud timing |
| Model behaviour was unreliable beyond observed training support | Separated the ₹10,00,000 serving input cap from the ₹4,65,290.34 observed training maximum and added out-of-range reliability warnings | Score resolution can still flatten above approximately ₹50,000; warnings do not fix extrapolation |
| Supplied history could contain other-card, future, duplicate, or current-transaction rows | Filtered history to unique, strictly earlier transactions for the same card and exposed supplied/used row counts and warnings | History-dependent features remain unavailable when valid history is insufficient |
| New cards may have little or no history | Left unavailable history statistics missing and returned warnings rather than fabricating historical values | Sparse history means less behavioural context |
| NaN and infinity could cause validation errors that were not JSON-serializable | Added recursive JSON-safe validation-error handling and a regression test for NaN, positive infinity, and negative infinity | Invalid values are still rejected; the change makes the error response serializable |
| A dashboard test value did not match the amount actually sent to the API | Checked the form value, outgoing request, and API response, then corrected the test input | Manual tests should verify the actual request payload |
| Payment pipeline signals have different strengths and operational costs | Kept rules, XGBoost, graph signals, and autoencoder novelty distinct for risk fusion | The payment cascade still has a Stage 1 recall ceiling; some fraud cases are never escalated |
| Graph detection cannot infer links absent from the observed data | Restricted graph conclusions to observable relationships | In the evaluated synthetic payment scenarios, 21 of 42 mule rings had no observable links |
| Anomaly detection can flag legitimate unusual behaviour | Treated autoencoder reconstruction error as an independent signal, not as fraud truth | In the evaluated synthetic payment scenarios, it recovered four frauds but newly flagged 55 legitimate cases |
| Deployment could use the wrong branch or include unrelated generated outputs | Deployed the `phase10` branch and selectively staged intended project files | Generated reports and diagnostic artifacts left unstaged should be reviewed separately before future commits |

The payment-pipeline counts above describe evaluated synthetic scenarios, not live payment-network measurements.

---

# 9. Model Limitations and Responsible Use

The Phase 10 credit-card model is an engineering prototype.

- Trained and evaluated on synthetic simulator data only.
- Fraud patterns and transaction distributions are simulator assumptions.
- The model score is a ranking score, not a calibrated real-world fraud probability.
- Synthetic evaluation metrics do not establish real-world fraud-detection performance.
- Tree models cannot extrapolate beyond learned split boundaries.
- Transactions outside observed training support may produce unreliable scores.
- Out-of-training-range warnings must be considered when interpreting predictions.

**Do not use this prototype to approve, decline, block, or otherwise make real financial authorisation decisions.**

Real-world deployment would require representative transaction data, temporal and out-of-distribution validation, calibration, operational threshold selection, monitoring, security review, and compliance approval.

---

# 10. Technology Stack

| Layer | Technologies |
|---|---|
| Backend | Python, FastAPI, Uvicorn, Pydantic |
| Machine Learning | XGBoost, scikit-learn |
| Data Processing | NumPy, pandas |
| Payment Detection | Behavioral rules, graph signals, anomaly detection |
| Credit Card Detection | Feature engineering, XGBoost |
| Explainability | SHAP |
| Frontend | HTML, CSS, JavaScript |
| Testing | pytest |
| Backend Hosting | Render |
| Frontend Hosting | Vercel |
| Version Control | Git, GitHub |

---

# 11. API Endpoints

## Digital Payment Fraud Detection

| Method | Endpoint | Description |
|---|---|---|
| GET | `/healthz` | Backend health |
| GET | `/readyz` | Backend readiness |
| POST | `/api/score` | Payment trace scoring |

Refer to the live API documentation for the complete payment-scoring route definitions and schemas.

## Credit Card Fraud Detection

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/card/status` | Model readiness and configuration |
| GET | `/api/card/model-card` | Model evaluation information |
| POST | `/api/card/predict` | Score one card transaction |
| POST | `/api/card/predict-batch` | Score a CSV batch of card transactions |

---

# 12. Deployment

The application is deployed using separate frontend and backend services.

### Frontend

- Platform: Vercel
- Dashboard: https://ai-defence-sys.vercel.app/

### Backend

- Platform: Render
- API: https://ai-defense-api.onrender.com
- Health endpoint: `/healthz`
- Readiness endpoint: `/readyz`

The backend serves both payment and card modules while keeping their API routes and model-serving logic separate.

---

# 13. Project Scope and Future Work

The current implementation is a research and engineering prototype. The following are important next steps before any real-world financial deployment:

1. Evaluate on representative, appropriately governed real transaction datasets.
2. Validate across different time periods, transaction channels, merchants, and fraud patterns.
3. Improve high-value legitimate transaction discrimination.
4. Investigate the card model's score saturation at higher transaction amounts.
5. Calibrate scores using representative validation data.
6. Establish issuer-specific operating thresholds and review policies.
7. Add production-grade monitoring for data drift, model drift, latency, and service health.
8. Evaluate graph coverage and missing-relationship limitations.
9. Improve the payment cascade's Stage 1 recall without creating unacceptable review volume.
10. Conduct security, privacy, compliance, and operational reviews.

The long-term objective is to build an explainable fraud-intelligence layer that combines deterministic controls with behavioural machine learning and contextual risk signals, while preserving human and issuer control over financial decisions.
