# AI Defence System

## AI Risk Manager for Payment Fraud

An explainable, layered payment-risk system that combines behavioral rules, calibrated machine learning, graph signals, anomaly detection, risk fusion, and operational decision policy to evaluate synthetic payment-security events.

> **Buildathon project:** Razorpay AI Buildathon — Track 02: AI Risk Manager

---

## 1. What We Built

Payment fraud detection is not only a classification problem.

A production-oriented risk system needs to answer several questions:

- Is this behavior suspicious?
- How confident are we?
- Should the transaction be allowed, reviewed, or blocked?
- Can we explain why?
- Which operational party should act?
- What happens when a fraud case is missed?
- Can the system be evaluated without leaking hidden attack information into the model?

AI Defence System addresses these questions through a layered risk-management architecture.

The project combines:

1. **Behavioral rule filtering**
2. **Calibrated XGBoost fraud scoring**
3. **Graph-based relationship analysis**
4. **Autoencoder-based novelty detection**
5. **Risk fusion**
6. **Cost-aware decision policy**
7. **SHAP-based explainability**
8. **Miss collection and hard-example analysis**
9. **A production API and web dashboard**

The system is evaluated using controlled synthetic payment-security scenarios representing:

- `ACCOUNT_TAKEOVER` (ATO)
- `AUTHORIZED_PUSH_PAYMENT` (APP)
- `MULE_NETWORK`

The data is synthetic and does not interact with real payment systems, accounts, or payment networks.

---

# 2. Product View

At a high level:

```text
                 Synthetic Payment Telemetry
                           │
                           ▼
                 Observable Trace Boundary
                           │
                           ▼
                     Feature Engineering
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
           Rules       XGBoost       Graph
              │            │          Detector
              │            │            │
              │            └─────┬──────┘
              │                  │
              │            Autoencoder
              │                  │
              └────────────┬─────┘
                           ▼
                       Risk Fusion
                           │
                           ▼
                     Decision Policy
                           │
                 ┌─────────┼─────────┐
                 ▼         ▼         ▼
               ALLOW     REVIEW     BLOCK
                           │
                           ▼
                       Explanation
                           │
                           ▼
                       Miss Collection
                           │
                           ▼
                     Hard Examples /
                      Re-evaluation
```

The architecture deliberately separates:

- **observable behavioral evidence**
- **hidden ground truth used for evaluation**
- **model inference**
- **operational decisioning**

This separation is a central design requirement of the project.

---

# 3. Core Design Principle: No Ground-Truth Leakage

The Red Team produces two conceptually different pieces of information:

```text
AttackRecord
├── observable_trace
└── ground_truth
```

The model receives only the observable trace.

Ground truth is reserved for:

- labels
- stratification
- evaluation
- error analysis
- attack-family comparison
- difficulty-specific analysis

It is not used as a predictive feature.

For example, fields such as:

```text
attack_family
attack_difficulty
hidden_objective
planner_metadata
internal phase labels
ground-truth attack objective
```

must never become model features.

This preserves the distinction between:

> **what the attacker actually did**

and

> **what a payment-security system could observe.**

The final Red Team verification reported:

```text
Observable events scanned:       1,783
Observable leakage findings:          0
Ground-truth/observable mismatches:   0
```

---

# 4. Attack Scenarios

## ACCOUNT_TAKEOVER

ATO represents unauthorized access to a customer's account.

Observable behavioral signals can include:

- new-device registration
- login/session activity
- device changes
- transaction frequency
- transaction fragmentation
- failed transaction attempts
- timing patterns
- transaction progression

---

## AUTHORIZED_PUSH_PAYMENT

APP represents a legitimate customer being socially engineered into authorizing a payment.

Observable signals can include:

- trusted-device usage
- session continuity
- beneficiary activity
- hesitation gaps
- failed/retried payments
- amount reduction after failures
- transaction timing
- session-to-payment relationships

---

## MULE_NETWORK

MULE_NETWORK models coordinated movement of funds through related entities.

The graph layer can use observable relationships such as:

```text
Customer
   │
   ├── Device
   ├── Session
   ├── Beneficiary
   └── Transaction
```

Graph connectivity is only used where an observable relationship exists.

The system does not fabricate relationships for traces where the corpus does not expose a connecting field.

---

# 5. Risk Pipeline

## Stage 1 — Behavioral Rule Filter

Fast deterministic checks identify suspicious behavioral patterns.

The purpose is not to replace machine learning.

It acts as an initial routing layer:

```text
Low-risk behavior
      │
      └──► May be auto-cleared

Suspicious / uncertain behavior
      │
      └──► Escalated to ML scoring
```

This provides a measurable trade-off between fraud recall and the amount of legitimate traffic requiring further processing.

---

## Stage 2 — Calibrated XGBoost

The primary tabular ML detector uses engineered behavioral and event-sequence features.

Representative features include:

- transaction counts
- session/login counts
- device registrations
- beneficiary additions
- total events
- transactions per hour
- transactions per session
- authentication failures
- failed transactions
- amount statistics
- amount trends
- channel diversity
- event timing

The deployed API currently uses the validated Stage 1+2 model artifact.

Model provenance is tracked through the project's model registry.

---

## Stage 3 — Graph Escalation

The graph detector models relationships among:

```text
Customer
Device
Session
Beneficiary
Transaction
```

It is particularly useful when suspicious activity is connected through observable cross-customer relationships.

In the evaluated corpus:

- graph escalation rescued additional fraud cases
- graph processing is structurally a no-op for traces with no observable connecting relationship

This is an intentional limitation rather than an attempt to infer unavailable information.

---

## Stage 4 — Autoencoder Novelty Detection

The autoencoder provides an independent anomaly signal based on reconstruction error.

It is designed to identify behavior that differs from the learned normal behavioral representation.

Unlike the graph layer, the autoencoder can score every row.

This creates an important operational trade-off:

> unusual does not necessarily mean fraudulent.

Therefore anomaly scores are not treated as fraud truth by themselves.

---

## Stage 5 — Risk Fusion

Multiple signals can be combined into a unified risk score:

```text
Stage 1 / Stage 2 score
          +
Graph signal
          +
Autoencoder signal
          │
          ▼
       Risk Fusion
          │
          ▼
      Unified risk
```

The project evaluates multiple fusion strategies rather than assuming that adding another detector automatically improves performance.

For the real corpus, the reported comparison was:

| Configuration | Precision | Recall | F1 |
|---|---:|---:|---:|
| Stage 1+2 | 0.989 | 0.963 | 0.976 |
| Stage 1+2 + GCN | 0.989 | 0.968 | 0.978 |
| Stage 1+2 + Autoencoder | 0.854 | 0.973 | 0.910 |
| Naive max of all three | 0.855 | 0.979 | 0.913 |
| Stacked logistic-regression fusion | 0.986 | 0.965 | 0.976 |

These results illustrate an important engineering point:

> More detectors do not automatically produce a better production decision system.

Additional signals must be evaluated against both detection benefit and operational cost.

---

# 6. Decision Policy

The unified risk score is mapped to an operational action:

```text
                    Risk Score
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
        ALLOW         REVIEW         BLOCK
```

The decision policy is cost-aware rather than optimized solely for classification accuracy.

The evaluation considers factors such as:

- fraud prevalence
- review operations cost
- legitimate-user friction
- liability exposure
- fraud caught through review

The nested/fold-honest threshold analysis reported:

```text
ALLOW   73.5%
REVIEW   4.7%
BLOCK   21.8%

Fraud recall (BLOCK + REVIEW): 97.6%
```

These figures are evaluation results on the project's synthetic corpus and should not be interpreted as production fraud-rate estimates.

---

## Decision-Policy Sensitivity

The decision policy was tested under different assumptions about production fraud prevalence and review-operation cost.

Across the tested ranges:

- Production fraud rate: **0.2%–2.0%**
- Review-operation cost: **$5–$50**

the selected block threshold remained stable at:

```text
t_block = 0.9643
```

The block-threshold span across both sensitivity sweeps was:

```text
0.9643 - 0.9643
span = 0.000
```

At the reference sensitivity setting of **0.6% production fraud rate** and **$12 review-operation cost**, the evaluated policy produced:

| Decision | Share |
|---|---:|
| Allow | 72.9% |
| Review | 5.0% |
| Block | 22.1% |

Fraud recall when combining Review + Block was **98.1%** under this sensitivity setting.

The review threshold was more sensitive to the operating assumptions than the block threshold. This indicates that, within the tested range, the high-risk boundary was stable while the amount of traffic routed to manual review could change.

These are sensitivity-analysis results on the project's synthetic evaluation population and are not claims about real-world fraud prevalence, operating cost, or production performance.

---

# 7. Liability-Aware Decisions

The system does not treat every fraud family as having identical operational responsibility.

The evaluated policy records:

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

This allows the risk system to connect a detection decision to an operational action rather than stopping at:

> "This transaction is fraudulent."

The system can instead reason about:

> "This is risky, and this is the side that should act."

---

# 8. Explainability

The system provides both global and case-level explanations.

## Global explanations

SHAP is used to understand the features driving the Stage 2 model.

The strongest reported features included:

```text
min_time_between_transactions
failed_transaction_count
mean_time_between_transactions
count_beneficiary_addition
total_events
count_device_registration
window_seconds
transactions_per_hour
```

## Case-level explanations

Representative cases include:

- correctly blocked ATO
- correctly blocked APP
- MULE_NETWORK case rescued by fusion
- fraud routed to review
- fraud that remained allowed
- legitimate case routed to review
- ordinary legitimate activity auto-cleared

The project also generates investigator-oriented case dossiers for Review/Block cases.

---

# 9. Evaluation

The evaluation is designed around more than accuracy.

Metrics include:

- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- Confusion Matrix
- False-positive rate
- False-negative rate

Performance is also examined by attack family.

The primary Stage 1+2 evaluation on the validated evaluation population reported:

```text
Population:       1,556
Fraud:              374
Legitimate:       1,182

Accuracy:        98.84%
Precision:       98.37%
Recall:          96.79%
F1:              97.57%

False-positive rate: 0.51%
False-negative rate: 3.21%
```

The corresponding classification counts were:

```text
True Positive:   362
True Negative:  1176
False Positive:    6
False Negative:   12
```

These numbers are evaluation results on synthetic data. They are not claims of real-world payment-fraud performance.

---

# 10. Family-Level Evaluation

The project evaluates the three attack families separately because aggregate performance can hide family-specific weaknesses.

The full-cascade diagnostic reported:

| Attack family | Precision | Recall | F1 |
|---|---:|---:|---:|
| ACCOUNT_TAKEOVER | 0.959 | 0.959 | 0.959 |
| AUTHORIZED_PUSH_PAYMENT | 0.974 | 0.962 | 0.968 |
| MULE_NETWORK | 0.967 | 0.967 | 0.967 |

Family-level evaluation is especially important for MULE_NETWORK because graph reachability depends on whether the observable corpus actually exposes a shared relationship.

---

# 11. Data Quality

The persisted Red Team corpus contains:

```text
ATO:                 97 traces
APP:                156 traces
Total attacks:      253 traces
Observable events: 1,783
```

Final corpus verification reported:

```text
Malformed records:              0
Duplicate attack IDs:           0
Cross-family ID overlap:        0
GT/observable mismatches:       0
Observable leakage findings:    0
Chronology failures:            0
Schema failures:                0
```

The project deliberately preserves the actual qualified corpus size rather than fabricating records to reach a requested target.

---

# 12. Adaptive Feedback Loop

Missed fraud cases are collected after the full cascade.

The feedback workflow is:

```text
Full-cascade evaluation
          │
          ▼
      Miss collection
          │
          ▼
  Responsible-stage analysis
          │
          ▼
  Hard-example generation
          │
          ▼
  Controlled re-evaluation
```

The purpose is to identify where the cascade fails and whether targeted examples can improve the responsible stage.

The feedback loop distinguishes:

- Stage 1 misses
- Stage 2 misses
- graph-stage misses
- decision-policy misses

Importantly, the system does not simply lower thresholds to eliminate misses.

The current adaptive evaluation found that some Stage 1 misses can be recovered through additional validated examples, while Stage 2 generalization remained partial for the original miss set.

This is treated as an evaluation finding, not as evidence that every miss has been solved.

---

# 13. Production Deployment

The project includes a production API and web dashboard.

### Frontend

Vercel:

**https://ai-defence-sys.vercel.app/**

### Backend

Render:

**https://ai-defense-api.onrender.com/**

The production backend exposes health/readiness endpoints and the scoring API.

The readiness endpoint confirms that the required runtime artifact is available before the API accepts scoring traffic.

---

# 14. Production Scoring Flow

The live dashboard sends an `ObservableAttackTrace` to the API.

Conceptually:

```text
Browser
   │
   │ POST /api/score
   ▼
Render API
   │
   ▼
Observable trace validation
   │
   ▼
Stage 1 + Stage 2 scoring
   │
   ▼
Calibrated fraud probability
   │
   ▼
Operational decision
   │
   ├── ALLOW
   ├── REVIEW
   └── BLOCK
   │
   ▼
Explanation / evidence
```

The dashboard is designed to expose the real API response rather than fabricate evaluation statistics.

---

# 15. Important Deployment Scope

The production deployment should be interpreted precisely.

The currently deployed online scoring path is the validated Stage 1+2 path.

The Stage 3, Stage 4, and Stage 5 components have been implemented and evaluated offline, but their complete trained inference artifacts are not currently exposed through the production API as independent deployable stages.

Therefore the dashboard explicitly distinguishes between:

```text
Online scoring:
Stage 1 + Stage 2

Offline evaluation:
Stage 1 → Stage 2 → Stage 3 → Stage 4 → Stage 5
```

This distinction is intentional.

The project does not claim that a component is production-served merely because it exists in the research/evaluation pipeline.

---

# 16. Running Locally

## Requirements

Python 3.12 is the supported runtime used by the deployment configuration.

Install dependencies:

```bash
pip install -r requirements.txt
```

Set the project source path.

### Windows PowerShell

```powershell
$env:PYTHONPATH="src"
```

### Linux/macOS

```bash
export PYTHONPATH=src
```

---

## Run the API

From the repository root:

```bash
python web_prototype/run_api.py --host 0.0.0.0 --port 8000
```

The API can then be accessed locally at:

```text
http://localhost:8000
```

---

## API Health

Check:

```text
/healthz
/readyz
```

`/healthz` indicates that the process is alive.

`/readyz` verifies that the runtime model/artifact prerequisites are available.

---

# 17. Running Tests

The project's tests can be executed with:

### Windows PowerShell

```powershell
$env:PYTHONPATH="src"
pytest tests/ -v -W error::DeprecationWarning
```

The repository contains dedicated tests covering the Red Team generation, validation, model/evaluation behavior, integration, and robustness work.

---

# 18. Repository Structure

```text
AI_Defence_Sys/
│
├── src/
│   └── red_team/
│       ├── attacks/
│       ├── schemas/
│       ├── validation/
│       ├── world/
│       └── ml/
│
├── reports/
│   ├── ato_corpus_raw.json
│   └── app_corpus_raw.json
│
├── blue_team_output_FROZEN/
│   ├── xgb_model.joblib
│   ├── gnn_results.json
│   ├── stage4_autoencoder_results.json
│   ├── risk_fusion_results.json
│   └── explainability/
│
├── frozen_reports/
│
├── blue_team_output/
│
├── web_prototype/
│   ├── dashboard/
│   └── ...
│
├── streamlit_app/
│
├── tests/
│
├── RED_TEAM_HANDOFF.md
├── BLUE_TEAM_INTEGRATION_SPEC.md
├── requirements.txt
├── render.yaml
└── README.md
```

---

# 19. Security and Data Boundary

This project is a controlled payment-security research simulation.

It:

- does not access real banking systems
- does not interact with payment networks
- does not submit real payment requests
- does not target real customer accounts
- does not contain real customer payment information
- uses synthetic entities and simulated events

The Red Team exists to provide controlled adversarial scenarios for defensive evaluation.

---

# 20. Engineering Principles

The project follows several principles throughout the pipeline.

### 1. Observable data is the model boundary

Hidden attack metadata is never treated as behavioral evidence.

### 2. Accuracy is not enough

Precision, recall, F1, PR-AUC, false positives, false negatives, operational cost, and family-specific performance are considered.

### 3. More models are not automatically better

Each additional signal must demonstrate useful incremental value.

### 4. Detection and decisioning are separate concerns

A probability estimate is not itself an operational action.

### 5. Explainability is part of the system

The system produces evidence intended to support investigation rather than only returning a number.

### 6. Limitations are reported explicitly

Unavailable telemetry, graph-unreachable cases, synthetic-data limitations, and incomplete production deployment are documented rather than hidden.

### 7. Evaluation artifacts are treated as versioned evidence

Baseline results should not be silently overwritten by later experimental runs.

---

# 21. Demo

The recommended demonstration flow is:

### Step 1 — Open the dashboard

Open:

**https://ai-defence-sys.vercel.app/**

### Step 2 — Confirm backend connectivity

The dashboard should connect to:

**https://ai-defense-api.onrender.com**

### Step 3 — Load a real example trace

Use the dashboard's real-example workflow rather than inventing a synthetic response in the UI.

### Step 4 — Run scoring

Submit the observable trace to the live API.

### Step 5 — Show the decision

Demonstrate:

```text
Fraud probability
      ↓
Risk decision
      ↓
ALLOW / REVIEW / BLOCK
```

### Step 6 — Show the explanation

Highlight the behavioral evidence contributing to the decision.

### Step 7 — Explain the offline pipeline

Show how the same research system was evaluated across:

```text
Stage 1
   ↓
Stage 2
   ↓
Stage 3
   ↓
Stage 4
   ↓
Stage 5
   ↓
Decision Policy
   ↓
Explainability
```

Be explicit that the production API currently serves the Stage 1+2 online path while the complete cascade is evaluated offline.

---

# 22. Project Status

The project currently provides:

- synthetic adversarial payment-security corpora
- observable/ground-truth separation
- behavioral feature engineering
- calibrated Stage 1+2 ML scoring
- graph-based evaluation
- autoencoder novelty detection
- risk-fusion evaluation
- cost-aware decision policy
- liability-aware operational routing
- SHAP-based explainability
- miss collection
- hard-example analysis
- evaluation and robustness infrastructure
- production API deployment
- production web dashboard

The production deployment and research/evaluation pipeline are intentionally documented as separate scopes.

---

# 23. Documentation

Additional technical documentation is available in:

```text
RED_TEAM_HANDOFF.md
BLUE_TEAM_INTEGRATION_SPEC.md
```

The integration specification contains the detailed Red Team → Blue Team data contract, observable-data rules, feature-engineering guidance, evaluation requirements, and corpus verification information.

---

# 24. Final Note

AI Defence System is designed as an end-to-end risk-management research prototype rather than a claim of production fraud-detection performance.

The central engineering goal is to demonstrate a defensible path from:

```text
Observable behavior
        ↓
Detection
        ↓
Risk estimation
        ↓
Operational decision
        ↓
Explanation
        ↓
Evaluation
        ↓
Feedback
```

while maintaining a strict boundary between observable evidence and hidden ground truth.
