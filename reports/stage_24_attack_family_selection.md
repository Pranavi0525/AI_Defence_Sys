# Stage 24: Red Team Attack-Family Selection

## 1. Identified Remaining Attack Families
Based on the explicit research sources listed in `DATASET_LICENSE.md` (FATF, EMVCo, UK Finance, Javelin) and the provided project specification examples, the following families are intended for support:

**1. AUTHORIZED_PUSH_PAYMENT (Social Engineering / Scam)**
- **Intended Mechanism:** Attacker manipulates victim into authorizing a payment themselves.
- **Objective:** Steal funds via direct transfer to an attacker-controlled account.
- **Observable Behavior:** Activity originates from the victim's *known/primary* device and normal IP. Involves adding a new beneficiary and initiating a transfer, often with unusual velocity or amount.
- **Ground-Truth Representation:** `is_fraud=True`, `attack_family="AUTHORIZED_PUSH_PAYMENT"`.
- **Readiness:** Schemas (HIGH), WorldState (HIGH), Realism (MEDIUM), Novelty (ATO_SPECIFIC).

**2. MONEY_LAUNDERING_MULE (Layering)**
- **Intended Mechanism:** Using multiple accounts to obscure the origin of funds.
- **Objective:** Move illicit funds across the network.
- **Observable Behavior:** Rapid in-and-out transfers, spanning multiple customers.
- **Readiness:** Schemas (MEDIUM - needs network linkage), WorldState (MEDIUM), Realism (LOW - needs network graph validation).

**3. CARD_TESTING (Merchant Abuse)**
- **Intended Mechanism:** Rapid small transactions to verify stolen card numbers.
- **Objective:** Validate stolen credentials.
- **Observable Behavior:** High velocity, low-amount declines followed by approvals.
- **Readiness:** Schemas (LOW - missing `Card` entities and merchant context).

**4. IDENTITY_ABUSE (Synthetic/Application Fraud)**
- **Intended Mechanism:** Using fake/stolen details to open accounts.
- **Objective:** Acquire lines of credit.
- **Observable Behavior:** Account creation events, KYC flags.
- **Readiness:** Schemas (LOW - missing application/KYC lifecycle events).

## 2. Architecture Reuse Analysis
Before building another family, we must evaluate the ATO architecture for modularity:
- **StatefulSimulator:** `REUSABLE_WITH_EXTENSION` (The core event loop is reusable, but specific event synthesizers must be abstracted).
- **AttackPlan & AttackSignature:** `REUSABLE` (Graph-based state machine is generic).
- **VariationProfile:** `REUSABLE_WITH_EXTENSION` (Need different axes for different families).
- **NoveltyIndex:** `REUSABLE` (The algorithm is generic).
- **AttackFingerprint:** `ATO_SPECIFIC` (Currently hardcodes device patterns and phase assumptions specific to ATO).
- **RealismValidator:** `REUSABLE_WITH_EXTENSION` (Core loop is reusable, but behavioral expectations differ drastically by family).
- **Observable/GT Isolation:** `REUSABLE`.

## 3. Attack-Family Feasibility Matrix
| Family | Schema readiness | WorldState readiness | Realism readiness | Novelty readiness | Implementation effort | Research value | Observable richness |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **AUTHORIZED_PUSH_PAYMENT** | HIGH | HIGH | MEDIUM | BLOCKED (Needs new FP) | MEDIUM | HIGH | MEDIUM |
| **MONEY_LAUNDERING_MULE** | MEDIUM | MEDIUM | LOW | BLOCKED | HIGH | HIGH | HIGH |
| **CARD_TESTING** | LOW | LOW | LOW | BLOCKED | HIGH | MEDIUM | HIGH |
| **IDENTITY_ABUSE** | LOW | LOW | LOW | BLOCKED | HIGH | MEDIUM | MEDIUM |

## 4. Selection: AUTHORIZED_PUSH_PAYMENT (APP)
**Selected Candidate:** `AUTHORIZED_PUSH_PAYMENT` (APP)
**Why it was selected:**
1. It perfectly leverages the existing `ObservableAttackTrace` schemas (Sessions, Beneficiaries, Transactions) without requiring architectural rewrites to `WorldState`.
2. It tests a critical detection boundary: distinguishing between unauthorized ATO (new device, stolen creds) and authorized scams (known device, coerced user).
3. It provides immense research value because APP fraud is the fastest-growing payment threat globally.
4. It avoids the massive scope creep of implementing merchant networks or KYC lifecycles required for Card Testing or Identity Abuse.

**Deferred Families:**
- `MONEY_LAUNDERING_MULE`: Deferred until multi-actor trace generation is supported.
- `CARD_TESTING`: Deferred until `Card` and `Merchant` entities are added.
- `IDENTITY_ABUSE`: Deferred until `Application` schemas are added.

## 5. Redundancy Analysis (ATO vs APP)
APP is *not* a minor variation of ATO. The fundamental distinction lies in the **Session and Device context**.
- **ATO:** Attacker uses a *new* device, creating an anomaly in the session layer before the transaction layer.
- **APP:** Victim uses their *own* primary device. The anomaly exists *only* in the beneficiary and transaction layers (e.g., sudden transfer to a high-risk entity).
This requires completely different behavioral profiles and realism checks.

## 6. Stage 25 Implementation Specification (AUTHORIZED_PUSH_PAYMENT)
- **Attack family:** `AUTHORIZED_PUSH_PAYMENT`
- **Attacker objective:** Manipulate victim into sending funds.
- **Victim/context:** A legitimate customer using their primary registered device.
- **Entry conditions:** Account has sufficient balance, victim is logged in on a known device.
- **Attack phases:** `SOCIAL_ENGINEERING` (Off-platform) -> `SESSION_ACTIVE` -> `BENEFICIARY_ADDITION` -> `EXPLOITATION` -> `END`
- **Possible phase transitions:** Linear progression, or early termination if victim gets suspicious.
- **Observable events:** `SESSION_LOGIN`, `BENEFICIARY_ADDITION`, `TRANSACTION`
- **Ground-truth fields:** `is_fraud=True`, `attack_family="AUTHORIZED_PUSH_PAYMENT"`
- **State mutations:** Balance deductions.
- **Relationship mutations:** New beneficiary relationship created.
- **Transaction behavior:** High amounts, often single large transfers (scammers want money quickly before the victim realizes).
- **Timing behavior:** `RAPID` to `NORMAL` (scammer stays on the phone with the victim).
- **Failure modes:** Insufficient funds, or victim drops off.
- **Difficulty interpretation:**
  - `EASY`: Blatant high-value transfer to a brand new beneficiary.
  - `HARD`: Smaller, layered transfers to a beneficiary masquerading as a utility or known business.
- **Novelty dimensions:** Amount, split count, timing, and beneficiary type.
- **Realism dimensions:** MUST originate from a known device. Realism must REJECT any APP trace that originates from a brand-new device (as that would be ATO).
- **Required schemas:** No new schemas required.
- **Required simulator changes:** Need an `APPAttackFingerprint`. `StatefulSimulator` needs a mechanism to select the *primary* device instead of generating a new one.
- **Required tests:** Ensure APP traces isolate correctly and do not overlap with ATO fingerprints.

## 7. Data / Evidence Requirements
- **Session Origination:** `DOMAIN_MODELED` (Assume 100% known device for APP).
- **Transaction Amounts:** `DOMAIN_MODELED` (Typically 80%+ of available balance).
- **Timing:** `DOMAIN_MODELED` (Minutes between addition and transfer).
- **Beneficiary Risk:** `UNKNOWN` (No public dataset maps exact risk scores for APP beneficiaries).

## 8. Integration Safety & Regressions
Adding APP must not break ATO. 
**Required Regression Tests:**
1. `test_ato_corpus_generation_unaffected()`
2. `test_observable_isolation()`
3. `test_realism_validator_enforces_ato_device_rules()`
4. `test_attack_signature_parsing_multi_family()`

## 9. Final Decision
**Verdict: READY_FOR_STAGE_25**
The architecture is modular enough to accept a second family. APP is the optimal choice for the next vertical slice, providing high research value with bounded implementation effort.
