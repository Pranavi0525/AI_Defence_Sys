# Defence notes - AI Defence System (answer in your OWN words; write STUCK if you can't)

## Template for every component
1. What does it do (one sentence)? 2. Why this, not the obvious alternative? 3. How can it fail, and where do my results show it? 4. What number proves it works, and what would it look like if wrong?

## Phase 0 / 1.1
- Why freeze v1? Why pin versions? Why 5 seeds? What did the customer-grouped split show (88% overlap, F1 0.980 vs 0.981)?

## Phase 1 (evaluation)
- What was wrong with `failed_transaction_count`? Why did fixing it barely change F1?
- Which raw fields are perfect label proxies? Why don't v1 features exploit them? What would happen if a model read them?
- Why is the Stage 1 trade-off a curve (keep 90/95/99% of fraud) and not one number? What did v1's rules do differently?
- Calibration: raw XGBoost had a LOWER Brier score than sigmoid or isotonic. Why might calibrating hurt on ~1,500 rows?
- Why does precision at 0.6% prevalence (~0.43) differ so much from precision on the synthetic mix (~0.97)? Compute it by hand from TPR and FPR.
- Why a customer-cluster bootstrap instead of a row bootstrap?

## Phase 2 (new fraud)
- Why did supervised recall on an unseen family drop (MULE 0.23, FAKE_MERCHANT 0.00)?
- Why did the novelty detectors catch DEEPFAKE_KYC (~100%) but not FAKE_MERCHANT (~0-8%)?
- Why is "unusual" not "fraudulent"? What does the ~2% legit flag rate cost operationally?
- Why is the A->B transfer test confounded (legit AUC 0.996)? What does the 36.8% FPR tell you?

## Phase 3 (generators) - be ready to say this plainly
- What are hard negatives and why did I add them? Why can't I claim the new families are realistic? Why is World B "too easy"?

## Phase 4 (graph)
- What does each graph feature count? Why computed without labels? What is the transductive caveat? Gain: F1 0.975->0.979 (A).

## Phase 5 (fusion + policy)
- Why must thresholds be learned on inner out-of-fold scores of the training fold only?
- Why was the stacker equal to XGBoost in-distribution? Why was naive max much worse?
- What is the cost model and which numbers are assumptions? Why do thresholds sit in the extreme tail when legit scores are tiny?
- Why did the novelty-aware stacker help DEEPFAKE_KYC and MULE but hurt SYNTHETIC_IDENTITY? What does that say about "one fusion for all"?
- LOFO table (fixed 0.5) vs unseen-policy table (cost-optimised low review threshold) disagree about unseen recall. Why? (operating points)

## Making World B hard (new)
- Name the five giveaways I found in my own simulator and the fix for each. Why is "I audited my own simulator" a strength, not an embarrassment?
- What does `stealth` mean, and why report a sweep instead of one number?
- Why were merchants excluded from the connected-component feature? (hub nodes)
- Why is "new payee then pay" still a strong single feature even after hardening? Is that a flaw or realistic?
- World B legit "new payee then pay" is flagged 29% of the time. What would that cost a real bank, and what would you do about it?
- Why was the novelty-aware stacker a NEGATIVE result in World B? What would you try next?
- Why is precision at 0.6% prevalence only ~0.10 in World B while precision on the synthetic mix is ~0.88? Compute it from TPR and FPR.

## Phase 6 (serving)
- Why is the graph feature computed from history only? What would be wrong with a batch graph over all traces at serving time? What happened to the World A graph gain once I fixed it?
- What is "train/serve skew"? Which test proves there is none for graph features, and how does it work?
- Why export Isolation Forest and the autoencoder as arrays instead of pickling scikit-learn objects? What did the equivalence test compare?
- Why does `commit=true` return 403 by default? What attack does that prevent?
- What do PSI values of 0.05, 0.15 and 0.4 mean, and why does new fraud show up as drift first?
- The AI-scripted APP example scored 0.736 against a review threshold of 0.758 and was ALLOWED. What would you change and what would it cost (hint: threshold, prevalence assumption, review capacity)?
- A top feature is shown with value null. What does that mean, and why is "absence of a merchant" a simulator artifact?
- What is NOT protected in this API (auth, rate limits, multi-worker state), and how would you fix each?

## Phase 7 (case dossiers + feedback loop) - model answers are in DEFENCE_MODEL_ANSWERS_PHASE7.md; try these first
- What is in a case dossier and which parts are model output, which are rule-of-thumb flags, which are policy mapping? Why label each?
- Why is the acting-party block empty until an analyst supplies the fraud family? Which three liability mappings are v1 policy and which four are my assumptions?
- A driver shows value "missing" and raises risk. What does that mean and why is it partly a simulator artifact?
- Walk through the feedback loop step by step. Why use ONLY out-of-fold mistakes on the training part? Why is there a control arm?
- What did the loop show? State it in one honest sentence, then give the number from reports/v2/feedback_loop.md that supports it.
- Why might learning from mistakes not help here (name three reasons)? What would you do differently with real analysts and real labels?
- What does the feedback loop NOT address? (hint: unseen families, label delay, simulator artifacts)

## Phase 9 (packaging) - answers are in INTERVIEW_KIT.md section 4; try these in your own words first
- In one sentence each: what is real, what is offline-only and what is served? Which parts of the project are NOT in v2 serving?
- Your v1 README said F1 0.976 and World B says 0.824. Is v2 worse? What exactly changed, and which fixes did NOT change the score?
- Why do you report precision at 0.6% prevalence, and why can that number never be called a real-world result?
- Give the honest answer to "does it detect GenAI fraud?" and name two things you must not say in an interview.
- Pick one resume bullet and trace every word of it to a file in the claims ledger.
- Which sentence of your cold email can a recruiter verify on your GitHub in under a minute?
