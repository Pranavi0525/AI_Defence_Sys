# Phase 7 -- model answers (try DEFENCE_NOTES.md first, then compare)
Numbers below are read from reports/v2/feedback_loop_*.json by code when this file was written; regenerate the reports and re-check them if you re-run.

**1. What is a case dossier and why six parts?**
It is the one page an analyst opens for a REVIEW/BLOCK. Each part answers a different question: what was decided and how close to the line (decision), what each model said (model view), what pushed XGBoost (drivers), what a human can verify in the raw trace (evidence), who else is linked (graph), and who bears the loss (acting party). I label each part's source (model output / rule-of-thumb / policy mapping) so a reader never mistakes a hand-written flag for a model result.

**2. Why is the acting-party block empty by default?**
The fraud family is not known when a trace is scored. Filling it in would mean guessing, and a wrong liability story is worse than none. It fills only when an analyst triages the case and supplies the family. Failure mode if I had guessed: confident but unsupported claims in front of an analyst.

**3. Which liability mappings are real and which are assumptions?**
ACCOUNT_TAKEOVER, AUTHORIZED_PUSH_PAYMENT and MULE_NETWORK reuse the v1 mapping from decision_policy.py (a test fails if the two ever differ). The four new families (SYNTHETIC_IDENTITY, FAKE_MERCHANT, DEEPFAKE_KYC, AI_SCRIPTED_APP) are my assumptions, labelled ASSUMED, not a reading of any regulation.

**4. Evidence flags vs model drivers: what is the difference?**
Drivers are XGBoost's own log-odds contributions for this trace, so they explain the model. Evidence flags are fixed checks I wrote (payee paid within an hour, virtual camera, account under 90 days, ...) with cut-offs I did not tune. They help a human verify the case but say nothing about why the model scored it. They can disagree, and the dossier says when no flag fired.

**5. A driver shows value "missing" and raises risk. What does that mean?**
The feature was not present in the trace (for example no merchant payment, so no merchant age). XGBoost treats missing as information, and in my simulator "no merchant purchase" correlates with fraud. That is a simulator artifact, so the dossier warns to treat it as weak evidence.

**6. How does the feedback loop work and why is there a control arm?**
Round-0 model, out-of-fold mistakes on the training part only, fresh look-alike cases from the generator, retrain, test on an untouched customer-grouped quarter, repeated over 10 worlds. The control arm adds the same number of fresh traces (141 fraud + 35 legit on average) from the ordinary mix. Without it, "targeted beats baseline" could just mean "more data helps". The honest comparison is targeted minus control.

**7. What did the loop show?**
Small effects that are inconsistent across settings and not distinguishable from noise. Cost per 10k transactions, targeted minus the same-size control (negative = targeted better), with the sd across 10 worlds and the number of worlds where targeted was better:
- main: 2 fresh cases per mistake, all misses found: -232 (sd 1,135), better in 7 of 10 worlds; vs baseline -739 (sd 1,693).
- only 50% of misses found: -1,128 (sd 1,319), better in 8 of 10 worlds; vs baseline -517 (sd 1,645).
- 4 fresh cases per mistake: +292 (sd 1,284), better in 3 of 10 worlds; vs baseline -248 (sd 1,778).

Two of three settings lean slightly in favour, one leans against, and in every setting the sd is larger than the mean. Taken alone, the 50% setting is about 2.7 standard errors from zero, but I looked at three settings and many metrics, so that is not a finding. Relative to baseline, targeted changed by -517 and the control by +611, so about half of that gap is the control getting worse, not targeted getting better. Say "inconclusive: small and inconsistent, no detectable benefit", never "the feedback loop improved the model".

**8. Why would learning from mistakes not help here?**
(a) Almost all misses are stealth attackers who copy legitimate behaviour (reports/v2/feedback_miss_audit.md: 94% of misses are stealth, stealth miss rate 47% vs 3% for plain), so fresh look-alikes of a miss are mostly again hard-to-see cases the features cannot separate. (b) Only about 74% of fraud is flagged to begin with, and the training set is small (about 1,000 traces). (c) The thresholds are learned on small out-of-fold samples and move a lot between retrains (baseline review threshold ranged from 0.56 to 0.77 across worlds), which adds noise bigger than the effect. (d) The generator is the same one that made the test set, which is the easiest setting for a feedback loop. This run therefore gives no evidence that the loop would help on real data; it only shows it did not clearly help even in the favourable case.

**9. What would you do differently with real analysts?**
Keep the human's label and timestamp, version every retrain, evaluate on time-ordered data (train earlier, test later), track label delay, and compare each retrain against a same-size control. Treat the threshold as a separate decision from the model and re-tune it with enough volume to be stable.

**10. What does this loop NOT address?**
Unseen fraud families (Phase 2/5 showed supervised recall near zero on a held-out family), real analyst noise, and the simulator artifacts. A feedback loop can only teach the model about kinds of fraud someone has already seen and labelled.
