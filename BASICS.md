# AI Defence System - the basics, in plain language

## The problem
A bank sees a stream of events: logins, new devices, new payees, payments. Fraudsters do things that look different
from normal customers. The system must decide for each "episode" (a trace): ALLOW, REVIEW (send to a human), or BLOCK.

## Why a simulator?
Real fraud data is private. So the project builds a fake bank world:
- **Normal World**: simulated ordinary customers (the legitimate traffic).
- **Red Team**: scripted attackers that inject fraud. Three kinds: account takeover (a criminal uses your account),
  authorized push payment (YOU are tricked into paying), mule network (accounts that pass stolen money along).
- Each trace is stored in two parts: what the bank can SEE (events) and the hidden TRUTH (which attack). The model may
  only use what is seen. This is the "no ground-truth leakage" rule.

## The defence, stage by stage (what each part is FOR)
1. **Features**: turn events into numbers (how fast were the payments, how big, was a new payee added just before...).
2. **Stage 1 rules**: cheap filter that lets obviously-fine traffic skip the expensive model.
3. **Stage 2 XGBoost**: the main detector. Learns from labelled examples. Strong on fraud it has seen. 
4. **Graph features**: "does this device/payee/merchant connect to many other customers?" Finds rings.
5. **Novelty detectors (Isolation Forest, autoencoder)**: trained on normal traffic only; flag anything unusual. 
   They are the only part that can notice fraud nobody labelled - but unusual is not the same as fraudulent.
6. **Fusion**: combine the scores. (Finding: it does not help automatically.)
7. **Decision policy**: pick the ALLOW/REVIEW/BLOCK thresholds by minimizing expected cost, not by maximizing accuracy:
   a missed fraud costs a lot, a manual review costs a little, blocking a good customer costs friction.
8. **SHAP**: shows which features pushed a decision, so a human can check it.
9. **Miss collection / feedback loop**: study what was missed and train on harder examples.

## Words you must be able to say
- **Precision**: of the things I flagged, how many were really fraud. **Recall**: of the real fraud, how many I flagged.
- **Base rate / prevalence**: real fraud is maybe 0.6% of traffic, not 24%. At low prevalence, precision collapses.
- **Leakage**: the model sees something that secretly encodes the answer.
- **Calibration**: a score of 0.8 should mean about 80% of such cases are fraud.
- **Cross-validation**: train on part of the data, test on the rest, repeat. **Grouped** CV keeps one customer entirely in one side.
- **Hard negatives**: legitimate behaviour that looks like fraud (many payments in a row, a new payee, a new phone).
- **Leave-one-family-out (LOFO)**: train without one fraud type, test on it. The closest thing here to "new fraud".

## What the project honestly is
A layered fraud-risk PROTOTYPE on SYNTHETIC data. It shows the architecture banks use (rules + ML + graph + novelty +
cost-based decisions + explanations). It is NOT evidence of real-world accuracy and NOT yet a GenAI-fraud detector.
