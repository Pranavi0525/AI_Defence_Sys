# Phase 8 -- external METHOD check on a public transaction dataset

**What this is:** a check of the modelling method from Phases 1-5 on real public data. **What it is not:** a test of the AI Defence System.
The public data has no sessions, devices, payees or graph, so Stage 1, graph features, novelty stack and dossiers do not apply.
The one-sentence honest claim after you run it: *"I also checked the Stage 2 modelling method on a public card-fraud dataset with a
chronological split; it is a method check, not the same system."* Do not say "validated on real data".

## Questions it answers (and the ones it cannot)
| Question | Where it is answered |
|---|---|
| Does a chronological split give a different answer from a random split? | `time split` vs `random split` tables, with bootstrap interval |
| Does calibration (sigmoid / isotonic, fitted on a LATER validation block) help or hurt? | Brier and ECE columns |
| What recall do we get at a fixed alert budget (0.1%, 0.5%, 1% of traffic), and what is precision at the natural 0.17% fraud rate? | `recall @...% alerts`, `precision @0.5% alerts` |
| Is XGBoost better than logistic regression? | `logreg` rows |
| Does the pipeline leak labels? | Leakage control: training on shuffled labels must score near the base rate |
| Does the score hold up across time blocks? | Rolling-origin table (small fraud counts, look at the spread) |
| Customer leakage, graph value, GenAI-era fraud, simulator realism | NOT answerable here: no customer id, no graph, no such fraud |

## Get the data (you must do this; it needs a free Kaggle account)
1. Open https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and download `creditcard.csv` (about 150 MB; 284,807 rows, 492 frauds, columns Time, V1-V28, Amount, Class).
2. Put it at `data\public\creditcard.csv` in the repo. Keep it out of git: `echo data/public/>> .gitignore`
3. The data is released under the Database Contents License (DbCL) v1.0; cite the source (Worldline and ULB Machine Learning Group) if you mention it.

## Run (cmd; each stage takes seconds to a few minutes)
```
set PYTHONPATH=src
python scripts\run_phase8_public_check.py --stage main
python scripts\run_phase8_public_check.py --stage control
python scripts\run_phase8_public_check.py --stage rolling
python scripts\run_phase8_public_check.py --stage report
python scripts\build_phase9_docs.py
```
The last command adds an "External method check" section to `README_V2.md` and updates the "do not say" list in `INTERVIEW_KIT.md`.
Its sentences are chosen by comparing your numbers (for example it says "tie" if the three Brier scores are within 5%), so it cannot claim a result you did not get.

## Verified here vs not verified here
- Verified in my sandbox: the code runs end to end on a full-size synthetic stand-in with the same columns; split logic, metrics, calibrators, the label-shuffle control, rolling origin and the loader are unit-tested (`tests/test_phase8_public_check.py`).
- NOT verified: any result on the real dataset. My sandbox cannot reach Kaggle, so no real number exists yet. Whatever you get is the first time anyone has seen it.

## Honest limits
About two days of data (a chronological split tests short-term ordering only); no card or customer id (the same cards may sit on both sides of any split);
PCA-anonymised features; tens of frauds per test block, so intervals are wide and the bootstrap over rows ignores dependence;
Stage 2 hyper-parameters are not tuned for this data and use `tree_method=hist` for speed. A high score on this dataset is easy and proves little alone.

## Questions to answer after you run it (use YOUR numbers from reports/public/phase8_report.md; copy them into DEFENCE_NOTES.md when you answer)
- Why is this called a "method check" and not "validation of the system"? Name three parts of the system it cannot test.
- What does the label-shuffle control prove, and what number would make you distrust the pipeline?
- Time split vs random split: what did you see, how big were the bootstrap intervals, and why can you not conclude much from the gap?
- Did calibration help or hurt on real data? Does that repeat your synthetic finding, and what is one reason it might differ?
- Why report recall at an alert budget instead of accuracy or F1 for a 0.17% fraud rate?
- Why was customer-grouped splitting impossible here, and what does that leave unanswered?
