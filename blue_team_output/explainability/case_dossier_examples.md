# Case dossiers -- investigation aid only

These summaries help a human reviewer act faster on a decision already made by decision_policy.py. They never make or override that decision.

## easy ato correctly blocked

DECISION: BLOCK -- trace atk-73100246 (customer ace3b175-db94-4ca5-9d87-34d64fd6b993), $33,647.49 moved. Attack family: ACCOUNT_TAKEOVER (easy). Liable side under current policy: SENDING (acting side: SENDING). Signals that drove this decision: average transaction amount, largest transaction amount, shortest gap between two transactions. No graph-connected similar past cases found for this trace. NOTE: this dossier is an investigation aid only. It summarizes why the automated policy made its decision; it does not itself make or override that decision.

## hard app correctly blocked

DECISION: BLOCK -- trace atk-57b9b051 (customer 4a3172b4-c168-4e46-a2a2-f7a6442e491b), $12,359.16 moved. Attack family: AUTHORIZED_PUSH_PAYMENT (hard). Liable side under current policy: SHARED_50_50 (acting side: BOTH). Signals that drove this decision: shortest gap between two transactions, number of failed transaction attempts, average gap between transactions. No graph-connected similar past cases found for this trace. NOTE: this dossier is an investigation aid only. It summarizes why the automated policy made its decision; it does not itself make or override that decision.

## fraud case routed to review

DECISION: REVIEW -- trace atk-fc93ce72 (customer 2444528b-4509-4809-a516-d55fbd5d508d), $0.00 moved. Attack family: ACCOUNT_TAKEOVER (advanced). Liable side under current policy: SENDING (acting side: SENDING). Signals that drove this decision: shortest gap between two transactions, average gap between transactions, number of new payees added. No graph-connected similar past cases found for this trace. NOTE: this dossier is an investigation aid only. It summarizes why the automated policy made its decision; it does not itself make or override that decision.

## legitimate case routed to review

DECISION: REVIEW -- trace legit_sess_dbe8f9d7-59ef-43b1-b376-ac666195f18c (customer 6bec1ab7-0977-4df3-9e84-465a2e698e5f), $11,702.47 moved. True label: legitimate -- this is a false positive under the current thresholds. Liability fields don't apply. Signals that drove this decision: shortest gap between two transactions, number of failed transaction attempts, average gap between transactions. No graph-connected similar past cases found for this trace. NOTE: this dossier is an investigation aid only. It summarizes why the automated policy made its decision; it does not itself make or override that decision.
