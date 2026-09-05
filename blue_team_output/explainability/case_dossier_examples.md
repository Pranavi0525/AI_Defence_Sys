# Case dossiers -- investigation aid only

These summaries help a human reviewer act faster on a decision already made by decision_policy.py. They never make or override that decision.

## easy ato correctly blocked

DECISION: BLOCK -- trace atk-73100246 (customer ace3b175-db94-4ca5-9d87-34d64fd6b993), $33,647.49 moved. Attack family: ACCOUNT_TAKEOVER (easy). Liable side under current policy: SENDING (acting side: SENDING). Signals that drove this decision: largest transaction amount, shortest gap between two transactions, length of the observation window. No graph-connected similar past cases found for this trace. NOTE: this dossier is an investigation aid only. It summarizes why the automated policy made its decision; it does not itself make or override that decision.

## hard app correctly blocked

DECISION: BLOCK -- trace atk-57b9b051 (customer 4a3172b4-c168-4e46-a2a2-f7a6442e491b), $12,359.16 moved. Attack family: AUTHORIZED_PUSH_PAYMENT (hard). Liable side under current policy: SHARED_50_50 (acting side: BOTH). Signals that drove this decision: shortest gap between two transactions, average gap between transactions, number of failed transaction attempts. No graph-connected similar past cases found for this trace. NOTE: this dossier is an investigation aid only. It summarizes why the automated policy made its decision; it does not itself make or override that decision.

## mule ring member rescued by fusion

DECISION: BLOCK -- trace atk-ffc1c71f (customer b575ddfd-2001-4f40-bd67-cf26de6aeb97), $8,640.99 moved. Attack family: MULE_NETWORK (medium). Liable side under current policy: RECEIVING (acting side: RECEIVING). Signals that drove this decision: largest transaction amount, shortest gap between two transactions, number of failed transaction attempts. Similar past cases (graph-connected fraud traces): atk-7ddf88bd (customer 63f658c2-3b97-4cde-b5e5-b0a240494c1b), atk-b6ddddef (customer 5babe877-a3bd-47d7-bdcc-c3cdc1e4c08e). NOTE: this dossier is an investigation aid only. It summarizes why the automated policy made its decision; it does not itself make or override that decision.

## fraud case routed to review

DECISION: REVIEW -- trace atk-87e38be9 (customer 259fc45e-741c-4a4a-8d33-fe3f103add9a), $0.00 moved. Attack family: ACCOUNT_TAKEOVER (medium). Liable side under current policy: SENDING (acting side: SENDING). Signals that drove this decision: number of new payees added, number of new-device registrations, shortest gap between two transactions. No graph-connected similar past cases found for this trace. NOTE: this dossier is an investigation aid only. It summarizes why the automated policy made its decision; it does not itself make or override that decision.

## legitimate case routed to review

DECISION: REVIEW -- trace legit_sess_1139069f-d729-4e28-b894-28ae9d50e895 (customer f4237624-2ca8-41a9-a350-3592873b1cd3), $6,276.25 moved. True label: legitimate -- this is a false positive under the current thresholds. Liability fields don't apply. Signals that drove this decision: shortest gap between two transactions, number of failed transaction attempts, average gap between transactions. No graph-connected similar past cases found for this trace. NOTE: this dossier is an investigation aid only. It summarizes why the automated policy made its decision; it does not itself make or override that decision.
