# Per-case explainability reports

## easy ato correctly blocked

Trace atk-73100246 (customer ace3b175-db94-4ca5-9d87-34d64fd6b993), true label: FRAUD [ACCOUNT_TAKEOVER, easy], $33,647.49 moved in this trace. Stage 1 (rules) escalated this trace to the ML model because: a payee was added and then paid within the same session; a new device was registered on this trace; two transactions happened under an hour apart; transaction velocity exceeded 2.5/hour. Stage 2 (XGBoost) scored it 0.982. Top contributing signals: number of new payees added (1.0) pushed the score up; average transaction amount (33647.49) pushed the score up; shortest gap between two transactions (960.0) pushed the score down. Stage 3 (graph): this trace has no cross-customer graph connections -- a no-op. Stage 4 (decision policy): final score 0.983 vs. thresholds REVIEW>=0.066 / BLOCK>=0.935 -> decision: BLOCK.

## hard app correctly blocked

Trace atk-57b9b051 (customer 4a3172b4-c168-4e46-a2a2-f7a6442e491b), true label: FRAUD [AUTHORIZED_PUSH_PAYMENT, hard], $12,359.16 moved in this trace. Stage 1 (rules) escalated this trace to the ML model because: two transactions happened under an hour apart. Stage 2 (XGBoost) scored it 0.991. Top contributing signals: shortest gap between two transactions (39.0) pushed the score up; average gap between transactions (141.0) pushed the score up; smallest transaction amount (1927.66) pushed the score down. Stage 3 (graph): this trace has no cross-customer graph connections -- a no-op. Stage 4 (decision policy): final score 0.991 vs. thresholds REVIEW>=0.066 / BLOCK>=0.935 -> decision: BLOCK.

## mule ring member rescued by graph

Trace legit_sess_d6a380f4-b081-4172-ae0a-3b5c8a02327d (customer f99d139c-081c-4ba0-b4eb-b89cf23105c7), true label: FRAUD [ring_mule_synthetic, n/a], $3,466.92 moved in this trace. Stage 1 (rules) escalated this trace to the ML model because: two transactions happened under an hour apart. Stage 2 (XGBoost) scored it 0.001. Top contributing signals: relative spread in transaction amounts (0.143) pushed the score up; length of the observation window (445221.0) pushed the score up; average gap between transactions (85795.8) pushed the score up. Stage 3 (graph): this trace shares an entity (device or payee) with 5 other customer(s)' traces, 5 of which are part of a flagged mule ring. Illustrative graph-propagated score: 0.841 (out-of-fold decision score with graph escalation applied: 0.936). Graph escalation RAISED the final score above Stage 1+2 alone. Stage 4 (decision policy): final score 0.936 vs. thresholds REVIEW>=0.066 / BLOCK>=0.935 -> decision: BLOCK.

## fraud case routed to review

Trace atk-308113e9 (customer b56b42e3-4b67-4eee-a294-17aa8de74cbb), true label: FRAUD [ACCOUNT_TAKEOVER, easy], $0.00 moved in this trace. Stage 1 (rules) escalated this trace to the ML model because: a new device was registered on this trace; two transactions happened under an hour apart. Stage 2 (XGBoost) scored it 0.934. Top contributing signals: number of new payees added (1.0) pushed the score up; shortest gap between two transactions (1440.0) pushed the score down; length of the observation window (1440.0) pushed the score up. Stage 3 (graph): this trace has no cross-customer graph connections -- a no-op. Stage 4 (decision policy): final score 0.932 vs. thresholds REVIEW>=0.066 / BLOCK>=0.935 -> decision: REVIEW.

## fraud case that slipped through as allow

Trace atk-693a3017 (customer e9669dd5-9e35-4c39-901d-6ffb9e8ca7dd), true label: FRAUD [ACCOUNT_TAKEOVER, advanced], $0.00 moved in this trace. Stage 1 (rules) escalated this trace to the ML model because: a new device was registered on this trace; two transactions happened under an hour apart. Stage 2 (XGBoost) scored it 0.083. Top contributing signals: length of the observation window (420.0) pushed the score up; shortest gap between two transactions (420.0) pushed the score up; number of failed transaction attempts (0.0) pushed the score up. Stage 3 (graph): this trace has no cross-customer graph connections -- a no-op. Stage 4 (decision policy): final score 0.057 vs. thresholds REVIEW>=0.066 / BLOCK>=0.935 -> decision: ALLOW.

## legitimate case routed to review

Trace legit_sess_e4033923-6a25-4179-b764-988809100fb7 (customer bdbdb271-084a-4526-b317-d1fdb192051d), true label: legitimate, $0.00 moved in this trace. Stage 1 (rules) escalated this trace to the ML model because: a new device was registered on this trace. Stage 2 (XGBoost) scored it 0.113. Top contributing signals: shortest gap between two transactions (97359.0) pushed the score down; number of new-device registrations (1.0) pushed the score up; length of the observation window (97359.0) pushed the score down. Stage 3 (graph): this trace has no cross-customer graph connections -- a no-op. Stage 4 (decision policy): final score 0.107 vs. thresholds REVIEW>=0.066 / BLOCK>=0.935 -> decision: REVIEW.

## ordinary legitimate autocleared by stage1

Trace legit_sess_d0521084-07b0-4313-bc1e-449db3356b7e (customer e9c8c0c0-89cb-438d-b942-a85203b94ba5), true label: legitimate, $2,978.70 moved in this trace. Stage 1 (rules) did NOT escalate this trace -- none of the four trigger conditions fired, so it was auto-cleared without the ML model ever running. (For reference only, the model's score had it been escalated would have been 0.028.) Stage 3 (graph): this trace has no cross-customer graph connections -- a no-op. Stage 4 (decision policy): final score 0.000 vs. thresholds REVIEW>=0.066 / BLOCK>=0.935 -> decision: ALLOW.

