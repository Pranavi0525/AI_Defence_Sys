# Stage 16 Transaction Outcome Assessment

## A. What does an approved transaction currently mean?
An approved transaction currently means the `Transaction` entity is created (implicitly successful), and its `amount` is unconditionally subtracted from `account.balance` in the `WorldState`. The `ObservableTransactionEvent` is generated with `transaction_status` mirroring the transaction's status.

## B. What does an attempted-but-unsuccessful transaction currently mean?
Currently, there is no attempted-but-unsuccessful transaction generated natively by the ATO simulator. When an attack transfer amount exceeds the available balance, the simulator forcibly injects a fake "top-up" (`acct.balance += amt * 2`) to ensure the transaction succeeds. 

## C. Is there already a schema field capable of representing transaction outcome?
Yes. The `Transaction` schema in `src/red_team/schemas/entities.py` already includes:
`status: Literal["pending", "completed", "failed", "reversed"]`
Furthermore, `ObservableTransactionEvent` in `src/red_team/schemas/observable.py` includes:
`transaction_status: str`
Thus, the schema already natively supports recording the outcome of a transaction.

## D. What is the minimum architectural change needed?
1. Modify `src/red_team/attacks/simulator.py` to stop artificially topping up balances. Instead, if `acct.balance < amt`, set the transaction `status = "failed"`, do NOT mutate `acct.balance`, and record `pre_balance == post_balance`. If `acct.balance >= amt`, set `status = "completed"` and deduct the balance.
2. Modify `src/red_team/validation/realism.py` to respect `transaction_status`. If a transaction is "failed" (DECLINED), the validator should NOT subtract its amount from the tracked balance, and should NOT raise a `BALANCE_CONSTRAINT_VIOLATION` for insufficient funds (since failing was the correct physical outcome). If it is "completed" (APPROVED), it must enforce `amt <= balance` and then subtract `amt`.
3. Normal World (`behavior.py`) does not strictly require modification because its auto-top-up behavior legitimately simulates external income/deposits that keep customer accounts afloat. 

## E. Should outcome be represented as:
An enum field. Specifically, the existing `status` field on the `Transaction` entity. We will map `completed` to mean APPROVED, and `failed` to mean DECLINED. No new schemas or event types are needed.
