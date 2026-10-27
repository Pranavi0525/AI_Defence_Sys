# Amount sweep (synthetic model; all other inputs fixed)

### BEFORE (original Phase 10 bundle)
threshold = 0.00194; `*` = FRAUD

| amount (INR) | A familiar ECOM | B new merchant + risky MCC, home | C new merchant + risky MCC, Mumbai | D familiar POS |
|---:|---:|---:|---:|---:|
| 100 | 1.39e-06 | 6.09e-06 | 1.12e-05 | 1.14e-06 |
| 560 | 2.02e-06 | 8.87e-06 | 1.14e-05 | 1.66e-06 |
| 2,000 | 1.72e-05 | 5.48e-05 | 5.90e-05 | 1.05e-05 |
| 5,000 | 1.85e-05 | 3.57e-05 | 7.50e-05 | 1.35e-05 |
| 10,000 | 1.85e-05 | 3.57e-05 | 7.50e-05 | 1.35e-05 |
| 20,000 | 1.85e-05 | 3.57e-05 | 7.50e-05 | 1.35e-05 |
| 50,000 | 1.85e-05 | 3.57e-05 | 7.50e-05 | 1.35e-05 |
| 100,000 | 1.85e-05 | 3.57e-05 | 7.50e-05 | 1.35e-05 |
| 300,000 | 1.85e-05 | 3.57e-05 | 7.50e-05 | 1.35e-05 |
| 1,000,000 | 1.85e-05 | 3.57e-05 | 7.50e-05 | 1.35e-05 |
| 100,000,000 | 1.85e-05 | 3.57e-05 | 7.50e-05 | 1.35e-05 |
| 10,000,000,000 | 1.85e-05 | 3.57e-05 | 7.50e-05 | 1.35e-05 |

### AFTER (retrained bundle)
threshold = 0.04453; `*` = FRAUD

| amount (INR) | A familiar ECOM | B new merchant + risky MCC, home | C new merchant + risky MCC, Mumbai | D familiar POS |
|---:|---:|---:|---:|---:|
| 100 | 1.13e-04 | 6.99e-04 | 3.88e-03 | 4.36e-05 |
| 560 | 1.53e-05 | 9.46e-05 | 1.68e-04 | 6.36e-06 |
| 2,000 | 3.44e-03 | 4.01e-02 | 6.54e-02* | 7.14e-04 |
| 5,000 | 2.94e-02 | 1.82e-02 | 7.18e-02* | 5.93e-03 |
| 10,000 | 4.57e-02* | 1.18e-02 | 4.14e-02 | 9.37e-03 |
| 20,000 | 2.48e-01* | 2.79e-02 | 7.86e-02* | 6.10e-02* |
| 50,000 | 2.91e-01* | 3.77e-02 | 1.61e-01* | 7.50e-02* |
| 100,000 | 2.91e-01* | 3.77e-02 | 1.61e-01* | 7.50e-02* |
| 300,000 | 2.91e-01* | 3.77e-02 | 1.61e-01* | 7.50e-02* |
| 1,000,000 | 2.91e-01* | 3.77e-02 | 1.61e-01* | 7.50e-02* |
| 100,000,000 | 2.91e-01* | 3.77e-02 | 1.61e-01* | 7.50e-02* |
| 10,000,000,000 | 2.91e-01* | 3.77e-02 | 1.61e-01* | 7.50e-02* |

Latency of score_one with 7 history rows (30 runs, this sandbox): median 6.2 ms, p90 6.5 ms.

Training amount range: max 465,290 INR (legit 346,753, fraud 465,290). Scores are flat above the largest split the trees learned; the API rejects amounts above the business limit instead of extrapolating.
