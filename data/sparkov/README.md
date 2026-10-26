# Sparkov transaction dataset (optional Phase 10 real-ish-data evaluation)

The project includes `scripts/run_card_sparkov.py`, which expects these two files:

- `fraudTrain.csv`
- `fraudTest.csv`

Place them in this folder after extracting the separately supplied dataset archive, then run from the repository root:

```cmd
python scripts\run_card_sparkov.py data\sparkov
```

The script writes `reports/v2/phase10_sparkov.json`.

**Dataset handling:** The raw CSVs are intentionally not embedded in this project bundle. They are large third-party dataset files, and their redistribution/licensing terms should be checked against the original source before sharing or committing them. Keep the original archive separately and retain its source/attribution information. The dataset is Sparkov-generated/simulated transaction data with realistic fields and geography; do not describe it as actual bank production transactions.

The script's feature cache uses the operating system's temporary directory.
