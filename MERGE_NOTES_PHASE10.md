# Phase 10 merge notes

The Phase 10 card-transaction overlay has been merged into the Phase 8b project.

Added/updated files include:
- `v2/card/__init__.py`
- `v2/card/sim.py`
- `v2/card/rules.py`
- `v2/card/evaluate.py`
- `v2/card/features.py`
- `tests/test_card_phase10.py`
- `scripts/run_card_phase10.py`
- `scripts/run_card_sparkov.py`
- `reports/v2/phase10_card.md`
- `reports/v2/phase10_card_novel.md`
- `reports/v2/phase10_card_overlay_reference.json` (retained source-overlay JSON; not represented as the user's latest run)
- `reports/v2/phase10_card_novel.json`
- `reports/v2/phase10_sparkov.json` (created when the Sparkov script is run)
- `reports/v2/phase10_card_completed_run_2026-10-26.md` (exact summary from the user's completed 3-seed run)
- `data/sparkov/README.md` (instructions for optional dataset placement)

Configuration adjustments:
- Phase 10 XGBoost uses `tree_method="hist"` and `n_jobs=1`, matching the successful completed run and avoiding the slower exact tree method.
- Sparkov feature caching uses the OS temporary directory for Windows/Linux/macOS compatibility.

Validation previously recorded for the base merged project:
- Focused Phase 10 tests: 2 passed.
- Full project test suite: 594 passed.

The script writes a fresh `reports/v2/phase10_card.json` when run. The latest completed run's summary is in `reports/v2/phase10_card.md`. The final merge has been checked for Python syntax and the focused Phase 10 tests. See the completed-run report for the user's latest synthetic three-seed results.

Important: Phase 10 simulator results are synthetic. Sparkov files are not included in this bundle; check the original dataset license before redistributing raw CSVs. Treat these results as prototype/evaluation evidence, not proof of real-world fraud performance.
