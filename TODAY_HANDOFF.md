# SCOUT — 25 October 2026: Handoff for teammates and progress report

This is **local code**, not a GitHub merge or a claim that teammates approved every edit. Review in a task branch and create a real PR.

## Team decision and scientifically defensible interpretation

- The team wants **random cross-validation**, not a manually chosen season for the main train/validation split.
- Our primary method is **randomly shuffled GroupKFold by `player_name`** (5 folds), using a randomized 20%-of-players reserve and the remaining development set. It is random over players and not determined by season, but keeps all seasons of one player in one fold.
- Ordinary shuffled row KFold and chronological expanding windows are available as **diagnostics** to demonstrate how methodological choices affect outcomes.
- This estimates **end-of-season market value using the season's performance**, not a pre-season future value.

## Changes made

- `src/data.py`: same random player grouping, strengthened checks, true missingness restored without editing XLSX, fixed group holdout membership before dropping one unknown target.
- `src/example_baselines.py`: carry-forward start valuation uses median fallback fitted on training fold; no zero-is-missing error.
- `src/model_comparison.py`: tests all regression models consistently *with and without* start-of-season value.
- `src/exp_encoding.py`: professor's encoding ablation (three encodings × Ridge/Random Forest).
- `src/exp_splits.py`: professor's splitting comparison, with explicit player-overlap diagnostic.
- `src/plot_results.py`: reproducible two report-ready charts from new CSV tables.
- `src/scout_analysis.py`: now delegates to current group CV pipeline.
- `src/archive/scout_analysis_legacy_chronological.py`: original alternative pipeline preserved, not erased.
- `README.md`: updated protocol and limitations; `tests/test_scout.py`: five passing tests.
- Regenerated: `results/example_baselines.csv`, `results/model_comparison.csv`, `results/encoding.csv`, `results/splits.csv`.
- New plots: `figures/progress_models_mae.png`, `figures/progress_encoding_mae.png`.

## Verification run on local environment

- Five automated tests: PASS.
- Raw observations: 1,382; evaluated records: 1,381; development: 1,102 rows and 523 players; reserved group holdout: 279 rows and 132 players.
- Missing start values properly identified: 10. One unknown target excluded from development.
- All data/model experiment commands completed and saved CSVs without accessing/scoring holdout.

### Highlights (value in millions; currency unverified)

Five-fold shuffled GROUPED CV, no start valuation:

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Median baseline | 16.617 | 24.022 | -0.107 |
| Linear Regression | 10.888 | 14.757 | 0.551 |
| Ridge alpha=10 | 10.620 | 14.549 | 0.569 |
| Random Forest unlimited depth | 10.377 | 14.916 | 0.557 |

With starting valuation:

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| Carry forward starting value (training median fallback for missing) | 5.492 | 8.890 | 0.840 |
| Linear Regression | 5.241 | 7.535 | 0.883 |
| Ridge alpha=10 | 5.047 | 7.389 | 0.887 |
| Random Forest unlimited depth | 4.853 | 8.038 | 0.873 |

For the professor's two experiments:

- **Encoding:** grouped rare categories hardly differs from full one-hot for Ridge (MAE 10.620 vs 10.636) or Random Forest (10.849 vs 10.846); dropping nationality/club worsens MAE to 12.366 (Ridge) / 12.288 (forest).
- **Splits:** Ridge grouped 10.620 MAE and ZERO repeat-player overlap; ordinary shuffled KFold 10.278 MAE but ~136 shared player identities per fold; chronological 10.393 MAE but four folds with different training sizes. **Not apples-to-apples, not proof of leakage magnitude.**

**Do not claim:** The above MAEs are independent final-test results; that Ridge/random forest differences are statistically significant; that GBP is confirmed; or that all 2023 observations were never evaluated. They are CV model-selection results on DEVELOPMENT only.

## Suggested today's team workflow

1. Review these diffs on a real GitHub branch; do not overwrite teammates' uncommitted work.
2. All members run: `python -m unittest discover -s tests -v`, then the four experiment scripts and plotting script.
3. Decide which 4–6 rows form the one main report comparison table and what to say about the professor feedback.
4. Draft the 2-page progress report from **verified outcomes**, with figures and an AI-use declaration detailing tool, purpose, changes and independent human checks.
5. Make honest Git commits and get teammate review; do not fabricate commit history.

## Commands

```bash
python -m unittest discover -s tests -v
python src/data.py
python src/example_baselines.py
python src/model_comparison.py
python src/exp_encoding.py
python src/exp_splits.py
python src/plot_results.py
```
