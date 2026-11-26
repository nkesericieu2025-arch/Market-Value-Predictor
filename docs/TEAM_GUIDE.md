# SCOUT: Predicting Football Players' Estimated Market Values

Machine Learning & Communication Skills, IE University (2026–2027).
Team: Pablo Garma, Noa Keseric, Kristian Spassov.

Repository: https://github.com/nkesericieu2025-arch/Market-Value-Predictor

**Research question:** To what extent can player characteristics and performance
statistics predict football players' estimated market values, and which features
contribute most to predictive performance?

## What this project does, in plain words
Each row of our data is one player in one season: his age, position, club,
nationality, games, minutes, goals, assists and so on, plus his estimated market
value at the start and at the end of that season. We train regression models that
take the player's information and predict his **end-of-season value**
(`value_end_mln`). We then check how far off the predictions are, compare models,
and find out which pieces of information matter most.

## Data
Nisanov, R. *Transfer Value Determinants*, Version 2, Mendeley Data,
doi:10.17632/3btg6ptc7b.2, CC BY 4.0. Stored in `data/` unmodified.
Only the worksheet named `2019` is used: it already combines all 2019–2023 seasons
(1,382 rows, 656 players). The other sheets would duplicate rows.

Expected goals and assists (xG, xA) come from Understat (https://understat.com) via the
public export `understat_players_aggregated_2014_2024.csv` from
https://github.com/vibedatascience/understat_players_aggregated, stored unmodified in
`data/external/`. That repository states no licence: it is credited here, and the team
will confirm redistribution is allowed before the final report.
`src/integrate_understat.py` matches it to our data by normalised player name and season
(93.6% of rows matched; goal counts agree in 99.7% of matched rows).

Ten start values and one end value are empty in the raw columns but stored as 0 in the
`*_mln` columns; `load_data()` restores them as missing, leaving 1,381 usable rows.

Columns we never use as predictors (see `EXCLUDED` in `src/data.py`):
| Column | Why excluded |
|---|---|
| `end_value`, `start_value` | Same numbers as the target and `value_0_mln`, in other units |
| `delta_value`, `value_delta_mln` | Computed from the target, so they would give away the answer |
| `age_now` | Age when the data were collected, i.e. after the season |
| `player_name` | An identifier, not a characteristic (used only to keep players together in splits) |

## Setup (once per person)
```bash
git clone https://github.com/nkesericieu2025-arch/Market-Value-Predictor.git
cd Market-Value-Predictor
conda env create -f environment.yml
conda activate scout
python src/data.py            # must end with "OK"
python src/example_baselines.py   # prints a small results table
```

## Team guide: start here after cloning

### 1. Check that everything works (5 minutes)
```bash
git pull
conda activate scout
python src/data.py                # must end with "OK"
python src/example_baselines.py   # prints a table; linear regression must beat the median baseline
```
If either fails, do not start working: post the full error in the group chat.

### 2. What the shared code does
**`src/data.py`: the data and the splits (everyone uses the same ones)**
| Part | What it does | Why |
|---|---|---|
| `ROOT`, `DATA_PATH` | Finds the data file relative to the code | Works on every laptop without changing paths |
| `COMBINED_SHEET` | The sheet named `2019`, which actually holds all 2019–2023 rows | The season sheets 2020–2023 lack the three club variables, so they are not used |
| `NUMERIC`, `START`, `CATEGORICAL`, `EXCLUDED` | Which columns are predictors and which are never used | Keeps leakage columns out for everyone |
| `load_data()` | Reads the combined sheet, checks 1,382 rows and seasons 2019–2023, and restores the 11 values stored as 0 as missing | Stops immediately if the wrong data is loaded |
| `split_holdout()` | Locks away 20% of *players* (all their seasons) as the final test set, with seed 42 | Same 132 test players for everyone; **never used before the final report** |
| `cv_splits(dev, kind)` | Cuts the remaining 1,102 rows (523 players) into folds | `"group"`: main design, random across seasons, players kept together. `"random"` and `"chronological"`: only for the split experiment |
| `if __name__ == "__main__":` | Smoke test: prints sizes, checks no player is in both dev and test, builds all fold types | Run `python src/data.py` after every `git pull` |

**`src/pipeline.py`: preparation + model as one unit**
| Part | What it does | Why |
|---|---|---|
| `build_pipeline(model, encoding, use_start)` | Fills missing values (median), scales numbers, one-hot encodes categories, then the model | Everything is re-learned on the training folds only, so nothing leaks from validation |
| `encoding=` | `"onehot"` every nationality/club its own column; `"grouped"` categories with < 10 rows merged; `"drop"` nationality and club removed | Professor's sparsity question |
| `use_start=` | Adds the start-of-season value `value_0_mln` as a predictor | Separates performance data from what the market already knew |
| `evaluate(pipe, dev, splits, encoding, use_start)` | Runs cross-validation; returns mean and std of MAE, RMSE, R² | Std = how much the score varies across folds (our uncertainty) |
| `results_table(rows)` | Turns results into a rounded table for saving | Same format in every CSV |

### 3. How to write your experiment script
Copy the pattern of `src/example_baselines.py`:
```python
from sklearn.linear_model import Ridge

from data import ROOT, cv_splits, load_data, split_holdout
from pipeline import build_pipeline, evaluate, results_table

dev, _ = split_holdout(load_data())     # _ = test set, never used here
splits = cv_splits(dev, kind="group")   # same folds for everyone

rows = []
for use_start in (False, True):
    pipe = build_pipeline(Ridge(alpha=1.0), encoding="grouped", use_start=use_start)
    rows.append({"model": "Ridge", "use_start": use_start,
                 **evaluate(pipe, dev, splits, encoding="grouped", use_start=use_start)})

table = results_table(rows)
print(table.to_string(index=False))
table.to_csv(ROOT / "results" / "<your_file>.csv", index=False)
```
Rule: pass the **same** `encoding` and `use_start` to `build_pipeline` and `evaluate`.

### 4. Sanity checks before you share a result
- Every real model has a lower MAE than the median baseline.
- The median baseline has R² around 0 or slightly negative (it explains nothing).
- MAE is a few million to about 15 million. Values like 0.01 or 100 signal a bug.
- The `random` split scores the same as or better than `group`, never clearly worse.
- Re-running the script gives identical numbers (fixed seed).

### 5. Daily workflow
```bash
git pull                                  # get everyone's latest work
git checkout -b <name>/<task>             # once per task, e.g. pablo/models
# ... work, run, check ...
git add src/<your_script>.py results/<your_file>.csv
git commit -m "Describe what changed"     # small steps, several per day
git push -u origin <name>/<task>          # then open a Pull Request on GitHub
```
- Only edit your own files. Changes to `data.py` or `pipeline.py` are agreed in the group chat first.
- Another member reviews and merges every Pull Request.
- Note any AI help (tool, what it did, what you changed, how you checked) for the declaration.

## How we test the models
- **Final test set.** 20% of the *players* (with all their seasons) are put aside
  with a fixed seed in `src/data.py`. Nobody looks at them until the final model is
  chosen, so the final score is an honest estimate on unseen players.
- **Cross-validation on the rest.** The remaining players are split into 5 parts
  (folds). Each model is trained on 4 parts and tested on the 5th, five times, and
  we report the average error and how much it varies.
- **Grouped by player.** A player's seasons always land in the same fold. Otherwise
  the model could see Salah 2021 in training and be "tested" on Salah 2022, which
  makes it look better than it is.
- **Preprocessing inside the pipeline.** Filling missing values, scaling and
  one-hot encoding are re-learned on the training folds every time
  (`src/pipeline.py`), so no information from the test fold leaks in.
- **Metrics.** MAE (average error in millions), RMSE (like MAE but punishes big
  misses more) and R² (share of the variation in values the model explains).
- Random seed: 42 everywhere.

## Models
| Model | Role |
|---|---|
| Median baseline | Same typical value for everyone; any model must beat it |
| Start value carried forward | "Value stays what it was at the start of the season" |
| Start value × (1 + mean / median growth) | Start value grown by the average growth rate of the training players |
| Linear regression | First real model |
| Ridge regression | Linear regression with a penalty that keeps coefficients small |
| Random forest | Many decision trees on bootstrap samples; tuned with out-of-bag (OOB) error |
| Gradient boosting | Trees added one after another, each correcting the previous ones |

Every model is run **with and without** the start-of-season value (`value_0_mln`),
to separate what performance data explain on their own from what the market already knew.

## Scripts and what they produce
| Script | Purpose | Output |
|---|---|---|
| `src/final_analysis.py` | **Main analysis**: every result in the sentence outline (grouped CV, nested tuning, xG/xA, log target, PCA, feature importance, errors by group, case study). ~8 min | `results/final/T1–T8*.csv`, `figures/final/F1–F4*.png` |
| `src/exp_growth_baseline.py` | Percentage-growth baseline | `results/growth_baseline.csv` |
| `src/integrate_understat.py` | Merges xG/xA into the player data | `data/processed/scout_with_understat.csv` |
| `src/model_comparison.py`, `src/exp_encoding.py`, `src/exp_splits.py`, `src/plot_results.py` | Progress-report experiments (25 October) | `results/*.csv`, `figures/progress_*.png` |
| `src/scout_analysis.py`, `src/exp_*boosting*.py`, `src/exp_log_target.py`, `src/exp_weighted_ridge.py`, `src/exp_engineered_features.py`, `src/compare_xg_xa*.py`, `src/feature_importance_xg.py` | Earlier experiments on a chronological split (train 2019–2021, validate 2022, test 2023); kept as a **robustness check**, not as main results | `results/scout/`, `figures/scout/` |
| `tests/test_scout.py` | Checks data loading, missing values and that no player is in both development and reserve | run `python -m unittest discover -s tests -v` |

**Encoding options** (`encoding=` in `build_pipeline`): `onehot` gives every
nationality and club its own 0/1 column; `grouped` merges categories with fewer than
10 rows into one "infrequent" column; `drop` removes nationality and club.

**Split options** (`kind=` in `cv_splits`): `group` (main design, players kept
together), `random` (players can appear on both sides), `chronological` (train on
earlier seasons, test on the next one).

## Current results (development players, grouped 5-fold CV)
From `results/final/T1_model_comparison.csv`. MAE and RMSE in millions (currency to be
confirmed), mean ± SD over five folds. **The 132 reserved players have not been evaluated.**

| Model | Start value | MAE | RMSE | R² |
|---|:---:|---:|---:|---:|
| Median baseline | no | 16.62 ± 3.57 | 24.02 | −0.11 |
| Linear regression | no | 10.83 ± 0.90 | 14.67 | 0.55 |
| Ridge | no | 10.67 ± 0.93 | 14.50 | 0.56 |
| Random forest | no | 10.52 ± 0.94 | 15.15 | 0.52 |
| **Gradient boosting** | no | **9.66 ± 1.22** | **13.69** | **0.61** |
| Start value carried forward | yes | 5.49 ± 1.01 | 8.89 | 0.84 |
| Ridge | yes | 5.00 ± 0.56 | 7.35 | 0.89 |
| Random forest | yes | 4.95 ± 0.94 | 8.18 | 0.87 |
| **Gradient boosting** | yes | **4.78 ± 0.86** | 7.84 | 0.88 |

- Season data alone explain about 60% of the variance; the start-of-season value is the
  strongest single predictor.
- Club and age matter most (`T6_feature_importance.csv`); nationality adds almost nothing.
- xG/xA, PCA and a log target do not change these conclusions (`T2`, `T4`, `T5`).
- Random-forest OOB error (9.31) is more optimistic than grouped CV (10.52), because most
  players appear in several seasons (`T3`).

The final evaluation on the reserved players will be run once, after the team agrees on
the final model.

## Structure
```
data/            Nisanov dataset (never edited by hand)
data/external/   Understat xG/xA export (unmodified)
data/processed/  merged data written by src/integrate_understat.py
src/             shared code (data.py, pipeline.py: change only by agreement) + experiments
tests/           automated checks
results/final/   main results (src/final_analysis.py)
results/         progress-report results; results/scout/: chronological robustness check
figures/final/   figures for the outline and report; figures/scout/: robustness figures
report/          LaTeX sources (sentence outline, bibliography)
```

## AI use
Parts of the code were drafted with generative AI (Claude, Anthropic) and reviewed by
the team. Each submission's declaration lists which files, what was changed and how
it was verified.
