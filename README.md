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
| `load_data()` | Reads the combined sheet and checks it has 1,382 rows and seasons 2019–2023 | Stops immediately if the wrong data is loaded |
| `split_holdout()` | Locks away 20% of *players* (all their seasons) as the final test set, with seed 42 | Same 132 test players for everyone; **never used before the final report** |
| `cv_splits(dev, kind)` | Cuts the remaining 1,103 rows into folds | `"group"`: main design, random across seasons, players kept together. `"random"` and `"chronological"`: only for the split experiment |
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

## Models (progress report)
| Model | Role |
|---|---|
| Median baseline | Predicts the same typical value for everyone; any model must beat it |
| Start value carried forward | Predicts "value stays what it was at the start of the season" |
| Linear regression | First real model |
| Ridge regression | Linear regression with a penalty that keeps coefficients small; tests whether this helps with many correlated or sparse columns |

Every model is run **with and without** the start-of-season value (`value_0_mln`),
to separate what performance data explain on their own from what the market already
knew. Random forest is planned after it is covered in the course.

## Experiments for the progress report
| Script | Question it answers | Output | Owner |
|---|---|---|---|
| `src/example_baselines.py` | Template showing the pattern every script follows | `results/example_baselines.csv` | shared |
| `src/exp_models.py` | Which model predicts best, and what does the start value add? | `results/models.csv` | Person B |
| `src/exp_encoding.py` | Do the many sparse columns from nationality/club hurt the models? | `results/encoding.csv`, `figures/encoding.png` | Person C |
| `src/exp_splits.py` | Does the way we split the data change the results (leakage, year effects)? | `results/splits.csv`, `results/by_season.csv` | Person A |

**Encoding options** (`encoding=` in `build_pipeline`): `onehot` gives every
nationality and club its own 0/1 column; `grouped` merges categories with fewer than
10 rows into one "infrequent" column; `drop` removes nationality and club.

**Split options** (`kind=` in `cv_splits`): `group` (main design, players kept
together), `random` (players can appear on both sides), `chronological` (train on
earlier seasons, test on the next one).

## Structure
```
data/      raw data (never edited by hand)
src/       data.py, pipeline.py (shared, change only by agreement) + experiment scripts
results/   CSV outputs written by scripts
figures/   figures written by scripts
report/    LaTeX source of the reports
```

## AI use
Parts of the code were drafted with generative AI (Claude, Anthropic) and reviewed by
the team. Each submission's declaration lists which files, what was changed and how
it was verified.