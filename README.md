# SCOUT: Predicting Football Players' Estimated Market Values

Machine Learning & Communication Skills, IE University (2026–2027).
Team: Pablo Garma, Noa Keseric, Kristian Spassov.

**Research question:** To what extent can player characteristics and performance
statistics predict football players' estimated market values, and which features
contribute most to predictive performance?

## Data
Nisanov, R. *Transfer Value Determinants*, Version 2, Mendeley Data,
doi:10.17632/3btg6ptc7b.2, CC BY 4.0. Stored in `data/` unmodified.
Only the worksheet named `2019` is used: it already combines all 2019–2023 seasons.

## Setup (once)
```bash
conda env create -f environment.yml
conda activate scout
python src/data.py            # smoke test, must print "OK"
```

## Evaluation design
- **Final test set:** 20% of *players* (all their seasons) held out with a fixed
  seed in `src/data.py`. Not used until the final model is chosen.
- **Model selection:** 5-fold cross-validation on the remaining players, grouped by
  player so one player's seasons never sit in both training and validation folds.
- **Preprocessing** (median imputation, scaling, one-hot encoding) is inside the
  pipeline and re-fitted in every fold.
- Random seed: 42.

## Structure
```
data/      raw data (never edited by hand)
src/       data.py, pipeline.py (shared, change only by agreement) + experiment scripts
results/   CSV outputs written by scripts
figures/   figures written by scripts
report/    LaTeX source of the reports
```

## Reproduce
| Script | Output | Owner |
|---|---|---|
| `src/example_baselines.py` | `results/example_baselines.csv` | template |
| `src/exp_models.py` | `results/models.csv` | Person B |
| `src/exp_encoding.py` | `results/encoding.csv`, `figures/encoding.png` | Person C |
| `src/exp_splits.py` | `results/splits.csv`, `results/by_season.csv` | Person A |

## Team workflow
1. `git pull` before you start working.
2. Work on your own branch: `git checkout -b <your-name>/<task>`.
3. Commit small, descriptive steps (e.g. "Add ridge alpha grid search").
4. Push and open a pull request; another member reviews and merges it.
5. Only edit files you own. Changes to `data.py` / `pipeline.py` are agreed first.
