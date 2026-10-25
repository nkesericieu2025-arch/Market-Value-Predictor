# SCOUT — Player Market Value Prediction

Team: Pablo Garma, Noa Keseric, Kristian Spassov. IE University, 2026–2027.
Original repository: https://github.com/nkesericieu2025-arch/Market-Value-Predictor

## Research objective and timing

We **estimate end-of-season player market value** (`value_end_mln`) from player
attributes, same-season performance (minutes, goals, assists, etc.), and club
information known by season end. We compare models and examine **predictive**
feature importance; this is not a pre-season market-value forecast, and
predictive importance is not causation. We evaluate separately with and without
start-of-season valuation, to ask whether performance features improve on the
existing market estimate.

## What changed on 25 October 2026

The **primary evaluation** is random 5-fold **player-grouped** CV, not a
season-based split. The players are shuffled independently of seasons, while
all seasons of an individual player stay in one fold. This avoids the repeated
player appearing in both training and validation folds. The 20%-of-players
holdout is assigned once using `GroupShuffleSplit(random_state=42)` and not
used for today's experiment selection. Ordinary random-row CV and expanding
chronological CV are retained **only as diagnostics**, in `exp_splits.py`.

### Historical chronological pipeline

The former `scout_analysis.py` was moved **unchanged** to
`src/archive/scout_analysis_legacy_chronological.py`. The original
`results/scout/` and `figures/scout/` are historical exploratory outputs,
including 2023 test scores. **Do not label them as unseen final-test results.**
Because 53 of the earlier 2023 evaluated rows overlap the current reserved
player-level holdout, that holdout is not entirely untouched in a strict
research sense. This limitation must be disclosed, rather than relabeling an
earlier experiment as if nobody had seen it.

## Data and quality controls

Dataset: Ronald Nisanov, *Transfer Value Determinants*, Mendeley Data v2,
doi:10.17632/3btg6ptc7b.2, CC BY 4.0. `data/Nisanov_final_data.xlsx`
is unchanged. The `2019` sheet combines all five seasons: **1,382 raw rows**.

The original workbook encodes missing raw values as zeros in derived columns:
10 missing raw starting values were silently converted to 0 in `value_0_mln`;
one missing raw ending value was represented as 0 in `value_end_mln`.
`src/data.py` restores the 10 start missing values (`NaN`), which are imputed
**within each training fold**. The holdout players are assigned using all 1,382
source records **before** removing one row with an unknown target, leaving
**1,381 modeling rows** without changing which players were reserved.
Genuine zeros with actual nonmissing source
values are retained. This cleaning does not edit the Excel file.

The estimated-value currency is **not established here**. Model MAE/RMSE are
reported in target-column millions without claiming GBP/EUR. Historical
figures may contain an unverified GBP label and should not be reused as-is.
The dataset appears to be scoring players from a restricted league/sample;
inspect the source documentation before generalizing.

### Leakage prevention

- Target: `value_end_mln`.
- Exclude `end_value` and changes (`delta_value`, `value_delta_mln`): direct or target-derived information.
- Exclude `age_now`: age measured after the observed season.
- Exclude `player_name`: player grouping key, never a feature.
- Exclude `season` from primary prediction features: it is used only for diagnostics.
- Imputation, numeric scaling and category encoding are inside the fitted sklearn pipeline, retrained from each **training fold only**.
- If predicting **before** the season, same-season goals/minutes and end-year club outcomes would additionally be forbidden; that is not this project's current setup.

## Run and reproduce

```bash
conda env create -f environment.yml
conda activate scout
python -m unittest discover -s tests -v
python src/data.py
python src/example_baselines.py   # results/example_baselines.csv
python src/model_comparison.py    # results/model_comparison.csv
python src/exp_encoding.py        # results/encoding.csv
python src/exp_splits.py          # results/splits.csv
```

`python src/scout_analysis.py` runs the first two scripts with the consistent
primary method. The experiment scripts load only development data; the
historical 2023-scored script is archived, not part of the current run sequence.
Run from project root; code assumes scikit-learn **1.6+** (shuffled GroupKFold).

## Experiments aligned with professor feedback

| Experiment | Comparison | Primary conclusion it enables |
|---|---|---|
| `model_comparison.py` | Median baseline, linear regression, Ridge and Random Forest, without/with start value | What existing valuation contributes and whether ML helps |
| `example_baselines.py` | Also includes the start-value carried-forward baseline | Can a model beat market persistence? |
| `exp_encoding.py` | Full one-hot, rare-level grouping, remove nationality/club; Ridge and Random Forest | Is one-hot sparsity harmful? |
| `exp_splits.py` | Shuffled grouped CV, shuffled row CV, expanding-window chronological CV | Are results sensitive to player overlap and market-year drift? |

**Interpretation cautions:** Five-fold standard deviation is not a 95% CI.
Splitting methods have different validation structures and do not measure
identical generalization tasks. The current model comparisons use CV to
choose candidates and must not be described as independently tested final
performance. No conclusions should be declared statistically significant
from small differences without paired uncertainty analysis.

## Outstanding steps for today's progress report

1. Report reproducible baseline and comparative metrics from these scripts.
2. Include encoding and splitting diagnostics as direct answers to instructor feedback.
3. Discuss the two original-value missingness inconsistencies and caveats.
4. Explain the prediction-time interpretation and pre-existing 2023 evaluation.
5. Make one concise model table and one meaningful figure, then draft a two-page
   report and AI-use disclosure documenting which code was AI-assisted, what the
   team understood/changed, and how the results were verified.

## Collaboration

Real incremental commits, separate task branches, and teammate-reviewed pull
requests. Do not invent old commits or claim all teammates independently wrote
AI-assisted code. Before merging, have another member run the tests and scripts
and review every result used in the report.
