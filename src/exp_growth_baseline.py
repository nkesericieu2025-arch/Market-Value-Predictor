"""Percentage-growth baseline (team note: "compute percentage between target and
initial, then the mean for all the players").

For every training fold:
    growth_i = (end_value_i - start_value_i) / start_value_i   for players with a
                                                               known, positive start value
    g        = mean(growth_i)            (also reported: median, more robust)
    predict  = start_value * (1 + g)

Players with no start value (mostly academy players and new arrivals) cannot use
this rule; they fall back to the training-fold median end value, and are also
reported separately because no start-value baseline can price them.

The growth rate is learned on training folds only, so validation players never
influence their own prediction.

Run from the repository root:  python src/exp_growth_baseline.py
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from data import ROOT, START, TARGET, cv_splits, load_data, split_holdout


def growth_predictions(train, val, stat="mean"):
    known = train[START].notna() & (train[START] > 0)
    growth = (train.loc[known, TARGET] - train.loc[known, START]) / train.loc[known, START]
    g = growth.mean() if stat == "mean" else growth.median()
    fallback = train[TARGET].median()
    pred = val[START] * (1 + g)
    no_start = val[START].isna() | (val[START] <= 0)
    return pred.where(~no_start, fallback), g, no_start


def scores(y, p):
    return {"MAE": mean_absolute_error(y, p),
            "RMSE": root_mean_squared_error(y, p),
            "R2": r2_score(y, p)}


def main():
    dev, _holdout = split_holdout(load_data())          # holdout never touched
    rows = []
    for stat in ("mean", "median"):
        fold_scores, rates, n_fallback = [], [], 0
        for train_idx, val_idx in cv_splits(dev, kind="group"):
            train, val = dev.iloc[train_idx], dev.iloc[val_idx]
            pred, g, no_start = growth_predictions(train, val, stat)
            fold_scores.append(scores(val[TARGET], pred))
            rates.append(g)
            n_fallback += int(no_start.sum())
        df = pd.DataFrame(fold_scores)
        rows.append({"model": f"Start value x (1 + {stat} growth)",
                     "design": "grouped 5-fold CV",
                     "growth_rate_mean_over_folds": np.mean(rates),
                     "rows_without_start_value": n_fallback,
                     **{f"{m}_mean": df[m].mean() for m in df},
                     **{f"{m}_std": df[m].std(ddof=0) for m in df}})

    # Same baseline on the chronological 2019-2021 -> 2022 split, so it can be
    # compared with the season-based experiments in results/scout/.
    train, val = dev[dev["season"] <= 2021], dev[dev["season"] == 2022]
    for stat in ("mean", "median"):
        pred, g, no_start = growth_predictions(train, val, stat)
        rows.append({"model": f"Start value x (1 + {stat} growth)",
                     "design": "train 2019-2021, validate 2022",
                     "growth_rate_mean_over_folds": g,
                     "rows_without_start_value": int(no_start.sum()),
                     **{f"{m}_mean": v for m, v in scores(val[TARGET], pred).items()}})

    table = pd.DataFrame(rows).round(3)
    print(table.to_string(index=False))
    table.to_csv(ROOT / "results" / "growth_baseline.csv", index=False)


if __name__ == "__main__":
    main()
