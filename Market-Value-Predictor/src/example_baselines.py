"""Reproducible baseline results from DEVELOPMENT folds only.

Run from project root: python src/example_baselines.py
"""
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from data import ROOT, START, TARGET, cv_splits, load_data, split_holdout
from pipeline import build_pipeline, evaluate, results_table


def main():
    dev, _holdout = split_holdout(load_data())
    splits = cv_splits(dev, kind="group")
    rows = [{"model": "Median baseline", **evaluate(
        build_pipeline(DummyRegressor(strategy="median")), dev, splits)}]

    # End value = start value. The 10 unknown starting values have been restored
    # to NaN; impute them with TRAINING-FOLD start-value median for fair scoring.
    scores = {"MAE": [], "RMSE": [], "R2": []}
    for train_idx, val_idx in splits:
        fallback = dev[START].iloc[train_idx].median()
        y = dev[TARGET].iloc[val_idx]
        pred = dev[START].iloc[val_idx].fillna(fallback)
        scores["MAE"].append(mean_absolute_error(y, pred))
        scores["RMSE"].append(root_mean_squared_error(y, pred))
        scores["R2"].append(r2_score(y, pred))
    rows.append({"model": "Start value carried forward (train-median fallback)",
                 **{f"{metric}_{s}": getattr(np, s)(values)
                    for metric, values in scores.items() for s in ("mean", "std")}})

    for use_start in (False, True):
        rows.append({"model": f"Linear regression (start value: {use_start})",
                     **evaluate(build_pipeline(LinearRegression(), use_start=use_start),
                                dev, splits, use_start=use_start)})

    table = results_table(rows)
    (ROOT / 'results').mkdir(exist_ok=True)
    table.to_csv(ROOT / "results" / "example_baselines.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
