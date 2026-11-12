"""Worked example showing the pattern every experiment script follows.

1. load data, 2. take the dev set, 3. get folds, 4. build pipeline(s),
5. evaluate, 6. save a CSV in results/.

Run from the repository root:  python src/example_baselines.py
"""
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from data import ROOT, START, TARGET, cv_splits, load_data, split_holdout
from pipeline import build_pipeline, evaluate, results_table

dev, _ = split_holdout(load_data())          # never use the test part here
splits = cv_splits(dev, kind="group")

rows = []
# Baseline 1: predict the training median for everyone
pipe = build_pipeline(DummyRegressor(strategy="median"))
rows.append({"model": "Median baseline", **evaluate(pipe, dev, splits)})

# Baseline 2: end value = start value (no fitting needed)
scores = {"MAE": [], "RMSE": [], "R2": []}
for _, val in splits:
    y, p = dev[TARGET].iloc[val], dev[START].iloc[val]
    scores["MAE"].append(mean_absolute_error(y, p))
    scores["RMSE"].append(root_mean_squared_error(y, p))
    scores["R2"].append(r2_score(y, p))
rows.append({"model": "Start value carried forward",
             **{f"{k}_{s}": getattr(np, s)(v) for k, v in scores.items()
                for s in ("mean", "std")}})

# First real model: linear regression, without and with the start value
for use_start in (False, True):
    pipe = build_pipeline(LinearRegression(), use_start=use_start)
    rows.append({"model": f"Linear regression (start value: {use_start})",
                 **evaluate(pipe, dev, splits, use_start=use_start)})

table = results_table(rows)
print(table.to_string(index=False))
(ROOT / "results").mkdir(exist_ok=True)
table.to_csv(ROOT / "results" / "example_baselines.csv", index=False)
