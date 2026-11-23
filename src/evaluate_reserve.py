"""Final reserved-player evaluation for SCOUT.

Run only after model selection is complete.
No hyperparameters are selected using reserve results.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import GROUP, NUMERIC, START, TARGET, load_data, split_holdout
from final_analysis import (
    CAT, XG, make, metrics, pipe, tune, load_with_xg
)

OUTPUT = Path("results/final/reserve_test_results.csv")
data = load_with_xg()
dev, reserve = split_holdout(data)

assert set(dev[GROUP]).isdisjoint(set(reserve[GROUP]))
print(f"Development: {len(dev)} rows, {dev[GROUP].nunique()} players")
print(f"Reserve: {len(reserve)} rows, {reserve[GROUP].nunique()} players")

rows = []
features = NUMERIC + XG

def evaluate(name, predictions, start_value, params=None):
    scores = metrics(reserve[TARGET], predictions)
    rows.append({
        "model": name,
        "starting_value": start_value,
        "MAE": scores["MAE"],
        "RMSE": scores["RMSE"],
        "R2": scores["R2"],
        "hyperparameters": str(params or {}),
    })
    print(f"{name:35s} start={start_value} "
          f"MAE={scores['MAE']:.3f} "
          f"RMSE={scores['RMSE']:.3f} "
          f"R2={scores['R2']:.3f}", flush=True)

# Median baseline: training data only.
evaluate(
    "Median baseline",
    np.full(len(reserve), dev[TARGET].median()),
    False,
)

# Starting-value baselines: growth estimated only from development.
valid = dev[START].notna() & (dev[START] > 0)
growth = (
    (dev.loc[valid, TARGET] - dev.loc[valid, START])
    / dev.loc[valid, START]
)
fallback = dev[TARGET].median()

evaluate(
    "Carry-forward starting value",
    reserve[START].fillna(fallback),
    True,
)

for label, rate in [
    ("Mean growth baseline", growth.mean()),
    ("Median growth baseline", growth.median()),
]:
    predictions = (
        reserve[START] * (1 + rate)
    ).fillna(fallback)
    evaluate(label, predictions, True)

# Fit each model on all development players, then test on reserve.
for include_start in [False, True]:
    numeric = features + ([START] if include_start else [])
    for name in [
        "Linear regression",
        "Ridge",
        "Random forest",
        "Gradient boosting",
    ]:
        print(f"Tuning {name}, starting value={include_start}...", flush=True)

        params = tune(name, dev, numeric)
        model = pipe(make(name, **params), numeric)
        model.fit(dev[numeric + CAT], dev[TARGET])
        predictions = model.predict(reserve[numeric + CAT])

        evaluate(name, predictions, include_start, params)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).round(4).to_csv(OUTPUT, index=False)

print(f"\nSaved results to {OUTPUT}")
