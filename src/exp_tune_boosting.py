from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA = Path("data/processed/scout_outfield.csv")
OUTPUT = Path("results/scout/boosting_tuning.csv")
TARGET = "value_end_mln"

FEATURES = [
    "age_then", "position", "nationality", "height",
    "total_games", "total_minutes", "total_goals",
    "assists", "penalty_kicks", "xG", "xA"
]

df = pd.read_csv(DATA)
df = df.dropna(subset=[TARGET, "season"]).copy()
df = df[df["season"].between(2019, 2022)]

numeric = [x for x in FEATURES if x not in ("position", "nationality")]
categorical = ["position", "nationality"]

def make_model(iterations, leaves, min_leaf, learning_rate):
    preprocessing = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )),
        ]), categorical),
    ], sparse_threshold=0)

    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", HistGradientBoostingRegressor(
            max_iter=iterations,
            max_leaf_nodes=leaves,
            min_samples_leaf=min_leaf,
            learning_rate=learning_rate,
            random_state=42
        )),
    ])

# 36 configurations
grid = list(product(
    [100, 200, 350],
    [7, 15, 31],
    [10, 25],
    [0.03, 0.08]
))

rows = []

for i, (iterations, leaves, min_leaf, lr) in enumerate(grid, 1):
    fold_mae = []
    fold_r2 = []

    for year in [2020, 2021, 2022]:
        train = df[df["season"] < year]
        validation = df[df["season"] == year]

        model = make_model(iterations, leaves, min_leaf, lr)
        model.fit(train[FEATURES], train[TARGET])
        predicted = model.predict(validation[FEATURES])

        fold_mae.append(
            mean_absolute_error(validation[TARGET], predicted)
        )
        fold_r2.append(r2_score(validation[TARGET], predicted))

    rows.append({
        "iterations": iterations,
        "max_leaf_nodes": leaves,
        "min_samples_leaf": min_leaf,
        "learning_rate": lr,
        "mean_MAE": np.mean(fold_mae),
        "mean_R2": np.mean(fold_r2),
        "MAE_2020": fold_mae[0],
        "MAE_2021": fold_mae[1],
        "MAE_2022": fold_mae[2],
    })

    if i % 6 == 0:
        print(f"Completed {i}/{len(grid)} configurations", flush=True)

results = pd.DataFrame(rows).sort_values("mean_MAE")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(OUTPUT, index=False)

print("\nTOP 10 CONFIGURATIONS")
print(results.head(10).round(3).to_string(index=False))
print("\nBEST CONFIGURATION")
print(results.iloc[0].to_string())
print("\nSaved:", OUTPUT)
