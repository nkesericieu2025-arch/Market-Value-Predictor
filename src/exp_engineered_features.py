from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path("data/processed/scout_outfield.csv")
OUTPUT = Path("results/scout/engineered_features_comparison.csv")
TARGET = "value_end_mln"

df = pd.read_csv(DATA)

# Prevent target leakage: never use existing market value
# or fields calculated from the target as predictors.
baseline_features = [
    "age_then", "position", "nationality", "height",
    "total_games", "total_minutes", "total_goals",
    "assists", "penalty_kicks", "xG", "xA",
]

# Convert playing time into per-90-minute performance.
minutes = pd.to_numeric(df["total_minutes"], errors="coerce")
minutes = minutes.where(minutes > 0)

for original, new in [
    ("total_goals", "goals_per90"),
    ("assists", "assists_per90"),
    ("xG", "xg_per90"),
    ("xA", "xa_per90"),
]:
    df[new] = (
        pd.to_numeric(df[original], errors="coerce")
        / minutes * 90
    )

df["age_goals_interaction"] = (
    pd.to_numeric(df["age_then"], errors="coerce")
    * df["goals_per90"]
)

engineered = [
    "goals_per90", "assists_per90",
    "xg_per90", "xa_per90",
    "age_goals_interaction",
]

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=[TARGET, "season"])

train = df[df["season"].isin([2019, 2020, 2021])]
validation = df[df["season"] == 2022]

def evaluate(features, name):
    numeric = [
        c for c in features
        if c not in ["position", "nationality"]
    ]
    categorical = [
        c for c in features
        if c in ["position", "nationality"]
    ]

    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore",
                                      sparse_output=False)),
        ]), categorical),
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("regressor", HistGradientBoostingRegressor(
            max_iter=200,
            max_leaf_nodes=7,
            min_samples_leaf=15,
            learning_rate=0.05,
            random_state=42,
        )),
    ])

    model.fit(train[features], train[TARGET])
    predictions = model.predict(validation[features])

    return {
        "Experiment": name,
        "MAE": mean_absolute_error(validation[TARGET], predictions),
        "RMSE": np.sqrt(mean_squared_error(
            validation[TARGET], predictions
        )),
        "R2": r2_score(validation[TARGET], predictions),
    }

results = pd.DataFrame([
    evaluate(baseline_features, "Original features"),
    evaluate(baseline_features + engineered, "Engineered features"),
])

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(OUTPUT, index=False)

print("\nFEATURE ENGINEERING EXPERIMENT")
print(f"Training: {len(train)} | Validation: {len(validation)}")
print(results.round(3).to_string(index=False))
print("\nSaved:", OUTPUT)
