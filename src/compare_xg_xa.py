"""SCOUT: Compare market-value predictions with and without xG/xA."""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path("data/processed/scout_outfield.csv")
OUTPUT = Path("results/scout/xg_xa_comparison.csv")
TARGET = "value_end_mln"

df = pd.read_csv(DATA)

# Keep only the seasons in our original SCOUT project.
df = df[df["season"].isin([2019, 2020, 2021, 2022, 2023])].copy()

# Avoid leakage: don't use end-of-season valuation or its derivatives.
features = [
    "age_then", "position", "nationality",
    "total_games", "assists", "penalty_kicks",
    "total_minutes", "total_goals", "height",
    "value_0_mln",
]
features = [c for c in features if c in df.columns]

# Restrict both experiments to identical observations.
# This isolates the impact of xG/xA, but may introduce selection bias.
df = df.dropna(subset=[TARGET, "xG", "xA"]).copy()

train = df[df["season"].isin([2019, 2020, 2021])]
validation = df[df["season"] == 2022]
test = df[df["season"] == 2023]

print("Matched observations:", len(df))
print("Train:", len(train), "Validation:", len(validation), "Test:", len(test))

if min(len(train), len(validation), len(test)) == 0:
    raise ValueError("One chronological split is empty.")

def make_pipeline(columns, model):
    categorical = [
        c for c in columns
        if not pd.api.types.is_numeric_dtype(df[c])
    ]
    numeric = [c for c in columns if c not in categorical]

    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])

    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", model),
    ])

def score(y_true, predicted):
    return {
        "MAE": mean_absolute_error(y_true, predicted),
        "RMSE": np.sqrt(mean_squared_error(y_true, predicted)),
        "R2": r2_score(y_true, predicted),
    }

results = []

for name, columns in [
    ("Baseline", features),
    ("With xG/xA", features + ["xG", "xA"]),
]:
    for model_name, factory in [
        ("Ridge", lambda: Ridge(alpha=10)),
        ("Random Forest", lambda: RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )),
    ]:
        # Validation: train only on 2019-2021.
        pipeline = make_pipeline(columns, factory())
        pipeline.fit(train[columns], train[TARGET])
        val_metrics = score(
            validation[TARGET],
            pipeline.predict(validation[columns])
        )

        # Final test: retrain on 2019-2022.
        development = pd.concat([train, validation])
        pipeline = make_pipeline(columns, factory())
        pipeline.fit(development[columns], development[TARGET])
        test_metrics = score(
            test[TARGET],
            pipeline.predict(test[columns])
        )

        results.append({
            "Features": name,
            "Model": model_name,
            "Validation_MAE": val_metrics["MAE"],
            "Validation_RMSE": val_metrics["RMSE"],
            "Validation_R2": val_metrics["R2"],
            "Test_MAE": test_metrics["MAE"],
            "Test_RMSE": test_metrics["RMSE"],
            "Test_R2": test_metrics["R2"],
        })

results = pd.DataFrame(results)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(OUTPUT, index=False)

print("\nSCOUT MODEL COMPARISON")
print(results.round(3).to_string(index=False))
print("\nSaved:", OUTPUT)
