"""Compare original and log-transformed market-value targets."""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path("data/processed/scout_outfield.csv")
OUTPUT = Path("results/scout/log_target_comparison.csv")
TARGET = "value_end_mln"

# Performance-only features: no starting market value.
FEATURES = [
    "age_then", "height", "position", "nationality",
    "total_games", "total_minutes", "total_goals",
    "assists", "penalty_kicks", "xG", "xA"
]

df = pd.read_csv(DATA)
df = df.dropna(subset=[TARGET, "season"]).copy()

train = df[df["season"].between(2019, 2021)]
validation = df[df["season"] == 2022]

numeric = [c for c in FEATURES if c not in ["position", "nationality"]]
categorical = ["position", "nationality"]

def make_model(kind, log_target):
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])

    if kind == "Ridge":
        estimator = Ridge(alpha=10)
    else:
        estimator = RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )

    pipeline = Pipeline([
        ("preprocessing", preprocessing),
        ("model", estimator)
    ])

    if log_target:
        return TransformedTargetRegressor(
            regressor=pipeline,
            func=np.log1p,
            inverse_func=np.expm1,
            check_inverse=True
        )

    return pipeline

results = []

for kind in ["Ridge", "Random Forest"]:
    for log_target in [False, True]:
        model = make_model(kind, log_target)
        model.fit(train[FEATURES], train[TARGET])

        actual = validation[TARGET].to_numpy()
        predicted = np.maximum(
            model.predict(validation[FEATURES]), 0
        )

        # Report metrics in original market-value units.
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)

        label = "Log target" if log_target else "Original target"

        results.append({
            "Model": kind,
            "Target": label,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        print(f"\n{kind} | {label}")
        print(f"MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.3f}")

        # Four equally populated brackets based on actual values.
        brackets = pd.qcut(
            pd.Series(actual).rank(method="first"),
            q=4,
            labels=["Q1", "Q2", "Q3", "Q4"]
        )

        errors = pd.DataFrame({
            "Bracket": brackets,
            "Actual": actual,
            "Prediction": predicted,
            "Absolute_Error": np.abs(actual - predicted)
        })

        print(errors.groupby(
            "Bracket", observed=True
        )["Absolute_Error"].mean().round(3))

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(results).to_csv(OUTPUT, index=False)
print(f"\nSaved: {OUTPUT}")
