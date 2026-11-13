"""Compare ordinary Ridge against market-value-weighted Ridge."""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path("data/processed/scout_outfield.csv")
OUTPUT = Path("results/scout/weighted_ridge_comparison.csv")
TARGET = "value_end_mln"

df = pd.read_csv(DATA)

# Keep the same performance-only predictors as our previous experiments.
features = [
    "age_then", "position", "nationality", "height",
    "total_games", "total_minutes", "total_goals",
    "assists", "penalty_kicks", "xG", "xA",
]

# Avoid rows without the target or required features.
df = df.dropna(subset=[TARGET, "season"]).copy()
features = [c for c in features if c in df.columns]

train = df[df["season"].between(2019, 2021)].copy()
validation = df[df["season"] == 2022].copy()

numeric = [
    c for c in features
    if pd.api.types.is_numeric_dtype(df[c])
]
categorical = [c for c in features if c not in numeric]


def build_model():
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(
                strategy="most_frequent"
            )),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore"
            )),
        ]), categorical),
    ])

    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", Ridge(alpha=10)),
    ])


# Calculate weights using training values only.
# Higher-value players receive more importance.
y_train = train[TARGET].clip(lower=0)
scale = max(float(y_train.median()), 1.0)

experiments = {
    "Ordinary Ridge": np.ones(len(train)),
    "Mild weighting": np.sqrt(1 + y_train / scale),
    "Moderate weighting": 1 + y_train / scale,
}

results = []

# Fixed validation brackets based on training target quartiles.
edges = np.quantile(y_train, [0.25, 0.5, 0.75])
edges = np.unique(edges)

for name, weights in experiments.items():
    model = build_model()

    model.fit(
        train[features],
        train[TARGET],
        model__sample_weight=np.asarray(weights),
    )

    actual = validation[TARGET].to_numpy()
    predicted = model.predict(validation[features])

    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    row = {
        "Model": name,
        "Validation_MAE": mae,
        "Validation_RMSE": rmse,
        "Validation_R2": r2,
    }

    brackets = np.searchsorted(edges, actual, side="right")

    for i in range(len(edges) + 1):
        mask = brackets == i
        row[f"Q{i+1}_MAE"] = (
            mean_absolute_error(actual[mask], predicted[mask])
            if mask.any() else np.nan
        )

    results.append(row)

results = pd.DataFrame(results)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(OUTPUT, index=False)

print("\nWEIGHTED RIDGE COMPARISON")
print("Training:", len(train))
print("Validation:", len(validation))
print("\n", results.round(3).to_string(index=False))
print("\nSaved:", OUTPUT)
