"""Compare Ridge and Gradient Boosting using chronological validation."""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path("data/processed/scout_outfield.csv")
OUTPUT = Path("results/scout/gradient_boosting_comparison.csv")
TARGET = "value_end_mln"

df = pd.read_csv(DATA)
df = df.dropna(subset=[TARGET, "season"]).copy()

features = [
    "age_then", "position", "nationality", "height",
    "total_games", "total_minutes", "total_goals",
    "assists", "penalty_kicks", "xG", "xA",
]
features = [c for c in features if c in df.columns]

train = df[df["season"].between(2019, 2021)]
validation = df[df["season"] == 2022]

numeric = [
    c for c in features
    if pd.api.types.is_numeric_dtype(df[c])
]
categorical = [c for c in features if c not in numeric]

def make_pipeline(estimator):
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )),
        ]), categorical),
    ])

    return Pipeline([
        ("preprocessing", preprocess),
        ("model", estimator),
    ])

models = {
    "Ridge baseline": Ridge(alpha=10),
    "Boosting shallow": HistGradientBoostingRegressor(
        max_iter=200, max_leaf_nodes=7,
        min_samples_leaf=15, learning_rate=0.05,
        random_state=42,
    ),
    "Boosting medium": HistGradientBoostingRegressor(
        max_iter=250, max_leaf_nodes=15,
        min_samples_leaf=15, learning_rate=0.05,
        random_state=42,
    ),
    "Boosting regularized": HistGradientBoostingRegressor(
        max_iter=200, max_leaf_nodes=7,
        min_samples_leaf=25, l2_regularization=10,
        learning_rate=0.05, random_state=42,
    ),
}

results = []

for name, estimator in models.items():
    model = make_pipeline(estimator)
    model.fit(train[features], train[TARGET])

    actual = validation[TARGET].to_numpy()
    predicted = model.predict(validation[features])

    results.append({
        "Model": name,
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(mean_squared_error(actual, predicted)),
        "R2": r2_score(actual, predicted),
    })

results = pd.DataFrame(results).sort_values("MAE")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
results.to_csv(OUTPUT, index=False)

print("\nGRADIENT BOOSTING COMPARISON")
print("Training:", len(train))
print("Validation:", len(validation))
print("\n", results.round(3).to_string(index=False))
print("\nSaved:", OUTPUT)
