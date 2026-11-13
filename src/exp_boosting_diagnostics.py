"""Compare Ridge and shallow boosting errors and feature importance."""

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path("data/processed/scout_outfield.csv")
OUTPUT = Path("results/scout")
OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA).dropna(
    subset=["value_end_mln", "season"]
)

features = [
    "age_then", "position", "nationality", "height",
    "total_games", "total_minutes", "total_goals",
    "assists", "penalty_kicks", "xG", "xA",
]
features = [c for c in features if c in df.columns]

train = df[df["season"].between(2019, 2021)]
val = df[df["season"] == 2022]

numeric = [
    c for c in features
    if pd.api.types.is_numeric_dtype(df[c])
]
categorical = [c for c in features if c not in numeric]

def build(estimator):
    preprocessing = ColumnTransformer([
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
        ("preprocessing", preprocessing),
        ("model", estimator),
    ])

models = {
    "Ridge": Ridge(alpha=10),
    "Shallow Boosting": HistGradientBoostingRegressor(
        max_iter=200,
        max_leaf_nodes=7,
        min_samples_leaf=15,
        learning_rate=0.05,
        random_state=42,
    ),
}

actual = val["value_end_mln"].to_numpy()
edges = np.unique(
    np.quantile(train["value_end_mln"], [0.25, 0.5, 0.75])
)
brackets = np.searchsorted(edges, actual, side="right") + 1

rows = []
importance_rows = []
prediction_rows = pd.DataFrame({
    "player_name": val["player_name"].to_numpy(),
    "season": val["season"].to_numpy(),
    "actual": actual,
    "bracket": brackets,
})

for name, estimator in models.items():
    model = build(estimator)
    model.fit(train[features], train["value_end_mln"])
    predicted = model.predict(val[features])

    print(f"\n{name}")
    print("MAE:", round(mean_absolute_error(actual, predicted), 3))
    print("RMSE:", round(np.sqrt(mean_squared_error(actual, predicted)), 3))
    print("R2:", round(r2_score(actual, predicted), 3))

    prediction_rows[name] = predicted
    prediction_rows[f"{name}_absolute_error"] = np.abs(
        actual - predicted
    )

    for q in np.unique(brackets):
        mask = brackets == q
        rows.append({
            "Model": name,
            "Bracket": f"Q{q}",
            "Players": int(mask.sum()),
            "Mean_Actual_Value": actual[mask].mean(),
            "MAE": mean_absolute_error(
                actual[mask], predicted[mask]
            ),
            "Bias": (predicted[mask] - actual[mask]).mean(),
        })

    importance = permutation_importance(
        model,
        val[features],
        val["value_end_mln"],
        scoring="neg_mean_absolute_error",
        n_repeats=10,
        random_state=42,
    )

    for feature, score in zip(
        features, importance.importances_mean
    ):
        importance_rows.append({
            "Model": name,
            "Feature": feature,
            "MAE_Increase_When_Shuffled": score,
        })

bracket_results = pd.DataFrame(rows)
feature_results = pd.DataFrame(importance_rows)

bracket_results.to_csv(
    OUTPUT / "boosting_errors_by_bracket.csv",
    index=False,
)
feature_results.to_csv(
    OUTPUT / "boosting_feature_importance.csv",
    index=False,
)
prediction_rows.to_csv(
    OUTPUT / "boosting_validation_predictions.csv",
    index=False,
)

print("\nERRORS BY MARKET-VALUE BRACKET")
print(bracket_results.round(3).to_string(index=False))

print("\nTOP BOOSTING FEATURES")
print(
    feature_results[
        feature_results["Model"] == "Shallow Boosting"
    ].sort_values(
        "MAE_Increase_When_Shuffled",
        ascending=False,
    ).round(3).to_string(index=False)
)

print("\nSaved diagnostics to:", OUTPUT)
