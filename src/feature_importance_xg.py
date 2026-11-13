"""SCOUT: Feature importance and error analysis for outfield players."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path("data/processed/scout_outfield.csv")
RESULTS = Path("results/scout")
FIGURES = Path("figures/scout")

RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)

features = [
    "age_then", "position", "nationality",
    "total_games", "assists", "penalty_kicks",
    "total_minutes", "total_goals", "height",
    "xG", "xA",
]

target = "value_end_mln"

# Same matched sample as the earlier comparison.
df = df.dropna(subset=[target, "xG", "xA"]).copy()

train = df[df["season"].isin([2019, 2020, 2021])]
val = df[df["season"] == 2022]

numeric = [
    c for c in features
    if pd.api.types.is_numeric_dtype(df[c])
]
categorical = [c for c in features if c not in numeric]

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical),
])

models = {
    "Ridge": Ridge(alpha=10),
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    ),
}

all_importances = []

for name, model in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    pipe.fit(train[features], train[target])
    predictions = pipe.predict(val[features])

    print(f"\n{name}")
    print("Validation MAE:", round(
        mean_absolute_error(val[target], predictions), 3
    ))
    print("Validation R2:", round(
        r2_score(val[target], predictions), 3
    ))

    # Permute one raw feature at a time on validation data.
    # Positive importance means shuffling it increases prediction error.
    importance = permutation_importance(
        pipe,
        val[features],
        val[target],
        scoring="neg_mean_absolute_error",
        n_repeats=15,
        random_state=42,
        n_jobs=-1,
    )

    result = pd.DataFrame({
        "Feature": features,
        "Importance": importance.importances_mean,
        "Std": importance.importances_std,
        "Model": name,
    }).sort_values("Importance", ascending=False)

    all_importances.append(result)

    print(result[["Feature", "Importance"]].round(3).to_string(index=False))

    plot_data = result.sort_values("Importance")

    plt.figure(figsize=(9, 6))
    plt.barh(plot_data["Feature"], plot_data["Importance"])
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Increase in MAE when feature is shuffled")
    plt.title(f"SCOUT Feature Importance — {name}")
    plt.tight_layout()

    filename = name.lower().replace(" ", "_")
    plt.savefig(FIGURES / f"importance_{filename}.png", dpi=180)
    plt.close()

    # Examine prediction errors by actual market-value bracket.
    errors = pd.DataFrame({
        "Actual": val[target].to_numpy(),
        "Predicted": predictions,
    })
    errors["Absolute_Error"] = abs(
        errors["Actual"] - errors["Predicted"]
    )

    # Quartiles are relative to this validation sample.
    errors["Value_Bracket"] = pd.qcut(
        errors["Actual"],
        q=4,
        labels=False,
        duplicates="drop",
    )

    bracket_results = errors.groupby("Value_Bracket").agg(
        Count=("Actual", "size"),
        Mean_Actual_Value=("Actual", "mean"),
        MAE=("Absolute_Error", "mean"),
    )

    print("\nErrors by value bracket:")
    print(bracket_results.round(3).to_string())

    errors.to_csv(
        RESULTS / f"errors_{filename}_2022.csv",
        index=False,
    )

    plt.figure(figsize=(7, 6))
    plt.scatter(errors["Actual"], errors["Predicted"], alpha=0.5)

    lower = min(errors["Actual"].min(), errors["Predicted"].min())
    upper = max(errors["Actual"].max(), errors["Predicted"].max())
    plt.plot([lower, upper], [lower, upper], "k--")

    plt.xlabel("Actual market value")
    plt.ylabel("Predicted market value")
    plt.title(f"SCOUT Actual vs Predicted — {name}, 2022")
    plt.tight_layout()
    plt.savefig(FIGURES / f"actual_predicted_{filename}.png", dpi=180)
    plt.close()

combined = pd.concat(all_importances, ignore_index=True)
combined.to_csv(RESULTS / "xg_feature_importance.csv", index=False)

print("\nAnalysis complete.")
print("Results:", RESULTS)
print("Figures:", FIGURES)
