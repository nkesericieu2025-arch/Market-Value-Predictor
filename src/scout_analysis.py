"""SCOUT market-value prediction and interpretation pipeline.

Run from the repository root:
    python3 src/scout_analysis.py

Outputs are written to results/scout/ and figures/scout/.
The data file is read from data/Nisanov_final_data.xlsx, worksheet '2019'
(the workbook's combined 1,382-row dataset, not the individual year tabs).
"""
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Nisanov_final_data.xlsx"
TARGET = "value_end_mln"
START_VALUE = "value_0_mln"
SEED = 42

# Exclude identifiers, the target, duplicate valuation columns, and any fields
# derived from the end value. age_now is excluded because it is not a value
# available at the end of the corresponding season.
LEAKAGE_OR_ID = {
    "player_name", "season", TARGET, "end_value", "delta_value",
    "value_delta_mln", "value_end_mln", "value_delta_mln", "age_now",
    "start_value", "value_0_mln",
}

OUT_RESULTS = ROOT / "results" / "scout"
OUT_FIGURES = ROOT / "figures" / "scout"


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Keep Nisanov_final_data.xlsx in data/."
        )
    df = pd.read_excel(DATA_PATH, sheet_name="2019")
    required = {TARGET, "season", "player_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    df = df.copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=["season", TARGET])
    return df


def make_preprocessor(X, scale_numeric):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    transformers = [("numeric", Pipeline(num_steps), numeric)]
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
        ]), categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def model_specs():
    return {
        "Median baseline": (DummyRegressor(strategy="median"), False),
        "Linear Regression": (LinearRegression(), True),
        "Ridge (alpha=1)": (Ridge(alpha=1.0), True),
        "Ridge (alpha=10)": (Ridge(alpha=10.0), True),
        "Random Forest (depth=8)": (RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=3,
            random_state=SEED, n_jobs=-1), False),
        "Random Forest (unlimited depth)": (RandomForestRegressor(
            n_estimators=300, max_depth=None, min_samples_leaf=3,
            random_state=SEED, n_jobs=-1), False),
    }


def feature_sets(df):
    base = [c for c in df.columns if c not in LEAKAGE_OR_ID]
    with_start = base + ([START_VALUE] if START_VALUE in df.columns else [])
    return {
        "Performance_only": base,
        "With_start_value": with_start,
    }


def metrics(y, pred):
    return {
        "MAE_GBP_millions_assumed": mean_absolute_error(y, pred),
        "RMSE_GBP_millions_assumed": mean_squared_error(y, pred) ** 0.5,
        "R2": r2_score(y, pred),
    }


def save_figures(df, test_results, best_model, best_X_test, y_test, best_feature_names):
    plt.style.use("seaborn-v0_8-whitegrid")
    # 1. Target distribution, with a robustly readable long tail.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(df[TARGET], bins=35, edgecolor="white", alpha=.9)
    ax.axvline(df[TARGET].median(), linestyle="--", linewidth=2,
               label=f"Median £{df[TARGET].median():.1f}m (assumed)")
    ax.set(title="Distribution of estimated end market values",
           xlabel="Estimated market value (£ millions; currency assumption)", ylabel="Player-season observations")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT_FIGURES / "01_market_value_distribution.png", dpi=220); plt.close(fig)

    # 2. Missingness by variable.
    miss = df.isna().mean().mul(100).sort_values()
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(9, max(4, .36 * len(miss))))
    if len(miss):
        ax.barh(miss.index, miss.values)
        ax.set(xlabel="Missing values (%)", title="Missingness by variable")
    else:
        ax.text(.5, .5, "No missing values", ha="center", va="center")
    fig.tight_layout(); fig.savefig(OUT_FIGURES / "02_missing_values.png", dpi=220); plt.close(fig)

    # 3. Observations by season.
    counts = df.groupby("season").size()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(counts.index.astype(int).astype(str), counts.values, edgecolor="white")
    ax.set(title="Dataset observations by season", xlabel="Season", ylabel="Player-season observations")
    fig.tight_layout(); fig.savefig(OUT_FIGURES / "03_observations_by_season.png", dpi=220); plt.close(fig)

    # 4. Age versus target when age exists.
    if "age_then" in df:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        ax.scatter(df["age_then"], df[TARGET], alpha=.45, s=24, edgecolors="none")
        ax.set(title="Age and estimated market value", xlabel="Age during season", ylabel="Estimated market value (£ millions; assumed)")
        fig.tight_layout(); fig.savefig(OUT_FIGURES / "04_age_vs_market_value.png", dpi=220); plt.close(fig)

    # 5. Actual vs predicted.
    pred = best_model.predict(best_X_test)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, pred, alpha=.65, s=30, edgecolors="none")
    lo = min(float(np.min(y_test)), float(np.min(pred)))
    hi = max(float(np.max(y_test)), float(np.max(pred)))
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=2, label="Perfect prediction")
    ax.set(title="Held-out 2023: actual vs predicted", xlabel="Actual estimated market value (£m; assumed)", ylabel="Predicted estimated market value (£m; assumed)")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT_FIGURES / "05_actual_vs_predicted_2023.png", dpi=220); plt.close(fig)

    # 6. Residuals.
    residual = y_test - pred
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.scatter(pred, residual, alpha=.6, s=28, edgecolors="none")
    ax.axhline(0, linestyle="--", linewidth=2)
    ax.set(title="Held-out 2023: residual diagnostics", xlabel="Predicted market value (£m; assumed)", ylabel="Actual − predicted (£m; assumed)")
    fig.tight_layout(); fig.savefig(OUT_FIGURES / "06_residuals_2023.png", dpi=220); plt.close(fig)


def main():
    OUT_RESULTS.mkdir(parents=True, exist_ok=True)
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    df = load_data()
    print(f"Loaded {len(df):,} observations from combined worksheet '2019'.")
    print("Seasons:", sorted(df["season"].dropna().astype(int).unique().tolist()))
    print("NOTE: target columns are named *_mln; plots label values as GBP millions based on the project assumption. Verify currency/scale against dataset documentation before final submission.")

    # EDA tables.
    df.describe(include="all").transpose().to_csv(OUT_RESULTS / "descriptive_statistics.csv")
    df.isna().sum().rename("missing_count").to_csv(OUT_RESULTS / "missing_values.csv")
    df.groupby("season").size().rename("observations").to_csv(OUT_RESULTS / "observations_by_season.csv")

    # Chronological split: 2019-2021 train, 2022 validation, 2023 final test.
    train = df[df["season"] <= 2021].copy()
    valid = df[df["season"] == 2022].copy()
    test = df[df["season"] == 2023].copy()
    if min(len(train), len(valid), len(test)) == 0:
        raise ValueError("Expected non-empty train (<=2021), validation (2022), and test (2023) sets.")
    print(f"Chronological split: train={len(train)}, validation={len(valid)}, test={len(test)}")

    val_rows, test_rows = [], []
    selected = {}
    all_features = feature_sets(df)
    for feature_set_name, columns in all_features.items():
        columns = [c for c in columns if c in df.columns and c != TARGET]
        # fit validation candidates on training seasons only
        X_train, y_train = train[columns], train[TARGET]
        X_valid, y_valid = valid[columns], valid[TARGET]
        candidates = []
        for name, (estimator, scale) in model_specs().items():
            pipe = Pipeline([("preprocess", make_preprocessor(X_train, scale)), ("model", estimator)])
            pipe.fit(X_train, y_train)
            pred_valid = pipe.predict(X_valid)
            row = {"feature_set": feature_set_name, "model": name, "n_features_raw": len(columns), **metrics(y_valid, pred_valid)}
            val_rows.append(row)
            candidates.append((row["MAE_GBP_millions_assumed"], name, estimator, scale))
        # select by validation MAE, then refit using train + validation; test used once for final evaluation
        _, best_name, best_estimator, best_scale = min(candidates, key=lambda x: x[0])
        trainval = pd.concat([train, valid], ignore_index=True)
        X_trainval, y_trainval = trainval[columns], trainval[TARGET]
        X_test, y_test = test[columns], test[TARGET]
        final_pipe = Pipeline([("preprocess", make_preprocessor(X_trainval, best_scale)), ("model", best_estimator)])
        final_pipe.fit(X_trainval, y_trainval)
        pred_test = final_pipe.predict(X_test)
        test_rows.append({"feature_set": feature_set_name, "selected_model": best_name, "n_features_raw": len(columns), **metrics(y_test, pred_test)})
        selected[feature_set_name] = (final_pipe, X_test, y_test, columns, best_name)

        # Final-model permutation importance on held-out test. Interpret as predictive importance, not causation.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imp = permutation_importance(final_pipe, X_test, y_test, scoring="neg_mean_absolute_error", n_repeats=8, random_state=SEED, n_jobs=-1)
        importance = pd.DataFrame({"feature": columns, "permutation_importance_mae_increase": imp.importances_mean,
                                   "permutation_importance_std": imp.importances_std}).sort_values("permutation_importance_mae_increase", ascending=False)
        importance.to_csv(OUT_RESULTS / f"feature_importance_{feature_set_name.lower()}.csv", index=False)
        if feature_set_name == "Performance_only":
            fig, ax = plt.subplots(figsize=(9, 6.5))
            top = importance.head(15).sort_values("permutation_importance_mae_increase")
            ax.barh(top["feature"], top["permutation_importance_mae_increase"], xerr=top["permutation_importance_std"], capsize=2)
            ax.set(title=f"Top predictive features — {best_name}", xlabel="Increase in test MAE when shuffled (higher = more important)", ylabel="Feature")
            fig.tight_layout(); fig.savefig(OUT_FIGURES / "07_permutation_feature_importance.png", dpi=220); plt.close(fig)

    val_df = pd.DataFrame(val_rows).sort_values(["feature_set", "MAE_GBP_millions_assumed"])
    test_df = pd.DataFrame(test_rows).sort_values("feature_set")
    val_df.to_csv(OUT_RESULTS / "validation_model_comparison.csv", index=False)
    test_df.to_csv(OUT_RESULTS / "final_test_model_comparison.csv", index=False)

    # Main diagnostic charts use the best model chosen for performance-only features.
    pipe, X_test, y_test, cols, best_name = selected["Performance_only"]
    save_figures(df, test_df, pipe, X_test, y_test, cols)

    summary = [
        "SCOUT preliminary run summary", "==============================", "",
        f"Observations loaded: {len(df)}", f"Target: {TARGET}",
        "Currency/scale: assumed GBP millions for labels only; verify documentation before final report.",
        f"Train seasons <=2021: {len(train)}", f"Validation season 2022: {len(valid)}", f"Held-out test season 2023: {len(test)}", "",
        "Validation results are used to select models. The 2023 test set is held out until final evaluation.",
        "Feature importance is predictive association, not causal influence.",
        "This dataset appears to contain player-season rows; confirm league scope and sample selection in the source documentation.",
        "Season-end statistics mean this setup is not a pre-season forecast.", "",
        "See validation_model_comparison.csv and final_test_model_comparison.csv for metrics.",
    ]
    (OUT_RESULTS / "RUN_SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")
    print("\nVALIDATION MODEL COMPARISON (choose model here; lower MAE is better)")
    print(val_df[["feature_set", "model", "MAE_GBP_millions_assumed", "RMSE_GBP_millions_assumed", "R2"]].round(3).to_string(index=False))
    print("\nFINAL HELD-OUT 2023 TEST RESULTS")
    print(test_df.round(3).to_string(index=False))
    print(f"\nSaved tables to {OUT_RESULTS}")
    print(f"Saved charts to {OUT_FIGURES}")

if __name__ == "__main__":
    main()
