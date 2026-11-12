
"""
SCOUT: Compare regression models for football player market values.

Run from the repository root:
    python3 src/model_comparison.py

This first experiment compares models using the team's existing
cross-validation framework. The held-out test set is not used here.
"""

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

from data import TARGET, load_data, split_holdout, cv_splits
from pipeline import build_pipeline, evaluate, results_table


def main():
    # Load the dataset and create a development set.
    # Keep the held-out test set untouched during model selection.
    development, _test = split_holdout(load_data())

    # Use the existing grouped cross-validation strategy.
    folds = cv_splits(development, kind="group")

    # Models to compare. Start with the existing preprocessing pipeline.
    models = {
        "Median baseline": DummyRegressor(strategy="median"),
        "Linear Regression": LinearRegression(),
        "Ridge (alpha=1)": Ridge(alpha=1.0),
        "Ridge (alpha=10)": Ridge(alpha=10.0),
        "Random Forest (depth=8)": RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
        "Random Forest (unlimited depth)": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []

    for name, model in models.items():
        # build_pipeline handles missing values, scaling and categories.
        pipeline = build_pipeline(model)

        # Evaluate with cross-validation on development data only.
        scores = evaluate(pipeline, development, folds)

        results.append({
            "model": name,
            **scores,
        })

    # Print and save a table for the project.
    table = results_table(results)
    print("\nSCOUT MODEL COMPARISON")
    print(table.to_string(index=False))

    output_dir = "results"
    import os
    os.makedirs(output_dir, exist_ok=True)
    table.to_csv(
        os.path.join(output_dir, "model_comparison.csv"),
        index=False,
    )
    print("\nSaved results/model_comparison.csv")


if __name__ == "__main__":
    main()
