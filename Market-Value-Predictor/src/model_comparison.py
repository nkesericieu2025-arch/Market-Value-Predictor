"""Fair model comparisons using random GROUPED CV on development players.

Models use the same fold assignments. Results are development/CV estimates,
not independent generalization estimates after model/hyperparameter selection.
No held-out player is used or scored.
Run: python src/model_comparison.py
"""
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

from data import ROOT, cv_splits, load_data, split_holdout
from pipeline import build_pipeline, evaluate, results_table


def main():
    dev, _holdout = split_holdout(load_data())
    folds = cv_splits(dev, kind="group")
    models = {
        "Median baseline": DummyRegressor(strategy="median"),
        "Linear Regression": LinearRegression(),
        "Ridge (alpha=1)": Ridge(alpha=1.0),
        "Ridge (alpha=10)": Ridge(alpha=10.0),
        "Random Forest (depth=8)": RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=3, random_state=42, n_jobs=-1),
        "Random Forest (unlimited depth)": RandomForestRegressor(
            n_estimators=200, max_depth=None, min_samples_leaf=3, random_state=42, n_jobs=-1),
    }

    rows = []
    for use_start in (False, True):
        for name, model in models.items():
            # Median baseline with start value isn't a meaningful separate case.
            if name == "Median baseline" and use_start:
                continue
            pipe = build_pipeline(model, encoding="grouped", use_start=use_start)
            scores = evaluate(pipe, dev, folds, encoding="grouped", use_start=use_start)
            rows.append({"model": name, "use_start": use_start, "cv": "shuffled-GroupKFold-5",
                         "n_dev": len(dev), **scores})
            print(f"Done: {name:35s} | start: {use_start} | MAE: {scores['MAE_mean']:.3f}", flush=True)

    table = results_table(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    table.to_csv(ROOT / "results" / "model_comparison.csv", index=False)
    print("\nCV model comparison (not an untouched final-test score):")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
