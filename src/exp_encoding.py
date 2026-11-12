"""Professor feedback experiment: does sparse encoding change predictive MAE?

Exactly the same development players, grouped folds, target, and start-value
exclusion for all encoding choices. Tests Ridge and Random Forest. Does NOT
use the held-out player set.
Run: python src/exp_encoding.py
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from data import ROOT, CATEGORICAL, cv_splits, load_data, split_holdout
from pipeline import build_pipeline, evaluate, results_table


def main():
    dev, _holdout = split_holdout(load_data())
    folds = cv_splits(dev, kind="group")
    print("Category cardinality in development data:",
          {name: dev[name].nunique() for name in CATEGORICAL})

    models = {
        "Ridge alpha=10": Ridge(alpha=10),
        "Random Forest depth=8": RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=3,
            random_state=42, n_jobs=-1),
    }
    rows = []
    for name, model in models.items():
        for encoding in ("onehot", "grouped", "drop"):
            scores = evaluate(build_pipeline(model, encoding=encoding),
                              dev, folds, encoding=encoding)
            rows.append({"model": name, "encoding": encoding,
                         "n_folds": len(folds), "n_dev": len(dev), **scores})
            print(f"{name:25s} {encoding:8s} MAE={scores['MAE_mean']:.3f}", flush=True)

    table = results_table(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    table.to_csv(ROOT / "results" / "encoding.csv", index=False)
    print("\nEncoding ablation (same grouped folds; no start valuation):")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
