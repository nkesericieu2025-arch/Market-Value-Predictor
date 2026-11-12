"""Professor feedback experiment: split strategy can change apparent accuracy.

Primary: shuffled GroupKFold keeping a player in a single fold. Diagnostics:
shuffled KFold of player-season rows (repeat-player overlap) and chronological
expanding windows (train past years/validate next). The latter has FOUR folds
instead of five, and its training/validation sample sizes differ: *not*
like-for-like estimators. Treat differences as exploratory, not proof of leakage.
Holdout players are NEVER evaluated.
Run: python src/exp_splits.py
"""
from sklearn.linear_model import Ridge
from data import ROOT, GROUP, cv_splits, load_data, split_holdout
from pipeline import build_pipeline, evaluate, results_table


def main():
    dev, _holdout = split_holdout(load_data())
    rows = []
    for kind in ("group", "random", "chronological"):
        folds = cv_splits(dev, kind=kind)
        overlaps = [len(set(dev.iloc[tr][GROUP]) & set(dev.iloc[val][GROUP]))
                    for tr, val in folds]
        overlap_validation_rows = [int(dev.iloc[val][GROUP].isin(set(dev.iloc[tr][GROUP])).sum())
                                   for tr, val in folds]
        scores = evaluate(build_pipeline(Ridge(alpha=10), encoding="grouped"),
                          dev, folds, encoding="grouped")
        rows.append({"kind": kind, "n_folds": len(folds), "n_dev": len(dev),
                     "overlap_players_mean": sum(overlaps)/len(overlaps),
                     "overlap_val_rows_mean": sum(overlap_validation_rows)/len(overlap_validation_rows),
                     "train_rows_min": min(len(tr) for tr, val in folds),
                     "validation_rows_min": min(len(val) for tr, val in folds),
                     **scores})
        print(f"{kind:14s} MAE={scores['MAE_mean']:.3f} | "
              f"shared players per fold={sum(overlaps)/len(overlaps):.1f}", flush=True)
    table = results_table(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    table.to_csv(ROOT / "results" / "splits.csv", index=False)
    print("\nDiagnostic: different estimands/fold sizes, not a fair model leaderboard")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
