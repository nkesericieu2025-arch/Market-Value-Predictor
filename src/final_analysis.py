"""SCOUT final development analysis: one script for every result in the outline.

Design (as committed in the progress report of 25 October):
- Data: Nisanov combined sheet with missing values restored (src/data.py), plus
  Understat xG/xA merged by player name and season (professor's 2nd feedback).
- Evaluation: the same shuffled 5-fold GroupKFold by player on the 523
  development players for every model. The 132 reserved players are NOT used
  anywhere in this script; the final test is a separate, team-agreed step.
- Tuning happens inside each training fold only (nested): Ridge and gradient
  boosting by an inner 3-fold grouped CV, random forest by its out-of-bag (OOB)
  error, as the professor suggested. Reported scores always come from the
  untouched outer validation fold.

Outputs: results/final/*.csv and figures/final/*.png
Run from the repository root:  python src/final_analysis.py   (about 3-5 minutes)
"""
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import GROUP, NUMERIC, ROOT, SEED, START, TARGET, cv_splits, load_data, split_holdout
from integrate_understat import SOURCE as UNDERSTAT, normalize_name

warnings.filterwarnings("ignore", category=UserWarning)
RESULTS, FIGURES = ROOT / "results" / "final", ROOT / "figures" / "final"
CAT = ["position", "nationality", "club_name"]
XG = ["xG", "xA"]


# ---------------------------------------------------------------- data
def load_with_xg():
    """Corrected Nisanov data + Understat xG/xA (EPL, same season year)."""
    df = load_data().copy()
    u = pd.read_csv(UNDERSTAT)
    u = u[(u["league"] == "EPL") & u["year"].between(2019, 2023)].copy()
    u["match_name"], u["season"] = u["player_name"].map(normalize_name), u["year"]
    keys = ["match_name", "season"]
    u = u[~u.duplicated(keys, keep=False)]          # never guess ambiguous names
    df["match_name"] = df["player_name"].map(normalize_name)
    df = df.merge(u[keys + XG], on=keys, how="left", validate="many_to_one")
    return df.drop(columns="match_name")


# ---------------------------------------------------------------- models
def preprocessor(num, pca=False):
    steps = [SimpleImputer(strategy="median"), StandardScaler()]
    if pca:
        steps.append(PCA(n_components=0.90, random_state=SEED))
    return ColumnTransformer([
        ("num", make_pipeline(*steps), num),
        ("cat", OneHotEncoder(min_frequency=10, handle_unknown="infrequent_if_exist",
                              sparse_output=False), CAT)])


def pipe(model, num, log=False, pca=False):
    p = Pipeline([("pre", preprocessor(num, pca)), ("model", model)])
    return TransformedTargetRegressor(regressor=p, func=np.log1p, inverse_func=np.expm1) if log else p


GRIDS = {
    "Ridge": [{"alpha": a} for a in (0.1, 1, 3, 10, 30, 100, 300)],
    "Gradient boosting": [{"max_iter": it, "learning_rate": lr, "max_leaf_nodes": lv, "min_samples_leaf": 10}
                          for it in (150, 300) for lr in (0.03, 0.08) for lv in (7, 15)],
    "Random forest": [{"max_features": mf, "min_samples_leaf": ml}
                      for mf in (0.33, 0.5, 1.0) for ml in (1, 3, 5)],
}


def make(name, **params):
    if name == "Linear regression":
        return LinearRegression()
    if name == "Ridge":
        return Ridge(**params)
    if name == "Gradient boosting":
        return HistGradientBoostingRegressor(random_state=SEED, **params)
    return RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1, oob_score=True, **params)


def tune(name, train, num, log=False):
    """Choose hyperparameters using the TRAINING fold only."""
    if name not in GRIDS:
        return {}
    X, y = train[num + CAT], train[TARGET]
    if name == "Random forest":                      # out-of-bag error (professor)
        scores = []
        for params in GRIDS[name]:
            p = pipe(make(name, **params), num).fit(X, y)
            scores.append(mean_absolute_error(y, p["model"].oob_prediction_))
        return GRIDS[name][int(np.argmin(scores))]
    inner = list(GroupKFold(3, shuffle=True, random_state=SEED).split(X, groups=train[GROUP]))
    scores = []
    for params in GRIDS[name]:
        maes = [mean_absolute_error(y.iloc[va], pipe(make(name, **params), num, log)
                                    .fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[va]))
                for tr, va in inner]
        scores.append(np.mean(maes))
    return GRIDS[name][int(np.argmin(scores))]


def oof(name, dev, folds, num, log=False, pca=False, tuned=True):
    """Out-of-fold predictions (nested tuning) and per-fold scores."""
    pred = pd.Series(np.nan, index=dev.index)
    chosen, fold_scores = [], []
    for tr, va in folds:
        train, val = dev.iloc[tr], dev.iloc[va]
        params = tune(name, train, num, log) if tuned else {}
        model = make(name, **params)
        if name == "Random forest":
            model.set_params(oob_score=False)
        p = pipe(model, num, log, pca).fit(train[num + CAT], train[TARGET])
        pred.iloc[va] = p.predict(val[num + CAT])
        fold_scores.append(metrics(val[TARGET], pred.iloc[va]))
        chosen.append(params)
    return pred, pd.DataFrame(fold_scores), chosen


def metrics(y, p):
    return {"MAE": mean_absolute_error(y, p), "RMSE": root_mean_squared_error(y, p), "R2": r2_score(y, p)}


def summary(label, fold_df, **extra):
    row = {"model": label, **extra}
    for m in ("MAE", "RMSE", "R2"):
        row[f"{m}_mean"], row[f"{m}_sd"] = fold_df[m].mean(), fold_df[m].std(ddof=0)
    return row


# ---------------------------------------------------------------- baselines
def baselines(dev, folds):
    rows = []
    for label in ("Median baseline", "Carry-forward start value",
                  "Start value x (1 + mean growth)", "Start value x (1 + median growth)"):
        scores = []
        for tr, va in folds:
            train, val = dev.iloc[tr], dev.iloc[va]
            fallback = train[TARGET].median()
            if label == "Median baseline":
                p = pd.Series(fallback, index=val.index)
            else:
                known = train[START].notna() & (train[START] > 0)
                growth = (train.loc[known, TARGET] - train.loc[known, START]) / train.loc[known, START]
                g = 0 if label.startswith("Carry") else (growth.mean() if "mean" in label else growth.median())
                p = (val[START] * (1 + g)).fillna(fallback)
            scores.append(metrics(val[TARGET], p))
        rows.append(summary(label, pd.DataFrame(scores), start_value=label != "Median baseline"))
    return rows


# ---------------------------------------------------------------- main
def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = load_with_xg()
    dev, _reserved = split_holdout(data)            # reserved players: never used here
    folds = cv_splits(dev, kind="group")
    base_num = NUMERIC + XG
    print(f"Development: {len(dev)} rows, {dev[GROUP].nunique()} players | "
          f"xG/xA matched for {dev['xG'].notna().mean():.1%} of rows")

    # T1. Main model comparison --------------------------------------------
    rows, preds, params_log = baselines(dev, folds), {}, []
    for use_start in (False, True):
        num = base_num + ([START] if use_start else [])
        for name in ("Linear regression", "Ridge", "Random forest", "Gradient boosting"):
            p, s, chosen = oof(name, dev, folds, num)
            rows.append(summary(name, s, start_value=use_start))
            preds[(name, use_start)] = p
            params_log += [{"model": name, "start_value": use_start, "fold": i + 1, **c}
                           for i, c in enumerate(chosen)]
            print(f"  T1 {name:18s} start={use_start!s:5s} MAE {s['MAE'].mean():.2f}")
    t1 = pd.DataFrame(rows)
    t1.round(3).to_csv(RESULTS / "T1_model_comparison.csv", index=False)
    pd.DataFrame(params_log).to_csv(RESULTS / "T1_chosen_hyperparameters.csv", index=False)

    # T2. Does xG/xA add information? (no start value) ----------------------
    rows = []
    for name in ("Ridge", "Random forest", "Gradient boosting"):
        for feats, num in (("without xG/xA", NUMERIC), ("with xG/xA", base_num)):
            _, s, _ = oof(name, dev, folds, num)
            rows.append(summary(name, s, features=feats))
    pd.DataFrame(rows).round(3).to_csv(RESULTS / "T2_xg_xa_ablation.csv", index=False)
    print("  T2 done")

    # T3. Random forest: OOB error vs grouped CV ----------------------------
    best_rf = pd.DataFrame(params_log).query("model == 'Random forest' and not start_value")
    rf_params = best_rf[["max_features", "min_samples_leaf"]].mode().iloc[0].to_dict()
    rf_params["min_samples_leaf"] = int(rf_params["min_samples_leaf"])
    full = pipe(make("Random forest", **rf_params), base_num).fit(dev[base_num + CAT], dev[TARGET])
    oob_pred = full["model"].oob_prediction_
    cv_rf = t1.query("model == 'Random forest' and not start_value").iloc[0]
    repeat_share = dev.groupby(GROUP)["season"].transform("count").gt(1).mean()
    pd.DataFrame([
        {"estimate": "Out-of-bag (rows resampled)", **metrics(dev[TARGET], oob_pred)},
        {"estimate": "Grouped 5-fold CV (players held out)", "MAE": cv_rf["MAE_mean"],
         "RMSE": cv_rf["RMSE_mean"], "R2": cv_rf["R2_mean"]},
    ]).assign(rows_from_players_with_several_seasons=repeat_share).round(3).to_csv(
        RESULTS / "T3_oob_vs_grouped_cv.csv", index=False)
    print("  T3 done")

    # T4. log1p target and T5. PCA (no start value) -------------------------
    rows = []
    for name in ("Ridge", "Random forest", "Gradient boosting"):
        for log in (False, True):
            _, s, _ = oof(name, dev, folds, base_num, log=log)
            rows.append(summary(name, s, target="log1p" if log else "original"))
    pd.DataFrame(rows).round(3).to_csv(RESULTS / "T4_log_target.csv", index=False)
    n_comp = preprocessor(base_num, pca=True).fit(dev[base_num + CAT]).named_transformers_["num"][-1].n_components_
    rows = []
    for use_pca in (False, True):
        _, s, _ = oof("Ridge", dev, folds, base_num, pca=use_pca)
        rows.append(summary("Ridge", s, numeric_inputs=f"{n_comp} PCA components (90% variance)"
                            if use_pca else f"{len(base_num)} original variables"))
    pd.DataFrame(rows).round(3).to_csv(RESULTS / "T5_pca.csv", index=False)
    print("  T4/T5 done")

    # T6. Importance of each ORIGINAL variable (no start value) -------------
    imp = {}
    for name in ("Ridge", "Random forest", "Gradient boosting"):
        per_fold = []
        for tr, va in folds:
            train, val = dev.iloc[tr], dev.iloc[va]
            model = make(name, **tune(name, train, base_num))
            if name == "Random forest":
                model.set_params(oob_score=False)
            p = pipe(model, base_num).fit(train[base_num + CAT], train[TARGET])
            r = permutation_importance(p, val[base_num + CAT], val[TARGET], n_repeats=5,
                                       scoring="neg_mean_absolute_error", random_state=SEED, n_jobs=-1)
            per_fold.append(r.importances_mean)
        imp[name] = np.mean(per_fold, axis=0)
    t6 = pd.DataFrame(imp, index=base_num + CAT)
    names = full["pre"].get_feature_names_out()
    origin = [next(c for c in base_num + CAT if n.split("__", 1)[1] == c or n.split("__", 1)[1].startswith(c + "_"))
              for n in names]
    t6["RF impurity share"] = pd.Series(full["model"].feature_importances_).groupby(origin).sum()
    t6 = t6.sort_values("Gradient boosting", ascending=False)
    t6.round(3).to_csv(RESULTS / "T6_feature_importance.csv", index_label="variable")
    print("  T6 done")

    # T7. Where does the best performance-only model fail? ------------------
    best = t1.query("not start_value and model in ['Ridge','Random forest','Gradient boosting']") \
             .sort_values("MAE_mean").iloc[0]["model"]
    err = dev[[GROUP, "season", "position", TARGET, START]].copy()
    err["predicted"] = preds[(best, False)]
    err["abs_error"] = (err[TARGET] - err["predicted"]).abs()
    err["value_band"] = pd.cut(err[TARGET], [-0.01, 5, 15, 30, 60, 200],
                               labels=["<=5", "5-15", "15-30", "30-60", ">60"])
    groups = {"Goalkeeper": "Other", "Centre-Back": "Defender", "Left-Back": "Defender",
              "Right-Back": "Defender", "Defensive Midfield": "Midfielder", "Central Midfield": "Midfielder",
              "Attacking Midfield": "Midfielder", "Left Midfield": "Midfielder", "Right Midfield": "Midfielder"}
    err["position_group"] = err["position"].map(groups).fillna("Forward/Winger")
    t7 = pd.concat([err.groupby(k, observed=True).agg(n=("abs_error", "size"), MAE=("abs_error", "mean"),
                                                       median_value=(TARGET, "median"))
                    .reset_index().rename(columns={k: "group"}).assign(by=k)
                    for k in ("position_group", "value_band", "season")])
    t7.round(3).to_csv(RESULTS / "T7_errors_by_group.csv", index=False)
    err.to_csv(RESULTS / "oof_predictions_best_model.csv", index=False)

    # T8. Case study: players with no prior market value ---------------------
    case = err[err[START].isna()][[GROUP, "season", "position", TARGET, "predicted"]]
    case.round(2).to_csv(RESULTS / "T8_players_without_start_value.csv", index=False)
    print(f"  T7/T8 done (best performance-only model: {best})")

    figures(dev, t1, t6, err, best)
    print(f"Saved tables to {RESULTS} and figures to {FIGURES}")


# ---------------------------------------------------------------- figures
def figures(dev, t1, t6, err, best):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    # F1. Unit of observation and role of seasons (Communication Skills feedback)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    seasons = sorted(dev["season"].unique())
    data = [dev.loc[dev["season"] == s, TARGET] for s in seasons]
    ax.boxplot(data, showfliers=True, widths=0.55, medianprops={"color": "#c0392b"})
    ax.set_xticks(range(1, len(seasons) + 1),
                  [f"{s}/{str(s + 1)[2:]}\n(n = {len(d)})" for s, d in zip(seasons, data)])
    ax.set_xlabel("Season (one observation = one player in one season)")
    ax.set_ylabel("End-of-season market value (millions)")
    fig.tight_layout(); fig.savefig(FIGURES / "F1_value_by_season.png", dpi=220); plt.close(fig)

    # F2. Model comparison with and without start value
    models = ["Median baseline", "Carry-forward start value", "Linear regression", "Ridge",
              "Random forest", "Gradient boosting"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(models))
    for i, (flag, label) in enumerate(((False, "without start value"), (True, "with start value"))):
        sub = t1[t1["start_value"] == flag].set_index("model").reindex(models)
        ax.bar(x + (i - 0.5) * 0.38, sub["MAE_mean"], 0.38, yerr=sub["MAE_sd"], capsize=3, label=label)
    ax.set_xticks(x, [m.replace(" start value", "\nstart value").replace(" baseline", "\nbaseline")
                      .replace(" regression", "\nregression").replace(" boosting", "\nboosting")
                      .replace(" forest", "\nforest") for m in models])
    ax.set_ylabel("MAE (millions), mean ± SD over 5 folds")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIGURES / "F2_model_comparison.png", dpi=220); plt.close(fig)

    # F3. Feature importance of original variables
    top = t6.head(10)[::-1]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    y = np.arange(len(top))
    for i, name in enumerate(("Ridge", "Random forest", "Gradient boosting")):
        ax.barh(y + (i - 1) * 0.27, top[name], 0.27, label=name)
    labels = {"club_name": "Club", "age_then": "Age", "total_goals": "Goals", "total_minutes": "Minutes",
              "xA": "Expected assists (xA)", "xG": "Expected goals (xG)", "relegation": "Club relegated",
              "assists": "Assists", "height": "Height", "club_performance": "Club performance",
              "total_games": "Games", "position": "Position", "nationality": "Nationality",
              "penalty_kicks": "Penalties", "success_or_not": "Club success"}
    ax.set_yticks(y, [labels.get(v, v) for v in top.index])
    ax.set_xlabel("Increase in validation MAE when the variable is shuffled (millions)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(FIGURES / "F3_feature_importance.png", dpi=220); plt.close(fig)

    # F4. Actual vs predicted, best performance-only model
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.scatter(err[TARGET], err["predicted"], s=10, alpha=0.5)
    lim = [0, max(err[TARGET].max(), err["predicted"].max()) * 1.05]
    ax.plot(lim, lim, color="grey", lw=1, ls="--")
    ax.set(xlim=lim, ylim=lim, xlabel="Actual end-of-season value (millions)",
           ylabel=f"Out-of-fold prediction (millions)")
    ax.set_title(f"{best}, no start value", fontsize=10)
    fig.tight_layout(); fig.savefig(FIGURES / "F4_actual_vs_predicted.png", dpi=220); plt.close(fig)


if __name__ == "__main__":
    main()
