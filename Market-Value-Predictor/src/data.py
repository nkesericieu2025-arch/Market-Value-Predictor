"""Shared SCOUT dataset and RANDOM, player-grouped evaluation splits.

The combined '2019' sheet holds observations from 2019-2023. Our prediction
is end-of-season estimated market value using information available by the
end of that season; it is NOT a pre-season forecast.

Primary evaluation: GroupShuffleSplit for development/holdout, then shuffled
GroupKFold for CV within development. These splits are random across seasons
but keep *all seasons of each player together*.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Nisanov_final_data.xlsx"
SEED = 42
TARGET = "value_end_mln"
GROUP = "player_name"
COMBINED_SHEET = "2019"
NUMERIC = ["age_then", "height", "total_games", "total_minutes", "total_goals",
           "assists", "penalty_kicks", "club_performance", "relegation",
           "success_or_not"]
START = "value_0_mln"
CATEGORICAL = ["position", "nationality", "club_name"]
EXCLUDED = ["end_value", "start_value", "delta_value", "value_delta_mln",
            "age_now", "player_name", "season"]


def load_data():
    """Read all 1,382 source observations and restore missingness.

    The dataset's *_mln numeric columns misleadingly substitute zero when the
    corresponding raw value is empty. Recover missingness from the raw columns:
      * 10 missing 'start_value' -> NaN in START (imputed inside CV folds).
      * 1 missing 'end_value' -> unobserved TARGET; excluded from CV after
        generating the holdout on the ORIGINAL player list to preserve groups.
    Do NOT change genuine zero values where their raw source is zero.
    """
    df = pd.read_excel(DATA_PATH, sheet_name=COMBINED_SHEET)
    if len(df) != 1382 or sorted(df["season"].unique()) != [2019, 2020, 2021, 2022, 2023]:
        raise ValueError("Unexpected source dataset size or season coverage")
    if df.duplicated([GROUP, "season"]).any():
        raise ValueError("Duplicate player-season rows")
    if not np.allclose(df.loc[df['start_value'].notna(), 'start_value'] / 1e6,
                       df.loc[df['start_value'].notna(), START]):
        raise ValueError("Starting value columns disagree")
    if not np.allclose(df.loc[df['end_value'].notna(), 'end_value'] / 1e6,
                       df.loc[df['end_value'].notna(), TARGET]):
        raise ValueError("Target value columns disagree")

    df = df.copy()
    df.loc[df["start_value"].isna(), START] = np.nan
    df.loc[df["end_value"].isna(), TARGET] = np.nan
    return df


def split_holdout(df, test_size=0.2):
    """Split players at random, independently of season; return dev, holdout.

    Historical warning: the earlier chronological analysis evaluated 2023,
    which overlaps part of this holdout. Avoid describing it as fully unseen.
    Do not use holdout for today's model/feature selection.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    dev_idx, hold_idx = next(gss.split(df, groups=df[GROUP]))
    # Group assignment is fixed before removing the one invalid TARGET row,
    # so missing-target correction does not reshuffle the reserved players.
    dev = df.iloc[dev_idx].dropna(subset=[TARGET]).reset_index(drop=True)
    holdout = df.iloc[hold_idx].dropna(subset=[TARGET]).reset_index(drop=True)
    return dev, holdout


def cv_splits(dev, kind="group", n_splits=5):
    """Folds on development rows only.

    group: primary RANDOM 5-fold CV by player, irrespective of season.
    random: diagnostic only; ordinary shuffled rows can repeat players across folds.
    chronological: historical robustness diagnostic, not primary evaluation.
    """
    if kind == "group":
        cv = GroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        return list(cv.split(dev, groups=dev[GROUP]))
    if kind == "random":
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        return list(cv.split(dev))
    if kind == "chronological":
        years = sorted(dev["season"].unique())
        return [(np.where(dev["season"] < s)[0], np.where(dev["season"] == s)[0])
                for s in years[1:]]
    raise ValueError(f"Unknown split type: {kind}")


if __name__ == "__main__":
    df = load_data()
    dev, holdout = split_holdout(df)
    print("SCOUT dataset: source=", len(df), "; modeling=", len(dev)+len(holdout))
    print("Dev rows/players:", len(dev), dev[GROUP].nunique())
    print("Reserved holdout rows/players:", len(holdout), holdout[GROUP].nunique())
    assert not set(dev[GROUP]) & set(holdout[GROUP])
    for kind in ("group", "random", "chronological"):
        folds = cv_splits(dev, kind)
        print(kind, 'validation sizes:', [len(v) for _, v in folds])
        if kind == "group":
            assert all(not set(dev.iloc[t][GROUP]) & set(dev.iloc[v][GROUP]) for t,v in folds)
    print("Missing start values on model rows:", int(dev[START].isna().sum()+holdout[START].isna().sum()))
    print("OK")
