"""Shared data loading and splitting for SCOUT.

Everyone imports from here, so the whole team uses the same rows,
the same features, the same held-out test players and the same folds.

Key decisions (see README):
- Only the worksheet named "2019" is used: it already combines 2019-2023.
- 20% of PLAYERS are held out as a final test set and are not touched
  until the final report (all experiments use the remaining "dev" set).
- Cross-validation is grouped by player, so one player's seasons never
  appear in both the training and the validation fold.
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

NUMERIC = ["age_then", "height", "total_games", "total_minutes", "total_goals",
           "assists", "penalty_kicks", "club_performance", "relegation",
           "success_or_not"]
START = "value_0_mln"            # start-of-season value: used with AND without
CATEGORICAL = ["position", "nationality", "club_name"]

# Never used as predictors:
#   end_value, start_value       -> same values as target / value_0_mln, other units
#   delta_value, value_delta_mln -> computed from the target (leakage)
#   age_now                      -> measured after the season
#   player_name                  -> identifier (used only for grouping)
EXCLUDED = ["end_value", "start_value", "delta_value", "value_delta_mln",
            "age_now", "player_name"]


def load_data():
    """All 1,382 player-season rows from the combined worksheet."""
    return pd.read_excel(DATA_PATH, sheet_name="2019")


def split_holdout(df, test_size=0.2):
    """Return (dev, test). Test = ~20% of players, all their seasons.

    DO NOT evaluate anything on `test` before the final model is chosen.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    dev_idx, test_idx = next(gss.split(df, groups=df[GROUP]))
    return df.iloc[dev_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def cv_splits(dev, kind="group", n_splits=5):
    """List of (train_idx, val_idx) pairs on the dev set.

    kind="group"         main design: random folds, players kept together
    kind="random"        plain KFold (players can leak across folds)
    kind="chronological" train on earlier seasons, validate on the next one
    """
    if kind == "group":
        cv = GroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        return list(cv.split(dev, groups=dev[GROUP]))
    if kind == "random":
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        return list(cv.split(dev))
    if kind == "chronological":
        seasons = sorted(dev["season"].unique())
        return [(np.where(dev["season"] < s)[0], np.where(dev["season"] == s)[0])
                for s in seasons[1:]]
    raise ValueError(f"unknown kind: {kind}")


if __name__ == "__main__":
    # Smoke test: python src/data.py
    df = load_data()
    dev, test = split_holdout(df)
    print(f"all rows: {len(df)} | dev: {len(dev)} rows, {dev[GROUP].nunique()} players"
          f" | test: {len(test)} rows, {test[GROUP].nunique()} players")
    assert not set(dev[GROUP]) & set(test[GROUP]), "player in both dev and test!"
    for kind in ["group", "random", "chronological"]:
        sizes = [len(v) for _, v in cv_splits(dev, kind)]
        print(f"{kind:>13} folds, validation sizes: {sizes}")
    print("OK")
