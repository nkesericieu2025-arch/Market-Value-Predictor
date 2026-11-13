"""Integrate Understat EPL xG and xA with SCOUT market values."""

import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import load_data

SOURCE = ROOT / "data/external/understat_players_aggregated_2014_2024.csv"
OUTPUT = ROOT / "data/processed/scout_with_understat.csv"


def normalize_name(value):
    if pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(c for c in value if not unicodedata.combining(c))
    value = re.sub(r"[^a-z0-9 ]", " ", value.lower())
    return " ".join(value.split())


def main():
    scout = load_data().copy()
    understat = pd.read_csv(SOURCE)

    understat = understat.loc[
        (understat["league"] == "EPL")
        & (understat["year"].between(2019, 2023))
    ].copy()

    scout["match_name"] = scout["player_name"].map(normalize_name)
    understat["match_name"] = understat["player_name"].map(normalize_name)
    understat["season"] = understat["year"]

    keys = ["match_name", "season"]

    # Do not assign potentially incorrect xG/xA to ambiguous players.
    ambiguous = understat.duplicated(keys, keep=False)
    print("Ambiguous Understat rows:", int(ambiguous.sum()))

    lookup = understat.loc[
        ~ambiguous, keys + ["xG", "xA"]
    ]

    merged = scout.merge(
        lookup,
        on=keys,
        how="left",
        validate="many_to_one",
    )

    matched = int(merged["xG"].notna().sum())

    print("\nSCOUT observations:", len(scout))
    print("Matched observations:", matched)
    print("Unmatched observations:", len(scout) - matched)
    print(f"Match rate: {matched / len(scout):.1%}")

    print("\nMatch rate by season:")
    for season, group in merged.groupby("season"):
        rate = group["xG"].notna().mean()
        print(f"{season}: {rate:.1%}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    merged.drop(columns="match_name").to_csv(OUTPUT, index=False)
    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
