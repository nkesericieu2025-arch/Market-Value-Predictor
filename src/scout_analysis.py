"""SCOUT experiment entry point: RANDOM, player-grouped cross-validation.

The earlier analysis selected model(s) by 2022 and evaluated 2023. That code
is kept *unchanged* in src/archive/scout_analysis_legacy_chronological.py for
transparency, but must not be confused with the primary current pipeline.

Run the independent development-only scripts:
   python src/example_baselines.py
   python src/model_comparison.py
   python src/exp_encoding.py
   python src/exp_splits.py
All results use the same group-aware development folds and exclude the group
holdout. Existing results/scout and figures/scout are HISTORICAL outputs.
"""
from example_baselines import main as baselines_main
from model_comparison import main as models_main


def main():
    print("SCOUT: Primary randomized player-grouped validation (not by season)")
    baselines_main()
    models_main()
    print("For professor-feedback diagnostics, run exp_encoding.py and exp_splits.py")


if __name__ == '__main__':
    main()
