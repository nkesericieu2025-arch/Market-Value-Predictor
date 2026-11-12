"""Reproduce two figures for SCOUT progress report from CV result files.

Neither figure uses the reserved holdout. MAE is in units of target millions;
the underlying currency remains unverified.
Run: python src/plot_results.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data import ROOT

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


def main():
    df = pd.read_csv(ROOT / "results" / "model_comparison.csv")
    chosen = ["Linear Regression", "Ridge (alpha=10)", "Random Forest (unlimited depth)"]
    d = df[df.model.isin(chosen)]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    x = np.arange(len(chosen)); width = .32
    for i, use_start in enumerate((False, True)):
        subset = d[d.use_start == use_start].set_index("model").loc[chosen]
        ax.bar(x + (i-.5)*width, subset.MAE_mean, width,
               label="With start value" if use_start else "Performance only")
    ax.set_xticks(x, ["Linear", "Ridge (α=10)", "Random Forest"], fontsize=10)
    ax.set_ylabel("Five-fold CV MAE (value millions; currency unverified)")
    ax.set_title("Does knowing the season's initial market valuation improve prediction?")
    ax.legend(loc="upper right", frameon=False)
    ax.set_ylim(0, max(d.MAE_mean)*1.18)
    ax.spines[['top', 'right']].set_visible(False)
    fig.text(.5, .02, "Random player-grouped folds on development data; no holdout evaluation.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .05, 1, 1])
    fig.savefig(FIG / "progress_models_mae.png", dpi=190)
    plt.close(fig)

    df = pd.read_csv(ROOT / "results" / "encoding.csv")
    models = list(df.model.unique())
    x = np.arange(len(models)); width = .24
    fig, ax = plt.subplots(figsize=(8.7, 4.9))
    for i, enc in enumerate(("onehot", "grouped", "drop")):
        d = df[df.encoding==enc].set_index("model").loc[models]
        ax.bar(x+(i-1)*width, d.MAE_mean, width, label=enc.capitalize())
    ax.set_xticks(x, ["Ridge", "Random Forest"])
    ax.set_ylabel("Five-fold CV MAE (value millions; currency unverified)")
    ax.set_title("Nationality and club: retain, group rare levels, or omit?")
    ax.set_ylim(0, df.MAE_mean.max()*1.17)
    ax.legend(loc="upper right", frameon=False)
    ax.spines[['top','right']].set_visible(False)
    fig.text(.5, .02, "No start valuation; identical grouped player folds across encodings.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .05, 1, 1])
    fig.savefig(FIG / "progress_encoding_mae.png", dpi=190)
    plt.close(fig)
    print("Created figures/progress_models_mae.png and figures/progress_encoding_mae.png")


if __name__ == '__main__':
    main()
