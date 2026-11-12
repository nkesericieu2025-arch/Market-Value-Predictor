"""Shared preprocessing + evaluation for SCOUT.

build_pipeline(model, encoding=..., use_start=...) returns a scikit-learn
Pipeline. Because imputation, scaling and encoding live INSIDE the pipeline,
they are re-fitted on the training part of every fold (no leakage).

encoding options (professor's sparsity concern):
  "onehot"  one column per nationality / club (sparse)
  "grouped" categories with < MIN_FREQ training rows merged into "infrequent"
  "drop"    nationality and club_name removed; position kept
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data import CATEGORICAL, NUMERIC, START, TARGET

MIN_FREQ = 10
SCORING = {"MAE": "neg_mean_absolute_error",
           "RMSE": "neg_root_mean_squared_error",
           "R2": "r2"}


def feature_columns(encoding="grouped", use_start=False):
    num = NUMERIC + ([START] if use_start else [])
    cat = ["position"] if encoding == "drop" else CATEGORICAL
    return num, cat


def build_pipeline(model, encoding="grouped", use_start=False):
    num, cat = feature_columns(encoding, use_start)
    if encoding == "grouped":
        enc = OneHotEncoder(min_frequency=MIN_FREQ,
                            handle_unknown="infrequent_if_exist")
    else:
        enc = OneHotEncoder(handle_unknown="ignore")
    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num),
        ("cat", enc, cat),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def evaluate(pipe, dev, splits, encoding="grouped", use_start=False):
    """Cross-validated MAE / RMSE / R2 (mean and std across folds)."""
    num, cat = feature_columns(encoding, use_start)
    res = cross_validate(pipe, dev[num + cat], dev[TARGET], cv=splits,
                         scoring=SCORING)
    out = {}
    for name in SCORING:
        vals = res[f"test_{name}"] * (-1 if name != "R2" else 1)
        out[f"{name}_mean"], out[f"{name}_std"] = vals.mean(), vals.std()
    return out


def results_table(rows):
    """List of dicts -> tidy DataFrame rounded for the report."""
    return pd.DataFrame(rows).round(3)
