"""Basic reproducibility/leakage checks. Run: python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from data import (CATEGORICAL, EXCLUDED, GROUP, START, TARGET,
                  cv_splits, load_data, split_holdout)
from pipeline import build_pipeline, feature_columns
from sklearn.linear_model import Ridge


class DataProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_data()
        cls.dev, cls.holdout = split_holdout(cls.data)

    def test_recover_missingness_without_editing_raw_workbook(self):
        self.assertEqual(len(self.data), 1382)
        self.assertEqual(int(self.data[START].isna().sum()), 10)
        self.assertEqual(int(self.data[TARGET].isna().sum()), 1)
        self.assertFalse(self.data.duplicated([GROUP, "season"]).any())

    def test_disjoint_player_holdout(self):
        self.assertEqual(len(self.dev) + len(self.holdout), len(self.data)-1)
        self.assertFalse(self.dev[TARGET].isna().any())
        self.assertFalse(self.holdout[TARGET].isna().any())
        self.assertFalse(set(self.dev[GROUP]) & set(self.holdout[GROUP]))
        self.assertEqual(len(self.holdout[GROUP].unique()), 132)

    def test_primary_folds_are_random_across_seasons_and_group_disjoint(self):
        folds = cv_splits(self.dev, kind="group")
        self.assertEqual(len(folds), 5)
        covered = np.concatenate([v for _, v in folds])
        np.testing.assert_array_equal(np.sort(covered), np.arange(len(self.dev)))
        for train, validation in folds:
            self.assertFalse(set(self.dev.iloc[train][GROUP]) & set(self.dev.iloc[validation][GROUP]))
            self.assertGreater(self.dev.iloc[validation]["season"].nunique(), 1)

    def test_feature_sets_exclude_target_and_leakage(self):
        for encoding in ('grouped', 'onehot', 'drop'):
            for use_start in (False, True):
                num, cat = feature_columns(encoding, use_start)
                self.assertFalse(set(num+cat) & set(EXCLUDED + [TARGET]))
                self.assertEqual(START in num, use_start)

    def test_preprocessor_handles_missing_values_inside_training_fold(self):
        train, val = cv_splits(self.dev, kind='group')[0]
        num, cat = feature_columns(use_start=True)
        pipe = build_pipeline(Ridge(alpha=10), use_start=True)
        pipe.fit(self.dev.iloc[train][num+cat], self.dev.iloc[train][TARGET])
        pred = pipe.predict(self.dev.iloc[val][num+cat])
        self.assertTrue(np.isfinite(pred).all())


if __name__ == '__main__':
    unittest.main()
