"""
Tests for the cross-dataset validation harness (Option A: train on one
real dataset, test generalization on the other).
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cross_dataset_eval import precision_recall_auc, align_feature_count, evaluate_cross_dataset


class TestCrossDatasetHarness(unittest.TestCase):
    def test_align_feature_count_truncates(self):
        X = np.zeros((10, 12, 6))
        X_aligned = align_feature_count(X, target_k=4)
        self.assertEqual(X_aligned.shape, (10, 12, 4))

    def test_align_feature_count_pads(self):
        X = np.ones((10, 12, 3))
        X_aligned = align_feature_count(X, target_k=4)
        self.assertEqual(X_aligned.shape, (10, 12, 4))
        self.assertTrue(np.all(X_aligned[..., 3] == 0))  # padded column is zero

    def test_align_feature_count_noop_when_equal(self):
        X = np.ones((10, 12, 4))
        X_aligned = align_feature_count(X, target_k=4)
        self.assertTrue(np.array_equal(X, X_aligned))

    def test_precision_recall_auc_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.8, 0.9])
        p, r, auc = precision_recall_auc(y_true, y_pred, y_proba)
        self.assertEqual(p, 1.0)
        self.assertEqual(r, 1.0)
        self.assertAlmostEqual(auc, 1.0)

    def test_precision_recall_auc_all_wrong(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        p, r, auc = precision_recall_auc(y_true, y_pred)
        self.assertEqual(p, 0.0)
        self.assertEqual(r, 0.0)

    def test_evaluate_cross_dataset_runs_end_to_end(self):
        # Small synthetic run confirming the full train -> cross-test
        # pipeline executes without error and returns sane types.
        rng = np.random.default_rng(0)
        X_a = rng.normal(size=(60, 12, 4))
        y_a = (rng.random(60) < 0.2).astype(int)
        X_b = rng.normal(size=(40, 12, 4))
        y_b = (rng.random(40) < 0.2).astype(int)

        result = evaluate_cross_dataset(X_a, y_a, X_b, y_b, "A", "B", epochs=3)
        self.assertIn("precision", result)
        self.assertIn("recall", result)
        self.assertTrue(0.0 <= result["precision"] <= 1.0)
        self.assertTrue(0.0 <= result["recall"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
