"""
Unit tests for the real-data Cisco adapter and training harness.

These deliberately do NOT depend on the real cisco-ie/telemetry clone
(multiple GB, not part of this repository) — they test the pure logic
(binning/labeling math, grouped splitting, thresholding) against small
synthetic arrays shaped like the real data, so the suite stays fast and
self-contained. End-to-end parsing against the real dataset was verified
manually (see README.md and cisco_adapter.py's module docstring) and is
not re-asserted here.
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from adapters.cisco_adapter import _label_bins
from train_real_cisco import (
    grouped_split, undersample_negatives, predict_reactive_threshold, best_f1_threshold,
)


class TestLabelBins(unittest.TestCase):
    def test_marks_event_window_and_lead_in(self):
        bins = list(range(0, 20))  # bin_seconds=10 -> covers t=0..199s
        events = [("leaf1", 100.0, 120.0)]
        labels = _label_bins(bins, bin_seconds=10, device="leaf1", events=events, lead_seconds=15)
        # bin 8 -> t=80 (before lead-in) should be 0; bin 9 -> t=90 (within lead-in) should be 1
        self.assertEqual(labels[8], 0)
        self.assertEqual(labels[9], 1)
        # bin 12 -> t=120 (event end) should be 1; bin 13 -> t=130 (after event) should be 0
        self.assertEqual(labels[12], 1)
        self.assertEqual(labels[13], 0)

    def test_ignores_events_on_other_devices(self):
        bins = list(range(0, 5))
        events = [("spine1", 0.0, 40.0)]
        labels = _label_bins(bins, bin_seconds=10, device="leaf1", events=events, lead_seconds=5)
        self.assertTrue((labels == 0).all())


class TestGroupedSplit(unittest.TestCase):
    def test_no_group_appears_on_both_sides(self):
        rng = np.random.default_rng(0)
        groups = np.array([f"case::{i % 10}" for i in range(500)])
        y = (rng.random(500) < 0.1).astype(int)
        train_mask, test_mask = grouped_split(groups, y, test_fraction=0.3, seed=1)

        train_groups = set(groups[train_mask])
        test_groups = set(groups[test_mask])
        self.assertEqual(train_groups & test_groups, set())
        self.assertTrue((train_mask | test_mask).all())

    def test_positive_groups_appear_in_both_splits_when_enough_exist(self):
        groups = np.array([f"g{i}" for i in range(20) for _ in range(5)])
        y = np.zeros(100, dtype=int)
        # Make every group have at least one positive window
        for i in range(20):
            y[i * 5] = 1
        train_mask, test_mask = grouped_split(groups, y, test_fraction=0.25, seed=0)
        self.assertGreater(y[train_mask].sum(), 0)
        self.assertGreater(y[test_mask].sum(), 0)


class TestUndersampleNegatives(unittest.TestCase):
    def test_keeps_all_positives_and_caps_negatives(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(1000, 4, 2))
        y = np.zeros(1000, dtype=int)
        y[:10] = 1  # 10 positives, 990 negatives
        Xs, ys = undersample_negatives(X, y, ratio=5)
        self.assertEqual(ys.sum(), 10)
        self.assertEqual(len(ys), 10 + 10 * 5)


class TestReactiveThreshold(unittest.TestCase):
    def test_fires_when_established_below_total(self):
        feature_cols = ["global__established-neighbors-count-total", "global__neighbors-count-total"]
        window = np.array([[38, 38]] * 11 + [[0, 38]])  # last reading: all sessions down
        X = window[np.newaxis, :, :]
        preds = predict_reactive_threshold(X, feature_cols)
        self.assertEqual(preds[0], 1)

    def test_silent_when_healthy(self):
        feature_cols = ["global__established-neighbors-count-total", "global__neighbors-count-total"]
        window = np.tile([38, 38], (12, 1))
        X = window[np.newaxis, :, :]
        preds = predict_reactive_threshold(X, feature_cols)
        self.assertEqual(preds[0], 0)


class TestBestF1Threshold(unittest.TestCase):
    def test_finds_perfect_separator(self):
        y = np.array([0, 0, 0, 1, 1])
        proba = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        t, f1 = best_f1_threshold(y, proba)
        self.assertAlmostEqual(f1, 1.0)
        self.assertTrue(0.3 < t <= 0.8)


if __name__ == "__main__":
    unittest.main()
