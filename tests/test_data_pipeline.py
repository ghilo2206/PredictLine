"""
Basic test suite. Compatible with both `pytest` and plain `python -m unittest`.
Run with:
    pytest tests/
or:
    python -m unittest discover tests
"""

import sys
import os
import unittest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_pipeline import generate_synthetic_dataset, make_windows, normalize
from models.threshold_baseline import predict_threshold_windowed, DEFAULT_THRESHOLDS
from models.gru_model import SimpleGRUClassifier, sigmoid


class TestDataPipeline(unittest.TestCase):
    def test_generate_synthetic_dataset_shape(self):
        df = generate_synthetic_dataset()
        self.assertGreater(len(df), 0)
        expected_cols = {"segment_id", "hour", "signal_loss_db", "latency_ms",
                          "packet_loss_pct", "temperature_c", "fault_risk_label", "fault_hour"}
        self.assertTrue(expected_cols.issubset(set(df.columns)))

    def test_dataset_has_both_classes(self):
        df = generate_synthetic_dataset()
        labels = df["fault_risk_label"].unique()
        self.assertIn(0, labels)
        self.assertIn(1, labels)

    def test_no_negative_values_in_physical_metrics(self):
        df = generate_synthetic_dataset()
        for col in ["signal_loss_db", "latency_ms", "packet_loss_pct"]:
            self.assertGreaterEqual(df[col].min(), 0)

    def test_make_windows_shape(self):
        df = generate_synthetic_dataset()
        X, y = make_windows(df, window=12)
        self.assertEqual(X.shape[1], 12)
        self.assertEqual(X.shape[2], 4)
        self.assertEqual(len(X), len(y))

    def test_normalize_zero_mean_unit_std(self):
        df = generate_synthetic_dataset()
        X, _ = make_windows(df, window=12)
        Xn, mean, std = normalize(X)
        flat = Xn.reshape(-1, Xn.shape[-1])
        # After normalization, mean should be ~0 and std ~1 per feature
        self.assertTrue(np.allclose(flat.mean(axis=0), 0, atol=1e-6))
        self.assertTrue(np.allclose(flat.std(axis=0), 1, atol=1e-6))


class TestThresholdBaseline(unittest.TestCase):
    def test_predict_threshold_windowed_shape(self):
        df = generate_synthetic_dataset()
        X, y = make_windows(df, window=12)
        feature_cols = ["signal_loss_db", "latency_ms", "packet_loss_pct", "temperature_c"]
        preds = predict_threshold_windowed(X, feature_cols)
        self.assertEqual(len(preds), len(X))
        self.assertTrue(set(np.unique(preds)).issubset({0, 1}))

    def test_thresholds_fire_on_extreme_values(self):
        # Construct an obviously-faulty window and confirm the baseline fires
        feature_cols = ["signal_loss_db", "latency_ms", "packet_loss_pct", "temperature_c"]
        window = np.tile([100.0, 200.0, 50.0, 25.0], (12, 1))  # way above all thresholds
        X = window[np.newaxis, :, :]
        preds = predict_threshold_windowed(X, feature_cols)
        self.assertEqual(preds[0], 1)

    def test_thresholds_do_not_fire_on_healthy_values(self):
        feature_cols = ["signal_loss_db", "latency_ms", "packet_loss_pct", "temperature_c"]
        window = np.tile([1.0, 10.0, 0.05, 24.0], (12, 1))  # healthy baseline values
        X = window[np.newaxis, :, :]
        preds = predict_threshold_windowed(X, feature_cols)
        self.assertEqual(preds[0], 0)


class TestGRUModel(unittest.TestCase):
    def test_sigmoid_bounds(self):
        x = np.array([-100, 0, 100])
        y = sigmoid(x)
        self.assertTrue(np.all(y >= 0) and np.all(y <= 1))
        self.assertAlmostEqual(y[1], 0.5, places=5)

    def test_forward_pass_output_shape(self):
        model = SimpleGRUClassifier(n_features=4, hidden_size=8, seed=0)
        X = np.random.default_rng(0).normal(size=(5, 12, 4))
        y_hat, cache = model.forward(X)
        self.assertEqual(y_hat.shape, (5,))
        self.assertTrue(np.all((y_hat >= 0) & (y_hat <= 1)))

    def test_training_reduces_loss(self):
        # Small deterministic sanity check that backprop actually improves loss
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 6, 3))
        y = (X[:, -1, 0] > 0).astype(float)  # learnable rule based on last timestep
        model = SimpleGRUClassifier(n_features=3, hidden_size=6, seed=0)

        _, cache_before = model.forward(X)
        loss_before = -np.mean(y * np.log(cache_before["y_hat"] + 1e-8) +
                                (1 - y) * np.log(1 - cache_before["y_hat"] + 1e-8))

        model.fit(X, y, epochs=15, batch_size=20, lr=0.1, verbose=False)

        _, cache_after = model.forward(X)
        loss_after = -np.mean(y * np.log(cache_after["y_hat"] + 1e-8) +
                               (1 - y) * np.log(1 - cache_after["y_hat"] + 1e-8))

        self.assertLess(loss_after, loss_before)


if __name__ == "__main__":
    unittest.main()
