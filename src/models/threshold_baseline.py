"""
threshold_baseline.py

The reactive, rule-based fault detection approach that represents CURRENT
PRACTICE at Powertel/most utility NOCs: fire an alert once a metric crosses
a fixed threshold. This is deliberately simple — it exists so PredictLine's
GRU model has something concrete to be benchmarked against (see AI4I
proposal Section 2.2, "Why AI Is Necessary").

A threshold model only ever reacts to CURRENT readings — it has no memory
of trend, so it cannot fire early during the degradation ramp the way a
sequence model can.
"""

import numpy as np


DEFAULT_THRESHOLDS = {
    "signal_loss_db": 6.0,
    "latency_ms": 25.0,
    "packet_loss_pct": 2.0,
}


def predict_threshold(df, thresholds=None):
    """
    Given a telemetry dataframe (one row per hour per segment), return a
    binary alert array: 1 if ANY monitored metric exceeds its threshold at
    that hour, else 0. No history/trend is considered — purely reactive.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    alert = np.zeros(len(df), dtype=int)
    for col, limit in thresholds.items():
        alert |= (df[col].values > limit).astype(int)
    return alert


def predict_threshold_windowed(X, feature_cols, thresholds=None):
    """
    Same logic, but applied to windowed data (X shape: n_samples, window,
    n_features) to match the GRU's evaluation harness. Only the LAST
    timestep of each window is checked, since a threshold system has no
    concept of "look back over a window" — it only ever sees the present
    reading.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    last_step = X[:, -1, :]  # (n_samples, n_features)
    alert = np.zeros(len(X), dtype=int)
    for col, limit in thresholds.items():
        if col in feature_cols:
            idx = feature_cols.index(col)
            alert |= (last_step[:, idx] > limit).astype(int)
    return alert
