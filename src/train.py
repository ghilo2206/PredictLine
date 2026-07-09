"""
train.py

End-to-end training and evaluation harness. This is the script that
produces the evidence backing the AI4I proposal's Section 2.2 claim ("Why
AI Is Necessary") — it trains the GRU model, runs the threshold baseline
on the same data, and reports precision/recall/AUC for both, plus a
LEAD-TIME comparison: how many hours before the hard fault does each
approach first raise an alert?

Run:
    python src/train.py
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from data_pipeline import generate_synthetic_dataset, make_windows, normalize
from models.gru_model import SimpleGRUClassifier
from models.threshold_baseline import predict_threshold_windowed, DEFAULT_THRESHOLDS

FEATURE_COLS = ["signal_loss_db", "latency_ms", "packet_loss_pct", "temperature_c"]
WINDOW = 12
SEED = 0


def precision_recall_auc(y_true, y_pred, y_proba=None):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    auc = None
    if y_proba is not None and len(np.unique(y_true)) == 2:
        # simple rank-based AUC (Mann-Whitney U), no sklearn dependency required
        order = np.argsort(y_proba)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(y_proba) + 1)
        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)
        if n_pos > 0 and n_neg > 0:
            sum_ranks_pos = ranks[y_true == 1].sum()
            auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return precision, recall, auc


def lead_time_comparison(df, gru_model, mean, std, window=WINDOW):
    """
    For each segment that has a fault, find the first hour where each
    approach raises an alert, and compare it to the actual fault onset
    (the first hour labelled 1). Reports mean lead time in hours (positive
    = alert fired BEFORE the label window started; this is the headline
    "predictive vs reactive" number for the proposal).
    """
    results = []
    for segment_id, g in df.groupby("segment_id"):
        g = g.sort_values("hour").reset_index(drop=True)
        fault_hour = g["fault_hour"].iloc[0]
        if fault_hour < 0:
            continue  # healthy segment, nothing to measure lead time against

        onset_hour = int(fault_hour)  # the actual hard-fault moment

        values = g[FEATURE_COLS].values
        threshold_alert_hour = None
        gru_alert_hour = None

        for i in range(window, len(g)):
            window_vals = values[i - window:i]

            # threshold baseline: reactive, checks current reading only
            last = window_vals[-1]
            th_fire = any(last[FEATURE_COLS.index(c)] > lim for c, lim in DEFAULT_THRESHOLDS.items())
            if th_fire and threshold_alert_hour is None:
                threshold_alert_hour = i

            # GRU: normalize using TRAIN mean/std, predict
            wn = (window_vals - mean) / std
            proba = gru_model.predict_proba(wn[np.newaxis, :, :])[0]
            if proba > 0.5 and gru_alert_hour is None:
                gru_alert_hour = i

            if threshold_alert_hour is not None and gru_alert_hour is not None:
                break

        results.append({
            "segment_id": segment_id,
            "onset_hour": onset_hour,
            "threshold_alert_hour": threshold_alert_hour,
            "gru_alert_hour": gru_alert_hour,
            "threshold_lead_hours": (onset_hour - threshold_alert_hour) if threshold_alert_hour else None,
            "gru_lead_hours": (onset_hour - gru_alert_hour) if gru_alert_hour else None,
        })
    return results


def main():
    print("=" * 60)
    print("PredictLine — GRU vs Threshold Baseline Evaluation")
    print("(synthetic data — see README for real-data pathway)")
    print("=" * 60)

    df = generate_synthetic_dataset()
    X, y = make_windows(df, window=WINDOW, feature_cols=FEATURE_COLS)
    Xn, mean, std = normalize(X)

    n = len(Xn)
    idx = np.random.default_rng(SEED).permutation(n)
    split = int(n * 0.8)
    train_idx, test_idx = idx[:split], idx[split:]
    Xtr, ytr = Xn[train_idx], y[train_idx]
    Xte, yte = Xn[test_idx], y[test_idx]

    print(f"\nDataset: {n} windowed samples, {y.mean():.3f} positive rate")
    print(f"Train: {len(Xtr)}  Test: {len(Xte)}")

    print("\n--- Training GRU ---")
    model = SimpleGRUClassifier(n_features=Xn.shape[-1], hidden_size=8, seed=SEED)
    model.fit(Xtr, ytr, epochs=25, batch_size=64, lr=0.1, verbose=True)

    gru_proba = model.predict_proba(Xte)
    gru_preds = (gru_proba > 0.5).astype(int)
    gru_p, gru_r, gru_auc = precision_recall_auc(yte, gru_preds, gru_proba)

    th_preds = predict_threshold_windowed(Xte * std + mean, FEATURE_COLS)  # de-normalize for real-unit thresholds
    th_p, th_r, th_auc = precision_recall_auc(yte, th_preds, th_preds.astype(float))

    print("\n--- Held-out Test Set Comparison ---")
    print(f"{'Metric':<12} {'Threshold (current practice)':>30} {'GRU (proposed)':>16}")
    print(f"{'Precision':<12} {th_p:>30.3f} {gru_p:>16.3f}")
    print(f"{'Recall':<12} {th_r:>30.3f} {gru_r:>16.3f}")
    print(f"{'AUC':<12} {'n/a (binary rule)':>30} {gru_auc:>16.3f}")

    print("\n--- Lead-Time Comparison (hours before hard fault) ---")
    lead_results = lead_time_comparison(df, model, mean, std)
    th_leads = [r["threshold_lead_hours"] for r in lead_results if r["threshold_lead_hours"] is not None]
    gru_leads = [r["gru_lead_hours"] for r in lead_results if r["gru_lead_hours"] is not None]
    for r in lead_results:
        print(f"  {r['segment_id']}: onset={r['onset_hour']}h  "
              f"threshold_alert={r['threshold_alert_hour']}  gru_alert={r['gru_alert_hour']}")
    if th_leads:
        print(f"\n  Mean threshold lead time: {np.mean(th_leads):.1f} hours before fault")
    if gru_leads:
        print(f"  Mean GRU lead time:       {np.mean(gru_leads):.1f} hours before fault")

    print("\nNOTE: All numbers above are on SYNTHETIC data (see data_pipeline.py")
    print("and README.md Dataset Provenance section). They demonstrate the")
    print("evaluation methodology, not real-world Powertel performance.")


if __name__ == "__main__":
    main()
