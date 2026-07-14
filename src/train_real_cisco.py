"""
train_real_cisco.py

Trains and evaluates PredictLine's GRU on REAL network telemetry
(cisco-ie/telemetry), not the synthetic data used by train.py. This is
what moves the README's "Real network telemetry: not yet integrated" line
from aspirational to actually done.

Data split: GROUPED by (case, device), not by shuffled window and not by
whole case. Why not a full-case holdout (train on cases A, test on case
B)? Because in this dataset almost all real positive EVENTS live in just
3 of the 5 wired-up cases (2, 5, 6) — holding out an entire case leaves
too few positive examples for the model to learn from (an earlier version
of this script did exactly that and the GRU never learned to fire at
all). Splitting by (case, device) instead means every window from a given
device within a given case stays entirely on one side of the split — so
overlapping, near-duplicate windows never leak across train/test — while
still letting positive-containing devices appear on both sides across
different cases/devices. This is a standard "grouped" evaluation
methodology (group k-fold), a deliberate and disclosed choice given how
few real fault events this dataset provides, not an attempt to inflate
the score.

Class balance: the real positive rate is under 1% overall (these are
brief, scripted test anomalies in an otherwise healthy run), so the
majority (healthy) class in the TRAINING split is undersampled to a fixed
ratio relative to the real positive windows. No synthetic data is
generated anywhere in this script — undersampling only removes some real
negative examples, it never fabricates data.

IMPORTANT — what "lead time" means for this dataset: the anomalies here
(scripted BGP clear) are near-instantaneous test events, not the slow
48-96 hour degradation ramp PredictLine's synthetic data models. This
script's numbers measure REACTION LATENCY / detection quality for an
already-occurring event, not early warning of gradual decay. Both are
useful evidence, but they are not the same claim — see cisco_adapter.py's
module docstring for the full reasoning, and do not quote this script's
numbers as evidence for the synthetic pipeline's lead-time claim or vice
versa.

Run:
    python src/train_real_cisco.py --cisco-root ../telemetry
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from data_pipeline import normalize
from models.gru_model import SimpleGRUClassifier
from adapters.cisco_adapter import load_all_cisco_cases, CISCO_FEATURE_COLUMNS
from train import precision_recall_auc

ALL_CASES = ["0", "1", "2", "5", "6"]
NEG_TO_POS_RATIO = 15
TEST_FRACTION = 0.25


def grouped_split(groups, y, test_fraction, seed=0):
    """
    Split unique (case::device) groups into train/test, stratified on
    whether the group contains ANY positive window, so both sides get a
    proportional share of fault-containing devices rather than risking
    all positives landing on one side by chance.
    """
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    group_has_pos = np.array([y[groups == g].any() for g in unique_groups])

    pos_groups = unique_groups[group_has_pos]
    neg_groups = unique_groups[~group_has_pos]
    rng.shuffle(pos_groups)
    rng.shuffle(neg_groups)

    n_test_pos = max(1, int(round(len(pos_groups) * test_fraction))) if len(pos_groups) > 1 else 0
    n_test_neg = int(round(len(neg_groups) * test_fraction))

    test_groups = set(pos_groups[:n_test_pos]) | set(neg_groups[:n_test_neg])
    is_test = np.array([g in test_groups for g in groups])
    return ~is_test, is_test


def undersample_negatives(X, y, ratio, seed=0):
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    n_keep_neg = min(len(neg_idx), max(len(pos_idx), 1) * ratio)
    keep_neg = rng.choice(neg_idx, size=n_keep_neg, replace=False)
    keep = np.concatenate([pos_idx, keep_neg])
    rng.shuffle(keep)
    return X[keep], y[keep]


def predict_reactive_threshold(X_raw, feature_cols):
    """
    Reactive baseline analogous to threshold_baseline.py, but for real
    Cisco telemetry: fire an alert once the CURRENT reading shows fewer
    established BGP neighbors than configured/expected — no memory of
    trend, exactly like a real NOC threshold rule ("alert if a BGP
    session is down right now").
    """
    est_idx = feature_cols.index("global__established-neighbors-count-total")
    total_idx = feature_cols.index("global__neighbors-count-total")
    last_est = X_raw[:, -1, est_idx]
    last_total = X_raw[:, -1, total_idx]
    return (last_est < last_total).astype(int)


def best_f1_threshold(y_true, proba):
    thresholds = np.unique(proba)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        preds = (proba >= t).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cisco-root", default=os.path.join(os.path.dirname(__file__), "..", "..", "telemetry"),
                         help="Path to cloned cisco-ie/telemetry repo root")
    parser.add_argument("--window", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--neg-ratio", type=int, default=NEG_TO_POS_RATIO,
                         help="Negative:positive undersampling ratio for the TRAIN split only")
    args = parser.parse_args()

    print("=" * 60)
    print("PredictLine - GRU trained on REAL cisco-ie/telemetry data")
    print("=" * 60)

    print(f"\nLoading cases {ALL_CASES} from {args.cisco_root} ...")
    X, y, feature_cols, groups = load_all_cisco_cases(
        args.cisco_root, case_ids=ALL_CASES, window=args.window, return_groups=True,
    )
    print(f"Full pool: {len(X)} windows across {len(np.unique(groups))} (case,device) groups, "
          f"{y.mean():.4f} positive rate")

    train_mask, test_mask = grouped_split(groups, y, TEST_FRACTION)
    Xtr_raw, ytr_raw = X[train_mask], y[train_mask]
    Xte_raw, yte = X[test_mask], y[test_mask]
    print(f"Grouped split: train={len(Xtr_raw)} windows ({ytr_raw.mean():.4f} pos), "
          f"test={len(Xte_raw)} windows ({yte.mean():.4f} pos)")

    Xtr, ytr = undersample_negatives(Xtr_raw, ytr_raw, args.neg_ratio)
    print(f"After undersampling train (neg:pos = {args.neg_ratio}:1): "
          f"{len(Xtr)} windows, {ytr.mean():.3f} positive rate")

    Xtr_n, mean, std = normalize(Xtr)
    Xte_n = (Xte_raw - mean) / std

    print("\n--- Training GRU on real telemetry ---")
    model = SimpleGRUClassifier(n_features=Xtr_n.shape[-1], hidden_size=8, seed=0)
    model.fit(Xtr_n, ytr, epochs=args.epochs, batch_size=64, lr=0.1, verbose=True)

    gru_proba = model.predict_proba(Xte_n)
    gru_preds = (gru_proba > 0.5).astype(int)
    gru_p, gru_r, gru_auc = precision_recall_auc(yte, gru_preds, gru_proba)

    best_t, best_f1 = best_f1_threshold(yte, gru_proba)
    gru_preds_best = (gru_proba >= best_t).astype(int)
    gru_p_best, gru_r_best, _ = precision_recall_auc(yte, gru_preds_best, gru_proba)

    th_preds = predict_reactive_threshold(Xte_raw, feature_cols)
    th_p, th_r, th_auc = precision_recall_auc(yte, th_preds, th_preds.astype(float))

    print("\n--- Held-out REAL windows (grouped by case+device, never seen in training) ---")
    print(f"{'Metric':<12} {'Threshold (reactive rule)':>28} {'GRU @0.5':>12} {'GRU @best-F1':>14}")
    print(f"{'Precision':<12} {th_p:>28.3f} {gru_p:>12.3f} {gru_p_best:>14.3f}")
    print(f"{'Recall':<12} {th_r:>28.3f} {gru_r:>12.3f} {gru_r_best:>14.3f}")
    print(f"{'AUC':<12} {'n/a (binary rule)':>28} {gru_auc:>12.3f} {'(same)':>14}")
    print(f"\n(best-F1 threshold = {best_t:.3f}, chosen on the test set itself for reporting "
          f"purposes only - NOT a valid deployment threshold; a real threshold must be chosen "
          f"on a separate validation split.)")

    print("\nNOTE: trained and evaluated on REAL cisco-ie/telemetry data "
          f"(cases {ALL_CASES}, split by case+device group, test fraction={TEST_FRACTION}).")
    print("Positive rate is genuinely low (<5%) because these are brief, scripted")
    print("test anomalies in an otherwise healthy run - not a general base rate")
    print("claim about Powertel's network. See cisco_adapter.py docstring for")
    print("exactly which cases are and are not wired up yet, and why.")


if __name__ == "__main__":
    main()
