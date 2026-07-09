"""
cross_dataset_eval.py

Option A: cross-dataset validation. Instead of merging TelecomTS and
Cisco's telemetry into one blended pool, we train on ONE real dataset and
test generalization on the OTHER, in both directions. This is stronger
evidence than a single blended dataset: it shows the model isn't just
overfitting to one network's quirks.

Requires both adapters' TODOs to be filled in first (real KPI/column
names, confirmed label logic) — see src/adapters/telecomts_adapter.py
and src/adapters/cisco_adapter.py.

Run:
    python src/cross_dataset_eval.py \
        --telecomts data/telecomts_raw.jsonl \
        --cisco-root /path/to/cloned/cisco-ie-telemetry
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from models.gru_model import SimpleGRUClassifier
from data_pipeline import normalize


def precision_recall_auc(y_true, y_pred, y_proba=None):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    auc = None
    if y_proba is not None and len(np.unique(y_true)) == 2:
        order = np.argsort(y_proba)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(y_proba) + 1)
        n_pos, n_neg = np.sum(y_true == 1), np.sum(y_true == 0)
        if n_pos > 0 and n_neg > 0:
            auc = (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return precision, recall, auc


def align_feature_count(X, target_k):
    """
    If two datasets end up with a different number of selected features,
    truncate or zero-pad to a common width so one model can be applied
    to both. Prefer fixing this upstream (pick the same number of
    columns in both adapters) — this is a safety net, not the primary
    strategy.
    """
    k = X.shape[-1]
    if k == target_k:
        return X
    if k > target_k:
        print(f"  [align] truncating {k} -> {target_k} features")
        return X[..., :target_k]
    print(f"  [align] zero-padding {k} -> {target_k} features")
    pad = np.zeros(X.shape[:-1] + (target_k - k,))
    return np.concatenate([X, pad], axis=-1)


def evaluate_cross_dataset(X_train, y_train, X_test, y_test, train_name, test_name,
                            hidden_size=8, epochs=25, lr=0.1, seed=0):
    print(f"\n=== Train on {train_name} -> Test on {test_name} ===")
    print(f"Train: {len(X_train)} samples ({y_train.mean():.3f} positive rate)")
    print(f"Test:  {len(X_test)} samples ({y_test.mean():.3f} positive rate)")

    Xtr_n, mean, std = normalize(X_train)
    Xte_n = (X_test - mean) / std  # use TRAIN normalization stats on test set — this matters

    model = SimpleGRUClassifier(n_features=Xtr_n.shape[-1], hidden_size=hidden_size, seed=seed)
    model.fit(Xtr_n, y_train, epochs=epochs, batch_size=64, lr=lr, verbose=False)

    proba = model.predict_proba(Xte_n)
    preds = (proba > 0.5).astype(int)
    p, r, auc = precision_recall_auc(y_test, preds, proba)

    print(f"  Precision: {p:.3f}   Recall: {r:.3f}   AUC: {auc if auc is None else round(auc, 3)}")
    return {"train": train_name, "test": test_name, "precision": p, "recall": r, "auc": auc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--telecomts", help="Path to downloaded TelecomTS JSONL file")
    parser.add_argument("--cisco-root", help="Path to cloned cisco-ie/telemetry repo root")
    parser.add_argument("--dry-run", action="store_true",
                         help="Use mock data shaped like each schema, to test the harness "
                              "itself without needing the real files yet.")
    args = parser.parse_args()

    if args.dry_run or not (args.telecomts and args.cisco_root):
        print("Running in DRY-RUN mode (mock data shaped like each dataset's schema).")
        print("This proves the harness works — it is NOT a real result. Pass")
        print("--telecomts and --cisco-root with real paths for real numbers.\n")
        rng = np.random.default_rng(0)
        # Mock TelecomTS-shaped data: (n, 128, 4)
        X_tts = rng.normal(size=(400, 128, 4))
        y_tts = (rng.random(400) < 0.15).astype(int)
        # Mock Cisco-shaped data: (n, 12, 4)
        X_cisco = rng.normal(size=(600, 12, 4))
        y_cisco = (rng.random(600) < 0.10).astype(int)
    else:
        from adapters.telecomts_adapter import load_telecomts_windows
        from adapters.cisco_adapter import load_all_cisco_cases

        X_tts, y_tts, tts_cols = load_telecomts_windows(args.telecomts)
        X_cisco, y_cisco, cisco_cols = load_all_cisco_cases(args.cisco_root)
        print(f"TelecomTS features used: {tts_cols}")
        print(f"Cisco features used: {cisco_cols}")

        target_k = min(X_tts.shape[-1], X_cisco.shape[-1])
        X_tts = align_feature_count(X_tts, target_k)
        X_cisco = align_feature_count(X_cisco, target_k)

    results = []
    results.append(evaluate_cross_dataset(X_cisco, y_cisco, X_tts, y_tts, "Cisco", "TelecomTS"))
    results.append(evaluate_cross_dataset(X_tts, y_tts, X_cisco, y_cisco, "TelecomTS", "Cisco"))

    print("\n=== Summary ===")
    for r in results:
        auc_str = f"{r['auc']:.3f}" if r["auc"] is not None else "n/a"
        print(f"  {r['train']:>10} -> {r['test']:<10}  precision={r['precision']:.3f}  "
              f"recall={r['recall']:.3f}  auc={auc_str}")


if __name__ == "__main__":
    main()
