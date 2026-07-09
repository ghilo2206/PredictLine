"""
telecomts_adapter.py

Adapter for AliMaatouk/TelecomTS (HuggingFace), a real 5G telecom
observability dataset. Converts its chunked JSONL format into the same
(X, y) tensor shape used everywhere else in this repo: X of shape
(n_samples, window, n_features), y of shape (n_samples,).

DATASET SCHEMA (per HuggingFace dataset card, as of this writing):
Each JSONL record is a pre-chunked 128-timestep sample with keys:
    start_time, end_time, sampling_rate, KPIs, description,
    anomalies, statistics, labels, QnA

`KPIs` is expected to be a dict mapping KPI name -> list of values
(one per timestep in the chunk). `labels` / `anomalies` indicate whether
the chunk contains an anomaly.

HOW TO GET THE REAL FILE:
    pip install datasets
    python -c "
from datasets import load_dataset
ds = load_dataset('AliMaatouk/TelecomTS', data_files={'full': '**/chunked.jsonl'})['full']
ds.to_json('data/telecomts_raw.jsonl')
"

TODO (do this once you have the real file):
    Run `inspect_telecomts_schema()` below on a real downloaded file and
    confirm (a) the exact KPI names available, and (b) the exact format
    of the `labels` field (binary flag? list per-timestep? string?).
    Then set TELECOMTS_KPI_SELECTION below to the KPI names you want to
    use as your feature columns (pick however many you like; just note
    the count for the common-feature alignment step in
    cross_dataset_eval.py).
"""

import json
import numpy as np


# TODO: replace with real KPI names once you've inspected an actual file.
# These are placeholders illustrating the *shape* of the selection, not
# verified real KPI names from the dataset.
TELECOMTS_KPI_SELECTION = ["kpi_1", "kpi_2", "kpi_3", "kpi_4"]


def inspect_telecomts_schema(jsonl_path, n_samples=3):
    """
    Print the keys and KPI names found in the first few records of a real
    TelecomTS JSONL file. Run this FIRST after downloading the real data,
    before trusting anything else in this file.
    """
    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            record = json.loads(line)
            print(f"--- Record {i} ---")
            print("Top-level keys:", list(record.keys()))
            if "KPIs" in record and isinstance(record["KPIs"], dict):
                print("KPI names:", list(record["KPIs"].keys()))
            print("labels:", record.get("labels"))
            print("anomalies:", record.get("anomalies"))
            print()


def _label_from_record(record) -> int:
    """
    Best-effort binary label extraction. TelecomTS's exact label schema
    should be confirmed via inspect_telecomts_schema() before trusting
    this — this function handles a few plausible formats defensively so
    it doesn't silently mislabel everything if the real schema differs
    from what's assumed here.
    """
    anomalies = record.get("anomalies")
    labels = record.get("labels")

    if isinstance(anomalies, (list, dict)) and len(anomalies) > 0:
        return 1
    if isinstance(labels, (int, float)):
        return int(labels != 0)
    if isinstance(labels, str):
        return int(labels.lower() not in ("normal", "none", ""))
    if isinstance(labels, list) and len(labels) > 0:
        # if per-timestep labels, treat "any positive in the chunk" as positive
        try:
            return int(any(float(x) != 0 for x in labels))
        except (TypeError, ValueError):
            return 0
    return 0


def load_telecomts_windows(jsonl_path, kpi_selection=None, max_records=None):
    """
    Parse a real (downloaded) TelecomTS JSONL file into (X, y, feature_names).

    X: (n_samples, window, n_features) — window length = length of each
       chunk's KPI value lists (128 per the dataset card, but we read the
       actual length from the data rather than hardcoding it).
    y: (n_samples,) binary labels.
    """
    if kpi_selection is None:
        kpi_selection = TELECOMTS_KPI_SELECTION

    X_list, y_list = [], []
    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f):
            if max_records is not None and i >= max_records:
                break
            record = json.loads(line)
            kpis = record.get("KPIs", {})

            missing = [k for k in kpi_selection if k not in kpis]
            if missing:
                # Skip records missing the selected KPIs rather than
                # crashing — but this should be rare/zero once
                # kpi_selection is set correctly from a real inspection.
                continue

            try:
                cols = [np.asarray(kpis[k], dtype=float) for k in kpi_selection]
                window = np.stack(cols, axis=-1)  # (timesteps, n_features)
            except (ValueError, TypeError):
                continue

            X_list.append(window)
            y_list.append(_label_from_record(record))

    if not X_list:
        raise ValueError(
            "No usable records found. Run inspect_telecomts_schema() first "
            "and update TELECOMTS_KPI_SELECTION to match real KPI names."
        )

    # Chunks should already be equal length (128), but guard against
    # ragged data by trimming to the shortest observed length.
    min_len = min(w.shape[0] for w in X_list)
    X = np.stack([w[:min_len] for w in X_list])
    y = np.array(y_list)
    return X, y, kpi_selection
