"""
cisco_adapter.py

Adapter for cisco-ie/telemetry (GitHub), a real network telemetry dataset
released alongside published anomaly-detection research (BGP anomalies,
port flaps, transceiver events, etc). Converts it into the same (X, y)
tensor shape used everywhere else in this repo.

DATASET LAYOUT (per repo structure, as of this writing):
    telemetry/
      <case_number>/           one folder per labeled scenario
        <case>.csv              the actual telemetry readings
        <case>_headers...       column/header definition file
        <case>_info / case file describing WHEN the anomaly occurred

TODO (do this once you've cloned the real repo):
    1. Run `inspect_cisco_case()` on one real case folder and confirm the
       actual column names in the header definition file and the CSV.
    2. Set CISCO_FEATURE_COLUMNS below to the real numeric column names
       you want to use as features (pick the same COUNT of features you
       chose for TelecomTS, so the two datasets can share one model — see
       cross_dataset_eval.py).
    3. Confirm how the case/info file expresses "when did the anomaly
       start" so `_label_from_case_info()` below can be corrected —
       the parsing logic here is a reasonable guess at a common format,
       not verified against the real files.

HOW TO GET THE REAL DATA:
    git clone https://github.com/cisco-ie/telemetry.git
"""

import os
import glob
import numpy as np
import pandas as pd


# TODO: replace with real column names once you've inspected a real case.
CISCO_FEATURE_COLUMNS = ["metric_1", "metric_2", "metric_3", "metric_4"]


def inspect_cisco_case(case_dir):
    """
    Print the files found in a Cisco telemetry case folder and preview
    any CSV's columns. Run this FIRST after cloning the real repo.
    """
    print(f"Contents of {case_dir}:")
    for f in sorted(os.listdir(case_dir)):
        print(" -", f)

    csvs = glob.glob(os.path.join(case_dir, "*.csv"))
    for csv_path in csvs:
        df = pd.read_csv(csv_path, nrows=5)
        print(f"\nPreview of {os.path.basename(csv_path)}:")
        print("Columns:", list(df.columns))
        print(df.head())


def _label_from_case_info(case_dir, timestamps):
    """
    Best-effort: look for a small text/info file in the case folder
    describing the anomaly window, and label timestamps that fall
    inside it as 1. Falls back to labeling the LAST portion of the case
    as the anomaly (a common convention in fault-injection datasets)
    if no parseable info file is found — this fallback should be
    replaced once the real info-file format is confirmed via
    inspect_cisco_case().
    """
    info_candidates = glob.glob(os.path.join(case_dir, "*info*")) + \
        glob.glob(os.path.join(case_dir, "*case*.txt"))

    # TODO: parse real start/end anomaly timestamps from the info file
    # once its format is confirmed. Placeholder fallback below.
    n = len(timestamps)
    labels = np.zeros(n, dtype=int)
    if n > 10:
        labels[int(n * 0.7):] = 1  # naive fallback: last 30% = anomaly window
    return labels


def make_windows_from_df(df, feature_cols, window=12):
    """Same sliding-window logic as data_pipeline.make_windows, applied
    to a single case's dataframe (no segment grouping needed here since
    a case file is already one continuous series)."""
    values = df[feature_cols].values
    labels = df["_label"].values
    X, y = [], []
    for i in range(window, len(df)):
        X.append(values[i - window:i])
        y.append(labels[i])
    return np.array(X), np.array(y)


def load_cisco_windows(case_dir, feature_cols=None, window=12):
    """
    Parse one real Cisco telemetry case folder into (X, y, feature_names).
    """
    if feature_cols is None:
        feature_cols = CISCO_FEATURE_COLUMNS

    csvs = glob.glob(os.path.join(case_dir, "*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {case_dir}")

    df = pd.read_csv(csvs[0])
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columns {missing} not found in {csvs[0]}. Run inspect_cisco_case() "
            f"and update CISCO_FEATURE_COLUMNS to match real column names."
        )

    df["_label"] = _label_from_case_info(case_dir, df.index.values)
    X, y = make_windows_from_df(df, feature_cols, window=window)
    return X, y, feature_cols


def load_all_cisco_cases(telemetry_repo_root, feature_cols=None, window=12):
    """
    Load and concatenate windows across ALL case folders in the cloned
    cisco-ie/telemetry repo (folders that are plain numbers, per the
    documented layout).
    """
    case_dirs = [
        os.path.join(telemetry_repo_root, d)
        for d in os.listdir(telemetry_repo_root)
        if d.isdigit() and os.path.isdir(os.path.join(telemetry_repo_root, d))
    ]
    if not case_dirs:
        raise FileNotFoundError(
            f"No numbered case folders found under {telemetry_repo_root}. "
            f"Confirm you cloned the repo and passed the right root path."
        )

    X_list, y_list = [], []
    for case_dir in sorted(case_dirs):
        try:
            X, y, cols = load_cisco_windows(case_dir, feature_cols, window)
            X_list.append(X)
            y_list.append(y)
        except (FileNotFoundError, ValueError) as e:
            print(f"Skipping {case_dir}: {e}")

    if not X_list:
        raise ValueError("No case folders could be loaded successfully.")

    return np.concatenate(X_list), np.concatenate(y_list), feature_cols or CISCO_FEATURE_COLUMNS
