"""
cisco_adapter.py

Adapter for cisco-ie/telemetry (GitHub) — REAL network telemetry released
alongside published anomaly-detection research (BGP clear events, port
flaps, transceiver pulls, admin-shut events). Converts it into the same
(X, y) tensor shape used everywhere else in this repo: X of shape
(n_samples, window, n_features), y of shape (n_samples,).

THIS FILE WAS WRITTEN AGAINST THE REAL, DOWNLOADED DATASET
(`git clone https://github.com/cisco-ie/telemetry.git`), and the feature
choice below was VALIDATED against a known labelled event before being
relied on (see "why these features" below) — not assumed from the header
list alone.

WHY THESE FEATURES (read this before changing them):
    Each case's CSV is a SPARSE, LONG-FORMAT telemetry log: one row per
    sensor-path publish event. The `name` column identifies which YANG
    sensor path the row reports; each path populates a different subset
    of the ~87 columns. A first version of this adapter used the
    densely-populated interface `data-rate` path (reliability, data-rate,
    load, packet-rate) — but checking it against a known ground-truth BGP
    clear event (case 2, device leaf1, 1498754414-1498754597) showed NO
    visible change in any of those columns during the labelled window.
    That makes sense in hindsight: a BGP session clear is a control-plane
    event, not a physical-layer one, so interface data-rate/reliability
    counters don't move.

    Checking the SAME event against
    `Cisco-IOS-XR-ipv4-bgp-oper:bgp/instances/instance/instance-active/default-vrf/process-info`
    instead showed `global__established-neighbors-count-total` drop from
    a steady 38 to 0 and recover over ~15-20 seconds, exactly inside the
    labelled event window. That is the real, verified fault signal this
    adapter now uses:

    BGP_FEATURE_COLUMNS (from the `process-info` path, per device):
        global__established-neighbors-count-total  -- drops when sessions clear (the direct signal)
        global__neighbors-count-total               -- configured/expected neighbor count (the baseline to compare against)
    COUNTER_FEATURE_COLUMNS (from the `generic-counters` path, per
    interface, summed across all interfaces on a device):
        input-errors, output-errors, carrier-transitions, crc-errors
        -- included as supplementary signal for physical-layer cases
        (port flap, transceiver pull) in later cases; largely flat for
        pure BGP-clear events, which is expected and fine — the GRU can
        learn to weight them down for this event type.

    Segments are DEVICE-level (Producer), not (device, interface) as an
    earlier version had it — BGP session state and the ground-truth
    labels are both per-device, and process-info is reported far less
    often (roughly once per ~10s per device) than interface counters
    (roughly once per few seconds per interface), so device-level binning
    is both the correct grain and necessary to align the two streams.

GROUND TRUTH by case (unchanged from earlier investigation):
        case 2  -> bgpclear_ground_truth.txt   (CSV: Node,Host,Start,End,Event,Type; epoch SECONDS)
        case 5  -> bgpclear_apptraffic_event.log   (JSON list; start_time/end_time epoch MILLISECONDS; host_key)
        case 6  -> bgpclear_no_traffic_event.log   (same JSON shape as case 5)
        case 0, 1 -> no anomaly file at all: labelled baselines (all-healthy).

WHAT IS DELIBERATELY NOT WIRED UP YET (stated plainly, not hidden):
    case 3, 4   -> ground truth is a hand-formatted pipe-table
                   (`*_casedata.txt`) with wall-clock Pacific Daylight
                   Time timestamps, needing PDT->UTC conversion and fuzzy
                   table parsing. Also, these are port-flap/transceiver
                   events, so the `process-info` BGP signal used here
                   would likely need the interface counters to be the
                   PRIMARY signal instead (unverified — not checked yet).
    case 7      -> `bgp_clear_72h.zip`, a large multi-day run; not yet
                   inspected for internal layout.
    case 9, 10  -> ground truth is only provided as a PNG image
                   (`*_event_key.png`), not a machine-parseable file.
    case 11, 12 -> nested `Dataset/` / per-scenario subfolders with their
                   own internal structure, not yet inspected.
    Cross-dataset validation against TelecomTS (see cross_dataset_eval.py)
    still uses mock data for the TelecomTS side — only the Cisco side is
    wired to real data so far.

HOW TO GET THE REAL DATA:
    git clone https://github.com/cisco-ie/telemetry.git
"""

import os
import io
import glob
import json
import zipfile
import numpy as np
import pandas as pd


BGP_PROCESS_INFO_NAME_SUBSTRING = "process-info"
GENERIC_COUNTERS_NAME_SUBSTRING = "generic-counters"

BGP_FEATURE_COLUMNS = [
    "global__established-neighbors-count-total",
    "global__neighbors-count-total",
]
COUNTER_FEATURE_COLUMNS = [
    "input-errors",
    "output-errors",
    "carrier-transitions",
    "crc-errors",
]
CISCO_FEATURE_COLUMNS = BGP_FEATURE_COLUMNS + COUNTER_FEATURE_COLUMNS

_READ_COLUMNS = ["name", "time", "Producer", "interface-name"] + CISCO_FEATURE_COLUMNS

# case_id -> how to find its ground truth. "baseline" means no anomaly
# ever occurs in that case (healthy-only data).
SUPPORTED_CASES = {
    "0": {"label_type": "baseline"},
    "1": {"label_type": "baseline"},
    "2": {"label_type": "ground_truth_csv", "file": "bgpclear_ground_truth.txt"},
    "5": {"label_type": "event_log_json", "file": "bgpclear_apptraffic_event.log"},
    "6": {"label_type": "event_log_json", "file": "bgpclear_no_traffic_event.log"},
}

# How long before an event's recorded start to also mark as "at risk"
# (label=1). The events in this dataset are deliberately-triggered test
# anomalies (a scripted BGP clear), not a slow multi-day degradation ramp
# like PredictLine's synthetic data models — so unlike data_pipeline.py's
# 48-96 hour ramp, there is no real pre-fault trend to detect here. This
# lead-in only captures the moment right before the logged start time.
# Evaluating "lead time" on this data therefore measures reaction latency
# to an instantaneous event, not early warning of gradual degradation — a
# real limitation of this dataset for PredictLine's use case, disclosed
# here rather than glossed over.
DEFAULT_LEAD_SECONDS = 8

# process-info reports roughly once per ~10s per device (see module
# docstring); bin width should be at least that coarse so each bin
# usually has a real reading rather than only forward-filled values.
DEFAULT_BIN_SECONDS = 10


def inspect_cisco_case(case_dir):
    """
    Print the files found in a Cisco telemetry case folder, plus the
    distinct telemetry sensor paths (the `name` column) and how many rows
    each contributes. Run this on any new case before assuming
    CISCO_FEATURE_COLUMNS applies to it — different case/event types
    (port flap vs BGP clear vs transceiver pull) may need different
    sensor paths as the primary signal (see module docstring).
    """
    print(f"Contents of {case_dir}:")
    for f in sorted(os.listdir(case_dir)):
        print(" -", f)

    csv_path = _find_case_csv(case_dir)
    if csv_path is None:
        print("No telemetry CSV found.")
        return

    df = _read_raw_csv(csv_path, usecols=["name", "Producer"], nrows=300000)
    print(f"\nSensor paths seen in first {len(df)} rows of {os.path.basename(csv_path)}:")
    print(df["name"].value_counts())
    print("\nDevices (Producer):", sorted(df["Producer"].dropna().unique().tolist()))


def _find_case_csv(case_dir):
    """Locate the main telemetry CSV in a case folder (.csv.gz or .csv.zip),
    ignoring header/case-data/ground-truth/log side-files."""
    candidates = glob.glob(os.path.join(case_dir, "*.csv.gz")) + \
        glob.glob(os.path.join(case_dir, "*.csv.zip"))
    return candidates[0] if candidates else None


def _read_raw_csv(csv_path, usecols, nrows=None):
    if csv_path.endswith(".gz"):
        return pd.read_csv(csv_path, usecols=usecols, nrows=nrows, low_memory=False)
    if csv_path.endswith(".zip"):
        with zipfile.ZipFile(csv_path) as z:
            inner_names = [n for n in z.namelist() if n.endswith(".csv")]
            if not inner_names:
                raise FileNotFoundError(f"No .csv found inside {csv_path}")
            with z.open(inner_names[0]) as f:
                return pd.read_csv(io.BytesIO(f.read()), usecols=usecols, nrows=nrows, low_memory=False)
    raise ValueError(f"Unsupported file type: {csv_path}")


def _load_events(case_dir, label_info):
    """Return a list of (device, start_epoch_s, end_epoch_s) tuples."""
    label_type = label_info["label_type"]
    if label_type == "baseline":
        return []

    path = os.path.join(case_dir, label_info["file"])

    if label_type == "ground_truth_csv":
        gt = pd.read_csv(path)
        return [(row.Node, float(row.Start), float(row.End)) for row in gt.itertuples()]

    if label_type == "event_log_json":
        with open(path, "r") as f:
            events = json.load(f)
        return [(e["host_key"], e["start_time"] / 1000.0, e["end_time"] / 1000.0) for e in events]

    raise ValueError(f"Unknown label_type: {label_type}")


def _read_case_rows(case_dir):
    csv_path = _find_case_csv(case_dir)
    if csv_path is None:
        raise FileNotFoundError(f"No telemetry CSV found in {case_dir}")
    df = _read_raw_csv(csv_path, usecols=_READ_COLUMNS)
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df = df.dropna(subset=["time", "Producer"])
    df["time_s"] = df["time"] / 1e9
    return df


def _bin_bgp_rows(df, bin_seconds):
    bgp = df[df["name"].astype(str).str.contains(BGP_PROCESS_INFO_NAME_SUBSTRING, na=False)].copy()
    for col in BGP_FEATURE_COLUMNS:
        bgp[col] = pd.to_numeric(bgp[col], errors="coerce")
    bgp = bgp.dropna(subset=BGP_FEATURE_COLUMNS)
    bgp["bin"] = (bgp["time_s"] // bin_seconds).astype(int)
    return bgp.groupby(["Producer", "bin"])[BGP_FEATURE_COLUMNS].mean()


def _bin_counter_rows(df, bin_seconds):
    counters = df[df["name"].astype(str).str.contains(GENERIC_COUNTERS_NAME_SUBSTRING, na=False)].copy()
    for col in COUNTER_FEATURE_COLUMNS:
        counters[col] = pd.to_numeric(counters[col], errors="coerce")
    counters = counters.dropna(subset=COUNTER_FEATURE_COLUMNS)
    counters["bin"] = (counters["time_s"] // bin_seconds).astype(int)
    # Sum across all interfaces on a device -> one row per (device, bin)
    return counters.groupby(["Producer", "bin"])[COUNTER_FEATURE_COLUMNS].sum()


def _label_bins(bin_values, bin_seconds, device, events, lead_seconds):
    labels = np.zeros(len(bin_values), dtype=int)
    bin_times = np.asarray(bin_values) * bin_seconds
    for ev_device, start_s, end_s in events:
        if ev_device != device:
            continue
        mask = (bin_times >= start_s - lead_seconds) & (bin_times <= end_s)
        labels[mask] = 1
    return labels


def load_cisco_case(case_dir, feature_cols=None, window=12, bin_seconds=DEFAULT_BIN_SECONDS,
                     lead_seconds=DEFAULT_LEAD_SECONDS, label_info=None, return_groups=False):
    """
    Parse one real Cisco telemetry case folder into (X, y, feature_names).

    Each device (Producer) is treated as its own segment (mirroring
    data_pipeline.py's segment_id concept). BGP process-info readings and
    summed interface counters are each binned to a uniform `bin_seconds`
    grid, joined per device, forward/back-filled across small gaps, then
    windowed exactly like data_pipeline.make_windows.

    If return_groups=True, also returns a `groups` array (one entry per
    window, equal to the device name) so callers can do a grouped
    train/test split — i.e. keep every window from a given device
    entirely on one side of the split, which avoids leaking near-duplicate
    overlapping windows across train/test while still allowing a split at
    finer granularity than "one whole case per side" (see
    train_real_cisco.py for why that matters: with only a handful of real
    positive EVENTS in this dataset, holding out entire cases starves the
    training set of positive examples).
    """
    if feature_cols is None:
        feature_cols = CISCO_FEATURE_COLUMNS
    if label_info is None:
        case_id = os.path.basename(os.path.normpath(case_dir))
        if case_id not in SUPPORTED_CASES:
            raise ValueError(
                f"Case {case_id!r} has no wired-up label source yet. "
                f"Supported cases: {sorted(SUPPORTED_CASES)}. See module docstring."
            )
        label_info = SUPPORTED_CASES[case_id]

    df = _read_case_rows(case_dir)
    events = _load_events(case_dir, label_info)

    bgp_binned = _bin_bgp_rows(df, bin_seconds)
    counter_binned = _bin_counter_rows(df, bin_seconds)
    joined = bgp_binned.join(counter_binned, how="outer")

    case_min_bin = int(df["time_s"].min() // bin_seconds)
    case_max_bin = int(df["time_s"].max() // bin_seconds)
    full_bins = range(case_min_bin, case_max_bin + 1)

    X, y, groups = [], [], []
    for device, dev_df in joined.groupby(level=0):
        dev_df = dev_df.droplevel(0).reindex(full_bins)
        dev_df[BGP_FEATURE_COLUMNS] = dev_df[BGP_FEATURE_COLUMNS].ffill().bfill()
        dev_df[COUNTER_FEATURE_COLUMNS] = dev_df[COUNTER_FEATURE_COLUMNS].fillna(0.0)
        if dev_df[feature_cols].isna().any().any():
            continue  # device never reported BGP process-info at all in this case; skip

        values = dev_df[feature_cols].values
        labels = _label_bins(dev_df.index, bin_seconds, device, events, lead_seconds)

        for i in range(window, len(dev_df)):
            X.append(values[i - window:i])
            y.append(labels[i])
            groups.append(device)

    if not X:
        raise ValueError(f"No usable windows produced from {case_dir}")

    if return_groups:
        return np.array(X), np.array(y), feature_cols, np.array(groups)
    return np.array(X), np.array(y), feature_cols


def load_all_cisco_cases(telemetry_repo_root, feature_cols=None, window=12,
                          case_ids=None, bin_seconds=DEFAULT_BIN_SECONDS,
                          lead_seconds=DEFAULT_LEAD_SECONDS, verbose=True,
                          return_groups=False):
    """
    Load and concatenate windows across the supported case folders in a
    cloned cisco-ie/telemetry repo. Defaults to SUPPORTED_CASES (0, 1, 2,
    5, 6) — see the module docstring for exactly why the other numbered
    folders aren't included yet.

    If return_groups=True, also returns a `groups` array of
    "<case_id>::<device>" strings, one per window (see load_cisco_case's
    docstring for why this exists).
    """
    if case_ids is None:
        case_ids = sorted(SUPPORTED_CASES, key=int)

    X_list, y_list, group_list = [], [], []
    for case_id in case_ids:
        case_dir = os.path.join(telemetry_repo_root, case_id)
        if not os.path.isdir(case_dir):
            print(f"Skipping case {case_id}: folder not found under {telemetry_repo_root}")
            continue
        try:
            result = load_cisco_case(
                case_dir, feature_cols=feature_cols, window=window,
                bin_seconds=bin_seconds, lead_seconds=lead_seconds,
                return_groups=return_groups,
            )
        except (FileNotFoundError, ValueError) as e:
            print(f"Skipping case {case_id}: {e}")
            continue
        if return_groups:
            X, y, cols, dev_groups = result
            group_list.append(np.array([f"{case_id}::{d}" for d in dev_groups]))
        else:
            X, y, cols = result
        if verbose:
            print(f"  case {case_id}: {len(X)} windows, {y.mean():.3f} positive rate")
        X_list.append(X)
        y_list.append(y)

    if not X_list:
        raise ValueError("No case folders could be loaded successfully.")

    Xc, yc = np.concatenate(X_list), np.concatenate(y_list)
    if return_groups:
        return Xc, yc, feature_cols or CISCO_FEATURE_COLUMNS, np.concatenate(group_list)
    return Xc, yc, feature_cols or CISCO_FEATURE_COLUMNS
