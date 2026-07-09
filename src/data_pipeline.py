"""
data_pipeline.py

Generates a SYNTHETIC network-telemetry dataset that mimics the structure of
real Powertel OPGW / IoT / IP-MPLS telemetry (signal loss, latency, packet
loss, temperature), with injected pre-fault degradation patterns.

This is a stand-in for real Powertel data, used until a formal data-sharing
agreement is in place. Every function here is documented so it is obvious
which parts are synthetic and how they were generated (see Dataset
Provenance disclosure in the AI4I proposal, Section 4.2).

Run directly to generate data/sample_synthetic.csv:
    python src/data_pipeline.py
"""

import numpy as np
import pandas as pd


SEGMENT_COUNT = 6
HOURS = 24 * 30  # 30 days of hourly telemetry per segment
RANDOM_SEED = 42


def _simulate_segment(segment_id: str, rng: np.random.Generator, will_fault: bool):
    """
    Simulate one network segment's hourly telemetry for HOURS hours.

    Baseline behaviour: signal loss, latency and packet loss fluctuate
    around a healthy set point with small noise. If `will_fault` is True,
    a slow degradation trend is injected over the final ~48-96 hours,
    followed by a hard fault (label = 1) — this is the pattern PredictLine
    is meant to learn to recognise *before* the hard fault occurs.
    """
    t = np.arange(HOURS)

    signal_loss_db = 2.0 + 0.3 * np.sin(t / 24) + rng.normal(0, 0.15, HOURS)
    latency_ms = 12 + 1.5 * np.sin(t / 24 + 1) + rng.normal(0, 0.8, HOURS)
    packet_loss_pct = np.abs(rng.normal(0.05, 0.03, HOURS))
    temperature_c = 24 + 4 * np.sin(t / (24 * 7)) + rng.normal(0, 0.5, HOURS)

    label = np.zeros(HOURS, dtype=int)
    fault_hour_value = -1  # -1 = this segment never has a hard fault

    if will_fault:
        fault_hour = int(rng.integers(24 * 10, HOURS - 24))
        fault_hour_value = fault_hour
        degrade_start = max(0, fault_hour - rng.integers(48, 96))
        ramp_len = fault_hour - degrade_start
        ramp = np.linspace(0, 1, ramp_len) ** 2

        signal_loss_db[degrade_start:fault_hour] += ramp * rng.uniform(4, 8)
        latency_ms[degrade_start:fault_hour] += ramp * rng.uniform(15, 30)
        packet_loss_pct[degrade_start:fault_hour] += ramp * rng.uniform(2, 5)

        repair_len = int(rng.integers(6, 18))
        fault_end = min(HOURS, fault_hour + repair_len)
        signal_loss_db[fault_hour:fault_end] += rng.uniform(10, 20)
        latency_ms[fault_hour:fault_end] += rng.uniform(40, 100)
        packet_loss_pct[fault_hour:fault_end] += rng.uniform(10, 40)

        label[degrade_start:fault_end] = 1

    df = pd.DataFrame({
        "segment_id": segment_id,
        "hour": t,
        "signal_loss_db": signal_loss_db.clip(min=0),
        "latency_ms": latency_ms.clip(min=0),
        "packet_loss_pct": packet_loss_pct.clip(min=0, max=100),
        "temperature_c": temperature_c,
        "fault_risk_label": label,
        "fault_hour": fault_hour_value,
    })
    return df


def generate_synthetic_dataset(seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate telemetry for SEGMENT_COUNT segments. Roughly half develop a
    fault within the 30-day window; the rest stay healthy throughout, so the
    model sees both classes.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(SEGMENT_COUNT):
        segment_id = f"SEG-{i+1:03d}"
        will_fault = (i % 2 == 0)
        frames.append(_simulate_segment(segment_id, rng, will_fault))
    return pd.concat(frames, ignore_index=True)


def make_windows(df: pd.DataFrame, window: int = 12, feature_cols=None):
    """
    Convert per-hour telemetry into sliding windows suitable for a sequence
    model. Returns X of shape (n_samples, window, n_features) and y of shape
    (n_samples,) — label is whether fault_risk is 1 at the END of the window.
    """
    if feature_cols is None:
        feature_cols = ["signal_loss_db", "latency_ms", "packet_loss_pct", "temperature_c"]

    X, y = [], []
    for segment_id, g in df.groupby("segment_id"):
        g = g.sort_values("hour").reset_index(drop=True)
        values = g[feature_cols].values
        labels = g["fault_risk_label"].values
        for i in range(window, len(g)):
            X.append(values[i - window:i])
            y.append(labels[i])
    return np.array(X), np.array(y)


def normalize(X: np.ndarray):
    """Z-score normalize features across the whole window set. Returns
    normalized X plus (mean, std) so the same scaling can be applied at
    inference time."""
    mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
    std = X.reshape(-1, X.shape[-1]).std(axis=0) + 1e-8
    return (X - mean) / std, mean, std


if __name__ == "__main__":
    data = generate_synthetic_dataset()
    out_path = "data/sample_synthetic.csv"
    data.to_csv(out_path, index=False)
    print(f"Wrote {len(data)} rows across {data['segment_id'].nunique()} segments to {out_path}")
    print(f"Fault-risk positive rate: {data['fault_risk_label'].mean():.3f}")
