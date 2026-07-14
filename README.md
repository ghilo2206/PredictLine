# PredictLine

**AI-based predictive fault detection for Powertel's OPGW & IoT-enabled smart grid communication networks.**

AI4I Challenge 2026 — Track 3 (Development) submission.

---

## 1. The problem

Powertel's OPGW fibre, IP/MPLS, and LoRaWAN-based IoT communication networks carry the telemetry that keeps Zimbabwe's electricity grid monitored. Fault management today is reactive, thresholds fire only after service has already degraded. PredictLine predicts fault risk **before** a hard failure, using a sequence model (GRU) trained on network telemetry, so operators get a lead-time window to intervene instead of just an outage alert.

## 2. What's real vs synthetic right now

| Component | Status |
|---|---|
| Synthetic telemetry data generator | ✅ Working, run and verified |
| Threshold baseline model | ✅ Working, run and verified |
| GRU fault-risk model | ✅ Working, trained and evaluated (from-scratch NumPy implementation — see note below) |
| Evaluation harness (GRU vs baseline, lead-time comparison) | ✅ Working, produces real numbers on synthetic data |
| **Real-data training on cisco-ie/telemetry** (`train_real_cisco.py`) | ✅ Working, trained and evaluated on real network telemetry — AUC 0.935 held out by case+device. See Section 5b. |
| Cross-dataset validation harness (Option A) | ✅ Harness built and dry-run verified; ⚠️ the Cisco side now trains on real data (Section 5b), but the TelecomTS side of this specific harness still uses mock data — see Section 5a |
| Unit tests | ✅ 25/25 passing |
| FastAPI backend | ⚠️ Written and syntax-checked, **not execution-tested**|
| Real Powertel telemetry | ❌ Not yet integrated — see Dataset Provenance below |
| Dashboard / frontend | ❌ Not yet built |


## 3. Why a from-scratch NumPy GRU instead of PyTorch/TensorFlow?

This repository was built in a development environment without internet access to install ML frameworks. Rather than fake the model or skip it, `src/models/gru_model.py` implements the actual GRU gate equations (update gate, reset gate, candidate state) and real backpropagation-through-time by hand in NumPy. It is a genuine, working, trainable model — not a placeholder.

**Before the bootcamp / ZCHPC CCE training run, this should be swapped for `torch.nn.GRU`** for GPU speed and easier hyperparameter search. The model architecture, feature set, and evaluation harness stay identical — only the execution backend changes. This is flagged in the AI4I proposal (Section 3.3) as planned, not hidden.

## 4. Quickstart

```bash
pip install -r requirements.txt

# Generate the synthetic dataset
python src/data_pipeline.py

# Train the GRU and compare against the threshold baseline (synthetic data)
python src/train.py

# Train and evaluate on REAL cisco-ie/telemetry data (see README Section 5b)
# Requires: git clone https://github.com/cisco-ie/telemetry.git as a sibling of this repo
python src/train_real_cisco.py --cisco-root ../telemetry

# Run the test suite
python -m unittest discover tests
# or: pytest tests/
```

Expected output from `train.py` (numbers will vary slightly run to run due to random seeding, but should be in this ballpark):

```
--- Held-out Test Set Comparison ---
Metric         Threshold (current practice)   GRU (proposed)
Precision                             0.941            0.966
Recall                                0.364            0.636
AUC                       n/a (binary rule)            0.945

--- Lead-Time Comparison ---
Mean threshold lead time: ~21 hours before fault
Mean GRU lead time:       ~43 hours before fault
```

**This is the evidence behind the proposal's "Why AI Is Necessary" argument (Section 2.2):** the GRU roughly doubles the warning lead-time versus the current reactive threshold approach, on top of higher precision and recall. All numbers are on synthetic data — see Section 5 below.

## 5. Dataset provenance (read before citing these numbers anywhere)

`src/data_pipeline.py` generates **synthetic** telemetry: signal loss, latency, packet loss, and temperature time series with an injected slow degradation ramp (48–96 hours) before a hard fault. This mimics the general shape of OPGW/IoT fault progression described in the literature (see proposal references), but is **not** real Powertel data.

- **Real data pathway:** one of our team members works at Powertel; this gives us a genuine internal channel to pursue a formal data-sharing agreement, but that agreement is not yet in place as of this submission.
- **Public real-network validation (in progress):** rather than only relying on synthetic data, we are validating the model against two independent, real, publicly available telecom telemetry sources — see Section 5a below.
- No personal or individually identifiable data is used or generated anywhere in this repository.

## 5a. Cross-dataset validation (Option A — in progress, not yet run on real data)

To strengthen the "does this generalize beyond our synthetic data" question, we are validating PredictLine against two real, independent, public network telemetry datasets, **without merging them into one blended pool** (their KPIs, scales, and anomaly types differ too much for a naive merge to be meaningful):

| Source | What it is | Status |
|---|---|---|
| [cisco-ie/telemetry](https://github.com/cisco-ie/telemetry) | Real network telemetry released by Cisco for anomaly-detection research (BGP anomalies, port flaps, transceiver events), published alongside ACM SIGCOMM/IEEE INFOCOM papers | Adapter written (`src/adapters/cisco_adapter.py`), column mapping **not yet finalized** against real files |
| [AliMaatouk/TelecomTS](https://huggingface.co/datasets/AliMaatouk/TelecomTS) | Real observability data from a 5G telecom testbed, de-anonymized with absolute scale preserved | Adapter written (`src/adapters/telecomts_adapter.py`), KPI selection **not yet finalized** against real files |

**The approach (`src/cross_dataset_eval.py`):** train the GRU on one real dataset, test it on the other, then flip direction. A model that holds up when tested on a completely different real network is stronger evidence than one only validated on synthetic data — this is a deliberately harder bar to clear than reporting a single in-sample number.

**Current status, stated plainly:** the harness itself is built, tested, and runs correctly end-to-end (verified via `python src/cross_dataset_eval.py --dry-run`, using mock data shaped like each dataset's real schema — see the script's docstring). **It has not yet been run against the actual downloaded datasets** — that requires pulling the real files (both need internet access we didn't have while building this) and finalizing two small column-mapping TODOs in the adapters. See those files for exact next steps.

## 5b. Real-data training on cisco-ie/telemetry (done, not a dry-run)

Separately from the cross-dataset harness above (which still awaits the TelecomTS side), **`src/train_real_cisco.py` trains and evaluates the GRU on real, downloaded cisco-ie/telemetry data** — a real network's BGP-clear test anomalies, not synthetic data. This closes one of the gaps flagged in Section 2.

**What we found inspecting the real files first:** the dataset is a sparse, long-format telemetry log — each row reports one YANG sensor path, and different paths populate different columns. An early version of this adapter used the densely-populated interface `data-rate` path (reliability, data-rate, load), but checking it against a known labelled BGP-clear event showed **no visible change** in any of those columns — expected in hindsight, since a BGP session clear is a control-plane event, not a physical-layer one. Checking the same event against the `bgp/.../process-info` sensor path instead showed `global__established-neighbors-count-total` drop from a steady 38 to 0 and recover, exactly inside the labelled window. That verified signal — plus summed interface error/drop counters as a supplementary feature — is what the adapter (`src/adapters/cisco_adapter.py`) now uses.

**Cases wired up:** 0, 1 (healthy baselines), 2, 5, 6 (real BGP-clear events, three different label-file formats: a ground-truth CSV, and two JSON event logs). Cases 3, 4, 7, 9, 10, 11, 12 are **not yet wired up** — reasons are stated case-by-case in `cisco_adapter.py`'s module docstring (PDT timestamp parsing, image-only ground truth, unexplored nested layouts).

**Evaluation methodology:** windows are split by **(case, device) group**, not by case alone and not by shuffled window — an entire case holdout left too few real positive examples to learn from (only 3 of the 5 wired-up cases contain any real fault events at all), and a shuffled-window split would leak near-duplicate overlapping windows across train/test. Grouping by case+device avoids both problems while still holding out ~25% of groups entirely.

**Result on real, held-out data** (`python src/train_real_cisco.py --cisco-root ../telemetry`):

```
Metric          Threshold (reactive rule)     GRU @0.5   GRU @best-F1
Precision                           0.008        0.391          0.889
Recall                              0.730        0.608          0.541
AUC                     n/a (binary rule)        0.935         (same)
```

The reactive threshold (alert whenever established BGP neighbor count < configured count, no memory of trend) fires constantly — 0.8% precision — because BGP session state flaps briefly and often even outside labelled fault windows. The GRU discriminates real fault windows from normal operation with **AUC 0.935**, well above the baseline's uninformative firing pattern.

**Read the caveats, don't just quote the AUC:**
- This is a real result on a real, independent network dataset — but it is not Powertel data, and BGP-clear is a scripted, near-instantaneous test event, not the slow 48-96 hour degradation ramp the synthetic pipeline (Section 4-5) models. **Do not use this section's numbers as evidence for the synthetic pipeline's lead-time claim, or vice versa** — they answer different questions ("can the GRU detect a real anomaly signature" vs. "does the GRU provide early warning before a slow-onset fault").
- The "best-F1" threshold (0.789) is chosen on the test set itself, for reporting only — it is not a valid deployment threshold; a real one must come from a separate validation split.
- Only 5 of 13 case folders are wired up; the real positive-event count in this dataset is small (a few dozen scripted events total).

## 6. Repository structure

```
predictline/
├── README.md                  ← you are here
├── requirements.txt           ← pinned dependency versions
├── .env.example                ← env var template, no secrets
├── LICENSE                    ← MIT (code only — not a licence over real utility data)
├── src/
│   ├── data_pipeline.py       ← synthetic data generation + windowing + normalization
│   ├── train.py               ← trains GRU, runs baseline, prints comparison report (synthetic data)
│   ├── train_real_cisco.py    ← trains GRU on REAL cisco-ie/telemetry data — see README Section 5b
│   ├── cross_dataset_eval.py  ← Option A: train on one real dataset, test on another
│   ├── api.py                 ← FastAPI skeleton (see status table above)
│   ├── adapters/
│   │   ├── cisco_adapter.py       ← loads real cisco-ie/telemetry (verified against a labelled event — see Section 5b)
│   │   └── telecomts_adapter.py   ← loads AliMaatouk/TelecomTS (needs KPI TODOs filled in)
│   └── models/
│       ├── gru_model.py            ← from-scratch NumPy GRU classifier
│       └── threshold_baseline.py   ← rule-based "current practice" baseline
├── data/
│   └── sample_synthetic.csv   ← generated sample (safe, synthetic, regenerable)
├── docs/
│   └── architecture_diagram.png    ← system architecture (also in the AI4I proposal)
└── tests/
    ├── test_data_pipeline.py      ← 11 tests, all passing
    ├── test_cross_dataset_eval.py ← 6 tests, all passing (harness logic, dry-run verified)
    └── test_cisco_adapter.py      ← 8 tests, all passing (real-data adapter/training logic; doesn't require the multi-GB dataset itself)
```

## 7. System architecture

![Architecture diagram](docs/architecture_diagram.png)

Full description in the AI4I proposal, Section 2.4.

## 8. Known limitations (stated plainly, per AI4I guidance)

- No real Powertel telemetry integrated yet — data-sharing agreement pending.
- FastAPI endpoint is untested against a live server (see Section 2).
- The from-scratch GRU has not been benchmarked against a framework implementation (PyTorch) for speed or accuracy parity — expected to be very close since gate math is identical, but not yet verified.
- Edge deployment (quantized model, <256MB RAM, <100ms latency) is a target, not yet validated on real hardware.
- No dashboard/frontend yet — a natural integration point for a Design-track partner team.

## 9. Team

- Mlalazi Mzwakhe — [role], University of Zimbabwe
- [Powertel team member] — [role], Powertel Communications
- [TelOne team member(s)] — [role], TelOne

## 10. Licence

MIT for code in this repository (see `LICENSE`). This does not extend to any real Powertel/ZESA data used in future work, which is subject to separate agreements and Zimbabwe's Data Protection Act [Chapter 12:07].
