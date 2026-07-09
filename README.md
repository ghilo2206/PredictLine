# PredictLine

**AI-based predictive fault detection for Powertel's OPGW & IoT-enabled smart grid communication networks.**

AI4I Challenge 2026 — Track 3 (Development) submission.

---

## 1. The problem, in one paragraph

Powertel's OPGW fibre, IP/MPLS, and LoRaWAN-based IoT communication networks carry the telemetry that keeps Zimbabwe's electricity grid monitored. Fault management today is reactive — thresholds fire only after service has already degraded. PredictLine predicts fault risk **before** a hard failure, using a sequence model (GRU) trained on network telemetry, so operators get a lead-time window to intervene instead of just an outage alert.

## 2. What's real vs synthetic right now — read this first

**Be honest with yourself and with judges about what's actually working here:**

| Component | Status |
|---|---|
| Synthetic telemetry data generator | ✅ Working, run and verified |
| Threshold baseline model | ✅ Working, run and verified |
| GRU fault-risk model | ✅ Working, trained and evaluated (from-scratch NumPy implementation — see note below) |
| Evaluation harness (GRU vs baseline, lead-time comparison) | ✅ Working, produces real numbers on synthetic data |
| Unit tests | ✅ 11/11 passing |
| FastAPI backend | ⚠️ Written and syntax-checked, **not execution-tested** (no internet access in the dev sandbox to install fastapi/uvicorn) |
| Real Powertel telemetry | ❌ Not yet integrated — see Dataset Provenance below |
| Dashboard / frontend | ❌ Not yet built — natural pairing point with a Design-track team |

We would rather show you exactly where the line between "working" and "planned" sits than dress up an untested claim. This matches the AI4I Supporting Guidance's explicit instruction that teams "explain what remains incomplete."

## 3. Why a from-scratch NumPy GRU instead of PyTorch/TensorFlow?

This repository was built in a development environment without internet access to install ML frameworks. Rather than fake the model or skip it, `src/models/gru_model.py` implements the actual GRU gate equations (update gate, reset gate, candidate state) and real backpropagation-through-time by hand in NumPy. It is a genuine, working, trainable model — not a placeholder.

**Before the bootcamp / ZCHPC CCE training run, this should be swapped for `torch.nn.GRU`** for GPU speed and easier hyperparameter search. The model architecture, feature set, and evaluation harness stay identical — only the execution backend changes. This is flagged in the AI4I proposal (Section 3.3) as planned, not hidden.

## 4. Quickstart

```bash
pip install -r requirements.txt

# Generate the synthetic dataset
python src/data_pipeline.py

# Train the GRU and compare against the threshold baseline
python src/train.py

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
- **Synthetic fallback (current state):** parametric simulation (see code, fully commented). A TimeGAN-based generator calibrated against public telecom fault datasets, with statistical correlation validation (KS-test on marginals, autocorrelation comparison), is the planned next step per the AI4I proposal Section 4.2 — not yet implemented in this repo.
- No personal or individually identifiable data is used or generated anywhere in this repository.

## 6. Repository structure

```
predictline/
├── README.md                  ← you are here
├── requirements.txt           ← pinned dependency versions
├── .env.example                ← env var template, no secrets
├── LICENSE                    ← MIT (code only — not a licence over real utility data)
├── src/
│   ├── data_pipeline.py       ← synthetic data generation + windowing + normalization
│   ├── train.py               ← trains GRU, runs baseline, prints comparison report
│   ├── api.py                 ← FastAPI skeleton (see status table above)
│   └── models/
│       ├── gru_model.py            ← from-scratch NumPy GRU classifier
│       └── threshold_baseline.py   ← rule-based "current practice" baseline
├── data/
│   └── sample_synthetic.csv   ← generated sample (safe, synthetic, regenerable)
├── docs/
│   └── architecture_diagram.png    ← system architecture (also in the AI4I proposal)
└── tests/
    └── test_data_pipeline.py  ← 11 tests, all passing (see Section 4 to run them)
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
