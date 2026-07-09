"""
api.py

FastAPI skeleton exposing PredictLine's fault-risk scoring as a REST API,
per the AI4I Development Track requirement for a defined "user interaction
plan" (Section 2 of the ToR).

REQUIRES: fastapi, uvicorn (see requirements.txt). This sandbox could not
install these (no internet access), so this file is written correctly and
documented, but has not been execution-tested here. Test locally with:

    pip install -r requirements.txt
    uvicorn src.api:app --reload

Then visit http://127.0.0.1:8000/docs for interactive Swagger UI.
"""

from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from models.gru_model import SimpleGRUClassifier

# NOTE: whoever wires up real model loading here should import the SAME
# feature ordering used at training time (see FEATURE_COLS in train.py)
# so the API and training pipeline never drift out of sync.

app = FastAPI(
    title="PredictLine API",
    description="Predictive fault-risk scoring for Powertel OPGW/IoT smart grid communication segments.",
    version="0.1.0",
)

FEATURE_NAMES = ["signal_loss_db", "latency_ms", "packet_loss_pct", "temperature_c"]
WINDOW_SIZE = 12

# In a real deployment this would load trained weights from the model
# registry (see architecture diagram, docs/architecture_diagram.png).
# For this skeleton we initialise an untrained model so the endpoint is
# structurally complete and testable; swap in a loaded checkpoint before
# real use.
_model = SimpleGRUClassifier(n_features=len(FEATURE_NAMES), hidden_size=8)
_norm_mean = np.zeros(len(FEATURE_NAMES))
_norm_std = np.ones(len(FEATURE_NAMES))


class TelemetryReading(BaseModel):
    signal_loss_db: float = Field(..., description="Optical signal loss in dB")
    latency_ms: float = Field(..., description="Link latency in milliseconds")
    packet_loss_pct: float = Field(..., description="Packet loss percentage")
    temperature_c: float = Field(..., description="Environmental/equipment temperature in Celsius")


class FaultRiskRequest(BaseModel):
    segment_id: str = Field(..., description="Network segment identifier, e.g. SEG-001")
    readings: List[TelemetryReading] = Field(
        ..., description=f"Chronological telemetry readings, most recent last. Must contain exactly {WINDOW_SIZE} readings."
    )


class FaultRiskResponse(BaseModel):
    segment_id: str
    fault_risk_probability: float
    alert: bool
    model_version: str = "predictline-gru-v0.1-untrained-skeleton"


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=FaultRiskResponse)
def predict_fault_risk(request: FaultRiskRequest):
    if len(request.readings) != WINDOW_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Expected exactly {WINDOW_SIZE} readings, got {len(request.readings)}.",
        )

    window = np.array([
        [r.signal_loss_db, r.latency_ms, r.packet_loss_pct, r.temperature_c]
        for r in request.readings
    ])
    window_norm = (window - _norm_mean) / _norm_std
    proba = _model.predict_proba(window_norm[np.newaxis, :, :])[0]

    return FaultRiskResponse(
        segment_id=request.segment_id,
        fault_risk_probability=float(proba),
        alert=bool(proba > 0.5),
    )

