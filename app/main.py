from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from app.data import ensure_dataset
from app.model import (
    FEATURES,
    load_model,
    prepare_frame,
    score_transactions,
    train_and_save_model,
)
from app.schemas import PredictionResponse, TransactionRequest


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_dataset()
    load_model()
    yield


app = FastAPI(
    title="Real-Time Payment Fraud Detection System",
    description=(
        "API for transaction analysis, fraud prevention, and "
        "financial risk management."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict:
    model_bundle = load_model()
    return {
        "project": "Real-Time Payment Fraud Detection System",
        "status": "online",
        "version": "1.0.0",
        "focus": [
            "transaction analysis",
            "suspicious payment detection",
            "risk scoring",
            "analyst workflow support",
        ],
        "docs": "/docs",
        "sample_transactions": "/transactions/sample",
        "model_version": model_bundle.model_version,
        "roc_auc": model_bundle.roc_auc,
    }


@app.get("/health")
def health() -> dict:
    model_bundle = load_model()
    return {
        "status": "ok",
        "model_version": model_bundle.model_version,
        "roc_auc": model_bundle.roc_auc,
    }


@app.get("/metrics")
def metrics() -> dict:
    model_bundle = load_model()
    return {
        "model_version": model_bundle.model_version,
        "roc_auc": model_bundle.roc_auc,
        "classification_report": model_bundle.classification_report,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionRequest) -> PredictionResponse:
    frame = prepare_frame([transaction.model_dump()])
    result = score_transactions(frame).iloc[0]

    explanation = (
        "Flagged due to elevated cross-border risk, velocity anomalies, "
        "or merchant/payment risk indicators."
        if result["is_suspicious"]
        else (
            "Risk profile is within expected operating bounds for this "
            "payment."
        )
    )

    return PredictionResponse(
        transaction_id=str(result["transaction_id"]),
        fraud_probability=float(result["fraud_probability"]),
        risk_score=int(result["risk_score"]),
        is_suspicious=bool(result["is_suspicious"]),
        risk_band=str(result["risk_band"]),
        recommended_action=str(result["recommended_action"]),
        model_version=str(result["model_version"]),
        explanation=explanation,
    )


@app.post("/analyze")
def analyze_transactions(transactions: list[TransactionRequest]) -> list[dict]:
    if not transactions:
        raise HTTPException(
            status_code=400,
            detail="At least one transaction is required.",
        )

    frame = prepare_frame(
        [transaction.model_dump() for transaction in transactions]
    )
    results = score_transactions(frame)
    return results.to_dict(orient="records")


@app.get("/transactions/sample")
def sample_transactions(limit: int = 50) -> list[dict]:
    dataset = ensure_dataset()
    subset = dataset.head(limit).copy()
    sample_records = subset[FEATURES + ["transaction_id"]].to_dict(
        orient="records"
    )
    scored = score_transactions(prepare_frame(sample_records))
    merged = pd.concat(
        [
            subset[["is_fraud"]].reset_index(drop=True),
            scored.reset_index(drop=True),
        ],
        axis=1,
    )
    return merged.to_dict(orient="records")


@app.post("/train")
def retrain_model() -> dict:
    bundle = train_and_save_model()
    return {
        "status": "retrained",
        "model_version": bundle.model_version,
        "roc_auc": bundle.roc_auc,
    }
