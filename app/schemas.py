from typing import Optional

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., examples=["TX-10025"])
    amount: float = Field(..., gt=0)
    account_age_days: int = Field(..., ge=0)
    transactions_last_24h: int = Field(..., ge=0)
    avg_amount_7d: float = Field(..., ge=0)
    chargeback_history: int = Field(..., ge=0)
    is_international: bool
    is_night_transaction: bool
    device_trusted: bool
    merchant_risk_score: float = Field(..., ge=0, le=1)
    ip_velocity: int = Field(..., ge=0)
    geo_distance_km: float = Field(..., ge=0)
    payment_method_risk: float = Field(..., ge=0, le=1)
    customer_tenure_segment: str = Field(..., examples=["new"])
    merchant_category: str = Field(..., examples=["electronics"])


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    risk_score: int
    is_suspicious: bool
    recommended_action: str
    model_version: str
    explanation: Optional[str] = None
