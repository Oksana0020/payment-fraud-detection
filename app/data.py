from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.config import DATASET_PATH, DEFAULT_RANDOM_STATE


MERCHANT_CATEGORIES = [
    "electronics",
    "grocery",
    "travel",
    "gaming",
    "luxury",
    "marketplace",
    "utilities",
]
TENURE_SEGMENTS = ["new", "growing", "established", "vip"]


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-values))


def generate_sample_dataset(rows: int = 1500, seed: int = DEFAULT_RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    amount = np.round(rng.gamma(shape=2.1, scale=120, size=rows), 2)
    account_age_days = rng.integers(1, 3650, size=rows)
    transactions_last_24h = rng.poisson(3.5, size=rows)
    avg_amount_7d = np.round(amount * rng.uniform(0.4, 1.4, size=rows), 2)
    chargeback_history = rng.binomial(3, 0.08, size=rows)
    is_international = rng.binomial(1, 0.24, size=rows)
    is_night_transaction = rng.binomial(1, 0.31, size=rows)
    device_trusted = rng.binomial(1, 0.78, size=rows)
    merchant_risk_score = np.round(rng.uniform(0.05, 0.95, size=rows), 3)
    ip_velocity = rng.integers(1, 45, size=rows)
    geo_distance_km = np.round(rng.exponential(scale=180, size=rows), 2)
    payment_method_risk = np.round(rng.uniform(0.05, 0.95, size=rows), 3)
    customer_tenure_segment = rng.choice(TENURE_SEGMENTS, size=rows, p=[0.22, 0.31, 0.35, 0.12])
    merchant_category = rng.choice(MERCHANT_CATEGORIES, size=rows)

    linear_score = (
        (amount > 900) * 1.2
        + (transactions_last_24h > 8) * 1.1
        + chargeback_history * 0.65
        + is_international * 0.9
        + is_night_transaction * 0.45
        + (1 - device_trusted) * 0.95
        + merchant_risk_score * 1.7
        + payment_method_risk * 1.4
        + (ip_velocity > 15) * 0.7
        + (geo_distance_km > 700) * 0.9
        + (account_age_days < 45) * 0.8
        + (avg_amount_7d > 0) * np.clip((amount / np.maximum(avg_amount_7d, 1)) - 1.4, 0, None) * 0.9
    )

    category_risk = {
        "electronics": 0.3,
        "grocery": -0.2,
        "travel": 0.4,
        "gaming": 0.55,
        "luxury": 0.65,
        "marketplace": 0.5,
        "utilities": -0.15,
    }
    tenure_risk = {"new": 0.7, "growing": 0.2, "established": -0.1, "vip": -0.25}

    linear_score += np.vectorize(category_risk.get)(merchant_category)
    linear_score += np.vectorize(tenure_risk.get)(customer_tenure_segment)
    linear_score += rng.normal(0, 0.55, size=rows)

    fraud_probability = _sigmoid(linear_score - 3.2)
    is_fraud = rng.binomial(1, np.clip(fraud_probability, 0.02, 0.97))

    df = pd.DataFrame(
        {
            "transaction_id": [f"TX-{100000 + index}" for index in range(rows)],
            "amount": amount,
            "account_age_days": account_age_days,
            "transactions_last_24h": transactions_last_24h,
            "avg_amount_7d": avg_amount_7d,
            "chargeback_history": chargeback_history,
            "is_international": is_international.astype(bool),
            "is_night_transaction": is_night_transaction.astype(bool),
            "device_trusted": device_trusted.astype(bool),
            "merchant_risk_score": merchant_risk_score,
            "ip_velocity": ip_velocity,
            "geo_distance_km": geo_distance_km,
            "payment_method_risk": payment_method_risk,
            "customer_tenure_segment": customer_tenure_segment,
            "merchant_category": merchant_category,
            "is_fraud": is_fraud,
        }
    )
    return df


def ensure_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return pd.read_csv(path)

    dataset = generate_sample_dataset()
    dataset.to_csv(path, index=False)
    return dataset
