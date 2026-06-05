from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.config import DEFAULT_RANDOM_STATE, MODEL_PATH, RISK_THRESHOLD
from app.data import ensure_dataset

NUMERIC_FEATURES = [
    "amount",
    "account_age_days",
    "transactions_last_24h",
    "avg_amount_7d",
    "chargeback_history",
    "merchant_risk_score",
    "ip_velocity",
    "geo_distance_km",
    "payment_method_risk",
]
BOOLEAN_FEATURES = [
    "is_international",
    "is_night_transaction",
    "device_trusted",
]
CATEGORICAL_FEATURES = ["customer_tenure_segment", "merchant_category"]
FEATURES = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL_FEATURES
MODEL_VERSION = "2026.06.05"


@dataclass
class ModelBundle:
    pipeline: Pipeline
    roc_auc: float
    classification_report: dict
    model_version: str


def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("boolean", "passthrough", BOOLEAN_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=220,
        max_depth=12,
        min_samples_leaf=3,
        random_state=DEFAULT_RANDOM_STATE,
        class_weight="balanced_subsample",
    )
    return Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", classifier)]
    )


def train_and_save_model() -> ModelBundle:
    df = ensure_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        df[FEATURES],
        df["is_fraud"],
        test_size=0.2,
        random_state=DEFAULT_RANDOM_STATE,
        stratify=df["is_fraud"],
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    bundle = ModelBundle(
        pipeline=pipeline,
        roc_auc=float(roc_auc_score(y_test, probabilities)),
        classification_report=classification_report(
            y_test,
            predictions,
            output_dict=True,
        ),
        model_version=MODEL_VERSION,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    return bundle


def load_model() -> ModelBundle:
    if not MODEL_PATH.exists():
        return train_and_save_model()
    return joblib.load(MODEL_PATH)


def prepare_frame(records: Iterable[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    for column in BOOLEAN_FEATURES:
        frame[column] = frame[column].astype(bool)
    return frame


def score_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    model_bundle = load_model()
    probabilities = model_bundle.pipeline.predict_proba(frame[FEATURES])[:, 1]
    scores = (probabilities * 100).round().astype(int)

    results = frame.copy()
    results["fraud_probability"] = probabilities.round(4)
    results["risk_score"] = scores
    results["is_suspicious"] = probabilities >= RISK_THRESHOLD
    results["recommended_action"] = results["is_suspicious"].map(
        {
            True: "Hold payment and escalate to analyst",
            False: "Approve with routine monitoring",
        }
    )
    results["model_version"] = model_bundle.model_version
    return results
