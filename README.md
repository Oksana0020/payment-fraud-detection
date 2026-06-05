# Real-Time Payment Fraud Detection System

A Python fraud detection project focused on suspicious payment detection, transaction risk scoring and analyst workflows for fraud prevention and financial risk management.

## Stack

- Python
- pandas
- scikit-learn
- FastAPI
- Streamlit

## Features

- Analyze transactions with a trained fraud model
- Detect suspicious payments and assign risk scores
- Expose API endpoints for health, metrics, prediction, batch analysis, and retraining
- Provide an analyst dashboard for reviewing flagged payments
- Generate a sample transaction dataset for local demos and experimentation

## Project structure

- `app/` API, data generation, schemas, and model pipeline
- `dashboard/` Streamlit analyst dashboard
- `data/` generated sample transaction CSV
- `models/` trained model artifact
- `train_model.py` local training entry point

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model.py
```

## Run the API

```powershell
uvicorn app.main:app --reload
```

API docs will be available at `http://127.0.0.1:8000/docs`.

## Run the dashboard

```powershell
streamlit run dashboard/streamlit_app.py
```

## Sample prediction payload

```json
{
  "transaction_id": "TX-900001",
  "amount": 1499.99,
  "account_age_days": 21,
  "transactions_last_24h": 13,
  "avg_amount_7d": 220.5,
  "chargeback_history": 2,
  "is_international": true,
  "is_night_transaction": true,
  "device_trusted": false,
  "merchant_risk_score": 0.88,
  "ip_velocity": 27,
  "geo_distance_km": 1450,
  "payment_method_risk": 0.79,
  "customer_tenure_segment": "new",
  "merchant_category": "luxury"
}
```
