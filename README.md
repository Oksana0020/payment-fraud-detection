# Real-Time Payment Fraud Detection System

A Python fraud detection project focused on suspicious payment detection, transaction risk scoring and analyst workflows for fraud prevention and financial risk management.

This project combines a fraud scoring model, operational API endpoints and an analyst dashboard that makes the prediction flow easier to inspect and explain.

## Highlights

- End-to-end workflow from data generation to model training to analyst review
- API-first design that makes the scoring service usable beyond the dashboard
- Dashboard experience tailored to fraud triage rather than generic data display
- Strong fit for AI engineering, FinTech, data science, and cybersecurity portfolios


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

### Key API routes

- `GET /` project summary and service status
- `GET /health` health check with model metadata
- `GET /metrics` model evaluation summary
- `GET /transactions/sample` sample scored transactions for demos
- `POST /predict` score a single transaction
- `POST /analyze` score a batch of transactions
- `POST /train` retrain and persist the model artifact

## Run the dashboard

```powershell
streamlit run dashboard/streamlit_app.py
```

The dashboard includes:

- A triage queue for flagged payments
- Visual breakdowns of risk distribution and merchant category exposure
- A scenario lab for manual payment simulation
- CSV upload support for batch analyst review


