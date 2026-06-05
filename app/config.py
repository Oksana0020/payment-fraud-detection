from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "fraud_model.joblib"
DATASET_PATH = DATA_DIR / "transactions_sample.csv"
DEFAULT_RANDOM_STATE = 42
RISK_THRESHOLD = 0.72
