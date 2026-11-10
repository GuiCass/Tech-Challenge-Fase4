from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
REFERENCE_DIR = DATA_DIR / "reference"
PRODUCTION_DIR = DATA_DIR / "production"
MODEL_DIR = DATA_DIR / "model"

REPORTS_DIR = PROJECT_ROOT / "reports"
DRIFT_REPORTS_DIR = REPORTS_DIR / "drift"
DASHBOARD_DIR = REPORTS_DIR / "dashboard"

RAW_APPLICATION_FILE = RAW_DIR / "application_train.csv"
REFERENCE_FILE = REFERENCE_DIR / "credit_reference.csv"
PRODUCTION_FILE = PRODUCTION_DIR / "credit_production.csv"
MODEL_FILE = MODEL_DIR / "baseline_model.joblib"
