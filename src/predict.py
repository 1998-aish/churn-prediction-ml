"""Load the saved churn model and score customers.

Example:
    from src.predict import predict_one
    predict_one({"tenure": 2, "Contract": "Month-to-month", ...})
"""
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import FEATURES, MODEL_PATH  # noqa: E402


@lru_cache(maxsize=1)
def load_model(path: Path = MODEL_PATH) -> dict:
    """Return {"model": fitted pipeline, "threshold": float}."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Train it first with: python src/train.py")
    return joblib.load(path)


def risk_band(probability: float, threshold: float) -> str:
    if probability >= max(threshold + 0.25, 0.75):
        return "High"
    if probability >= threshold:
        return "Medium"
    return "Low"


def predict(customers: pd.DataFrame) -> pd.DataFrame:
    """Score a DataFrame of customers; returns probability, flag and risk band per row."""
    missing = set(FEATURES) - set(customers.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    bundle = load_model()
    X = customers[FEATURES].copy()
    X["TotalCharges"] = pd.to_numeric(X["TotalCharges"], errors="coerce")
    proba = bundle["model"].predict_proba(X)[:, 1]
    return pd.DataFrame({
        "churn_probability": proba.round(3),
        "will_churn": proba >= bundle["threshold"],
        "risk": [risk_band(p, bundle["threshold"]) for p in proba],
    }, index=customers.index)


def predict_one(customer: dict) -> dict:
    return predict(pd.DataFrame([customer])).iloc[0].to_dict()
