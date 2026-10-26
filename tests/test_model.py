"""Basic tests for the churn pipeline. Run with:  pytest -q"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pipeline import FEATURES, load_data  # noqa: E402
from predict import load_model, predict, predict_one, risk_band  # noqa: E402

HIGH_RISK = {
    "tenure": 1, "MonthlyCharges": 95.0, "TotalCharges": 95.0, "gender": "Female",
    "SeniorCitizen": 1, "Partner": "No", "Dependents": "No", "PhoneService": "Yes",
    "MultipleLines": "Yes", "InternetService": "Fiber optic", "OnlineSecurity": "No",
    "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
}
LOW_RISK = {
    **HIGH_RISK, "tenure": 70, "TotalCharges": 1500.0, "MonthlyCharges": 20.0,
    "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes", "MultipleLines": "No",
    "InternetService": "No", "OnlineSecurity": "No internet service",
    "OnlineBackup": "No internet service", "DeviceProtection": "No internet service",
    "TechSupport": "No internet service", "StreamingTV": "No internet service",
    "StreamingMovies": "No internet service", "Contract": "Two year",
    "PaperlessBilling": "No", "PaymentMethod": "Credit card (automatic)",
}


def test_data_loads_cleanly():
    X, y = load_data()
    assert list(X.columns) == FEATURES
    assert len(X) == len(y) == 7043
    assert set(y.unique()) == {0, 1}
    assert 0.2 < y.mean() < 0.3  # ~26.5% churn rate


def test_saved_model_has_threshold():
    bundle = load_model()
    assert 0.0 < bundle["threshold"] < 1.0


def test_probabilities_are_valid():
    X, _ = load_data()
    out = predict(X.sample(200, random_state=0))
    assert out["churn_probability"].between(0, 1).all()
    assert set(out["risk"]) <= {"Low", "Medium", "High"}


def test_risk_ordering_makes_sense():
    assert predict_one(HIGH_RISK)["churn_probability"] > 0.7
    assert predict_one(LOW_RISK)["churn_probability"] < 0.1


def test_blank_total_charges_is_handled():
    out = predict_one({**HIGH_RISK, "TotalCharges": " "})
    assert 0 <= out["churn_probability"] <= 1


def test_missing_column_raises():
    try:
        predict(pd.DataFrame([{"tenure": 5}]))
    except ValueError as err:
        assert "Missing columns" in str(err)
    else:
        raise AssertionError("expected ValueError")


def test_risk_band_thresholds():
    assert risk_band(0.10, 0.5) == "Low"
    assert risk_band(0.60, 0.5) == "Medium"
    assert risk_band(0.90, 0.5) == "High"
