"""Shared data loading and model-building code for the churn project."""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Telco_churn.csv"
MODEL_PATH = ROOT / "models" / "churn_model.joblib"
SEED = 42

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load the Telco dataset and return features X and binary target y (1 = churn)."""
    df = pd.read_csv(path)
    # TotalCharges is blank for brand-new customers; coerce to NaN and let the imputer handle it
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    X = df[FEATURES].copy()
    y = (df["Churn"] == "Yes").astype(int)
    return X, y


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def build_model(C: float = 1.0, class_weight=None) -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(C=C, class_weight=class_weight,
                                          max_iter=1000, random_state=SEED)),
    ])
