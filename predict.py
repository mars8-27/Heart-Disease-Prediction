"""Prediction logic: loads the trained XGBoost model and scores one patient."""
from pathlib import Path

import joblib
import pandas as pd

MODEL_PATH = Path("models/xgb_model.joblib")

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

_model = None
_explainer = None


def load_model():
    """Load the model once and reuse it."""
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Model not found. Run `python train_model.py` first.")
        _model = joblib.load(MODEL_PATH)
    return _model


def predict(features: dict) -> tuple[int, float]:
    """Return (predicted_class, probability_of_heart_disease)."""
    model = load_model()
    row = pd.DataFrame([features])[FEATURES]
    probability = float(model.predict_proba(row)[0][1])
    prediction = int(probability >= 0.5)
    return prediction, probability


def explain(features: dict) -> pd.Series:
    """Return SHAP contributions per feature (log-odds), sorted by absolute impact.

    Positive = pushes the prediction toward heart disease, negative = away from it.
    """
    global _explainer
    import shap  # imported lazily so the app still starts if shap is missing

    model = load_model()
    if _explainer is None:
        _explainer = shap.TreeExplainer(model)
    row = pd.DataFrame([features])[FEATURES]
    values = _explainer.shap_values(row)
    contrib = pd.Series(values[0], index=FEATURES)
    return contrib.reindex(contrib.abs().sort_values(ascending=False).index)
