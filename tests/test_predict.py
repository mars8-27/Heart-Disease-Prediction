import numpy as np
import pandas as pd
import pytest

import predict


class StubModel:
    def predict_proba(self, frame):
        assert list(frame.columns) == predict.FEATURES
        assert len(frame) == 1
        return np.array([[0.2, 0.8]])


def test_predict_returns_class_and_probability(monkeypatch, patient_features):
    monkeypatch.setattr(predict, "_model", StubModel())

    prediction, probability = predict.predict(patient_features)

    assert prediction == 1
    assert probability == pytest.approx(0.8)


def test_predict_thresholds_at_0_5(monkeypatch, patient_features):
    class BoundaryModel:
        def predict_proba(self, frame):
            return np.array([[0.5, 0.5]])

    monkeypatch.setattr(predict, "_model", BoundaryModel())

    prediction, probability = predict.predict(patient_features)

    assert prediction == 1
    assert probability == pytest.approx(0.5)


def test_missing_model_raises_clear_error(monkeypatch):
    monkeypatch.setattr(predict, "_model", None)
    monkeypatch.setattr(predict, "MODEL_PATH", pd.io.common.stringify_path("missing-model.joblib"))

    with pytest.raises(FileNotFoundError, match="Model not found"):
        predict.load_model()
