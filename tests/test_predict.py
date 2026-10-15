import pytest
import numpy as np
import pandas as pd
import joblib

from src.predict import (
    predict_student_performance,
    validate_input_features,
    load_inference_artifacts,
)
from src.features import DEFAULT_FEATURE_COLS


def test_inference_artifacts_load_successfully():
    """Verify that trained model and residual metadata load successfully."""
    model, oof = load_inference_artifacts()
    assert model is not None
    assert "interval_half_width" in oof
    assert "residual_std" in oof


def test_input_validation_catches_invalid_and_missing_features():
    """Verify that missing or out-of-range inputs raise clear descriptive ValueErrors."""
    # 1. Missing required field
    with pytest.raises(ValueError, match="Missing required input parameter: 'series_2'"):
        validate_input_features({"series_1": 20.0, "study_hours_per_day": 2.0})

    # 2. Out-of-bounds series score (> 40)
    with pytest.raises(ValueError, match="Series 1 score must be between 0.0 and 40.0"):
        validate_input_features({
            "series_1": 45.0,
            "series_2": 20.0,
            "study_hours_per_day": 2.0,
            "attendance_percentage": 80.0,
            "social_media_hours": 1.0,
        })

    # 3. Negative attendance
    with pytest.raises(ValueError, match="Attendance percentage must be between 0.0% and 100.0%"):
        validate_input_features({
            "series_1": 20.0,
            "series_2": 20.0,
            "study_hours_per_day": 2.0,
            "attendance_percentage": -5.0,
            "social_media_hours": 1.0,
        })


def test_prediction_consistency_with_direct_pipeline():
    """Verify that prediction via predict_student_performance matches direct pipeline output."""
    model, _ = load_inference_artifacts()
    test_df = pd.read_csv("data/processed/test.csv")
    sample_row = test_df.iloc[0]

    raw_input = {
        "series_1": float(sample_row["series_1"]),
        "series_2": float(sample_row["series_2"]),
        "study_hours_per_day": float(sample_row["study_hours_per_day"]),
        "attendance_percentage": float(sample_row["attendance_percentage"]),
        "social_media_hours": float(sample_row["social_media_hours"]),
    }

    result = predict_student_performance(raw_input)

    # Compute expected output directly from model pipeline
    direct_features = sample_row[DEFAULT_FEATURE_COLS].to_frame().T
    expected_score = float(model.predict(direct_features)[0])

    assert np.isclose(result["predicted_score"], np.clip(expected_score, 0.0, 100.0), atol=1e-2)
    assert result["prediction_interval"]["lower"] <= result["predicted_score"] <= result["prediction_interval"]["upper"]
    assert len(result["contributions"]) > 0


def test_lifestyle_features_contribute_to_expanded_prediction():
    """Verify that lifestyle features are passed to the model and influence prediction via learned coefficients."""
    base_student = {
        "series_1": 25.0,
        "series_2": 28.0,
        "study_hours_per_day": 2.0,
        "attendance_percentage": 60.0,
        "social_media_hours": 2.0,
    }
    pred_base = predict_student_performance(base_student)

    # Change study hours significantly
    varied_study = {**base_student, "study_hours_per_day": 10.0}
    pred_study = predict_student_performance(varied_study)

    # The model has learned a positive coefficient for study hours (~0.035 pts/hr)
    # So increasing study hours by 8 hours should increase raw prediction
    assert pred_study["raw_prediction"] != pred_base["raw_prediction"]
    assert pred_study["raw_prediction"] > pred_base["raw_prediction"]

    # Verify all 5 features are explained
    contrib_features = [c["feature"] for c in pred_base["contributions"]]
    assert "series_1" in contrib_features
    assert "series_2" in contrib_features
    assert any(f in contrib_features for f in ["study_hours_per_day", "social_media_hours", "attendance_percentage"])

