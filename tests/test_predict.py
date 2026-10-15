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


def test_lifestyle_features_have_zero_impact_on_predicted_score():
    """Hard requirement: changing lifestyle features must have ZERO effect on predicted final exam mark."""
    base_student = {
        "series_1": 25.0,
        "series_2": 28.0,
        "study_hours_per_day": 2.0,
        "attendance_percentage": 60.0,
        "social_media_hours": 6.0,
    }
    pred_base = predict_student_performance(base_student)

    # Vary study hours dramatically
    varied_student_1 = {**base_student, "study_hours_per_day": 12.0}
    pred_varied_1 = predict_student_performance(varied_student_1)

    # Vary attendance dramatically
    varied_student_2 = {**base_student, "attendance_percentage": 98.0}
    pred_varied_2 = predict_student_performance(varied_student_2)

    # Vary social media hours dramatically
    varied_student_3 = {**base_student, "social_media_hours": 0.5}
    pred_varied_3 = predict_student_performance(varied_student_3)

    # All predicted scores and raw predictions must be EXACTLY identical
    assert pred_base["predicted_score"] == pred_varied_1["predicted_score"]
    assert pred_base["predicted_score"] == pred_varied_2["predicted_score"]
    assert pred_base["predicted_score"] == pred_varied_3["predicted_score"]
    assert pred_base["raw_prediction"] == pred_varied_1["raw_prediction"]
    assert pred_base["raw_prediction"] == pred_varied_2["raw_prediction"]
    assert pred_base["raw_prediction"] == pred_varied_3["raw_prediction"]

