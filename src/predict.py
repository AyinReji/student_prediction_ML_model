import json
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd

from src.config import Settings, load_settings
from src.features import DEFAULT_FEATURE_COLS, LIFESTYLE_FEATURES, REGRESSION_FEATURES
from src.priority import calculate_p_risk, get_risk_badge
from src.explain import ModelExplainer
from src.recommend import evaluate_student_rules

MODEL_PATH = Path("models/best_model.joblib")
OOF_RESIDUALS_PATH = Path("models/oof_residuals.json")


def load_inference_artifacts() -> Tuple[Any, Dict[str, Any]]:
    """Loads and caches the trained model and out-of-fold residual metadata."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run python -m src.train first.")
    model = joblib.load(MODEL_PATH)

    if OOF_RESIDUALS_PATH.exists():
        with open(OOF_RESIDUALS_PATH, "r", encoding="utf-8") as f:
            oof_meta = json.load(f)
    else:
        oof_meta = {
            "residual_std": 6.5338,
            "interval_half_width": 11.1229,
            "interval_level": 0.90,
        }

    return model, oof_meta


def validate_input_features(features: Dict[str, Any]) -> Dict[str, float]:
    """Validates and casts required model features according to domain bounds.
    Raises ValueError with descriptive messaging if validation fails.
    """
    required_fields = [
        "series_1",
        "series_2",
        "study_hours_per_day",
        "attendance_percentage",
        "social_media_hours",
    ]

    clean_features: Dict[str, float] = {}

    for field in required_fields:
        if field not in features or features[field] is None:
            raise ValueError(f"Missing required input parameter: '{field}'")
        try:
            val = float(features[field])
        except (ValueError, TypeError):
            raise ValueError(f"Field '{field}' must be a valid numeric value.")
        clean_features[field] = val

    # Range validations
    if not (0.0 <= clean_features["series_1"] <= 40.0):
        raise ValueError(f"Series 1 score must be between 0.0 and 40.0 (received {clean_features['series_1']})")

    if not (0.0 <= clean_features["series_2"] <= 40.0):
        raise ValueError(f"Series 2 score must be between 0.0 and 40.0 (received {clean_features['series_2']})")

    if not (0.0 <= clean_features["attendance_percentage"] <= 100.0):
        raise ValueError(f"Attendance percentage must be between 0.0% and 100.0% (received {clean_features['attendance_percentage']}%)")

    if not (0.0 <= clean_features["study_hours_per_day"] <= 24.0):
        raise ValueError(f"Daily study hours must be between 0.0 and 24.0 (received {clean_features['study_hours_per_day']})")

    if not (0.0 <= clean_features["social_media_hours"] <= 24.0):
        raise ValueError(f"Daily social media hours must be between 0.0 and 24.0 (received {clean_features['social_media_hours']})")

    # Add domain engineered features
    s1 = clean_features["series_1"]
    s2 = clean_features["series_2"]
    clean_features["series_avg"] = (s1 + s2) / 2.0
    clean_features["series_delta"] = s2 - s1

    return clean_features


def predict_student_performance(
    raw_inputs: Dict[str, Any],
    model: Any = None,
    oof_meta: Dict[str, Any] = None,
    settings: Settings = None,
) -> Dict[str, Any]:
    """Complete inference pipeline:
    1. Validates inputs and creates domain features (series_avg, series_delta)
    2. Executes inference through trained pipeline (StandardScaler + LinearRegression)
    3. Calculates 90% prediction interval from OOF residuals
    4. Computes risk probability and risk badge
    5. Computes score-point factor attribution
    6. Evaluates pedagogical recommendation rules
    """
    if settings is None:
        settings = load_settings()

    if model is None or oof_meta is None:
        m, oof = load_inference_artifacts()
        if model is None:
            model = m
        if oof_meta is None:
            oof_meta = oof

    # 1. Validation and feature preparation
    clean_dict = validate_input_features(raw_inputs)
    input_row = pd.Series(clean_dict)
    # Regression model receives authorized predictive features (series_1, series_2, lifestyle attributes)
    input_df = pd.DataFrame([clean_dict])[REGRESSION_FEATURES]

    # 2. Model prediction
    raw_pred = float(model.predict(input_df)[0])
    bounded_pred = round(float(np.clip(raw_pred, 0.0, 100.0)), 2)

    # 3. 90% Prediction Interval
    half_width = float(oof_meta.get("interval_half_width", 11.1229))
    interval_lower = round(float(np.clip(bounded_pred - half_width, 0.0, 100.0)), 2)
    interval_upper = round(float(np.clip(bounded_pred + half_width, 0.0, 100.0)), 2)

    # 4. Risk Assessment
    residual_std = float(oof_meta.get("residual_std", 6.5338))
    p_risk = float(calculate_p_risk(np.array([bounded_pred]), settings.target_score, residual_std)[0])
    risk_badge = get_risk_badge(p_risk, settings.risk_badges.high, settings.risk_badges.medium)
    is_at_risk = bounded_pred < (settings.target_score + settings.risk_margin)

    # 5. Explainability (computed across academic and lifestyle features)
    explainer = ModelExplainer(model=model, feature_names=REGRESSION_FEATURES)
    contributions = explainer.explain_student(input_row, top_k=5)

    # 6. Recommendation rules (evaluated for personalized support recommendations)
    rules_fired = evaluate_student_rules(input_row, settings)

    return {
        "success": True,
        "predicted_score": bounded_pred,
        "raw_prediction": round(raw_pred, 4),
        "prediction_interval": {
            "level": oof_meta.get("interval_level", 0.90),
            "lower": interval_lower,
            "upper": interval_upper,
            "half_width": round(half_width, 2),
        },
        "risk_evaluation": {
            "target_score": settings.target_score,
            "risk_margin": settings.risk_margin,
            "is_at_risk": is_at_risk,
            "risk_probability": round(p_risk, 4),
            "risk_probability_pct": round(p_risk * 100, 1),
            "risk_badge": risk_badge,
        },
        "contributions": contributions,
        "rules_fired": rules_fired,
        "features": clean_dict,
    }
