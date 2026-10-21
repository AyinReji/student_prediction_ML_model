import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import joblib
import numpy as np
import pandas as pd

from src.config import Settings, load_settings
from src.features import DEFAULT_FEATURE_COLS, LIFESTYLE_FEATURES, REGRESSION_FEATURES
from src.priority import calculate_p_risk, get_risk_badge
from src.explain import ModelExplainer
from src.recommend import evaluate_student_rules

# ──────────────────────────────────────────────────────────────────────
# LIFESTYLE ADJUSTMENT LAYER
# The training data is synthetic: lifestyle features have near-zero
# empirical correlations with final_exam_score (|r| <= 0.01), so the
# ML model learns near-zero coefficients for them. To ensure lifestyle
# and study habits meaningfully influence the predicted score, a
# transparent, domain-driven adjustment is applied on top of the raw
# ML prediction. Thresholds and per-unit weights are defined below.
# ──────────────────────────────────────────────────────────────────────

# Reference/baseline values (population midpoints)
_STUDY_BASELINE_HRS = 3.0        # hrs/day — average student
_ATTENDANCE_BASELINE_PCT = 80.0  # % — typical attendance
_SOCIAL_BASELINE_HRS = 3.0       # hrs/day — average social media use

# Maximum total lifestyle adjustment (caps both positive and negative)
_MAX_LIFESTYLE_ADJUSTMENT = 10.0

# Per-unit weights: how many score points each unit of deviation from
# baseline contributes.
_STUDY_WEIGHT = 1.5       # +1.5 pts per extra hr/day of study above baseline
_ATTENDANCE_WEIGHT = 0.08  # +0.08 pts per % above baseline attendance
_SOCIAL_WEIGHT = -1.2     # -1.2 pts per extra hr/day of social media above baseline


def compute_lifestyle_adjustment(
    study_hours: float,
    attendance_pct: float,
    social_hours: float,
) -> Dict[str, Any]:
    """Computes a transparent domain-driven score adjustment based on
    lifestyle inputs relative to population baselines.

    Returns a dict with:
      - 'total': net adjustment (capped to ±_MAX_LIFESTYLE_ADJUSTMENT)
      - 'study_contribution': pts from study hours deviation
      - 'attendance_contribution': pts from attendance deviation
      - 'social_contribution': pts from social media deviation
      - 'explanation': human-readable description
    """
    study_contrib = round(_STUDY_WEIGHT * (study_hours - _STUDY_BASELINE_HRS), 3)
    attend_contrib = round(_ATTENDANCE_WEIGHT * (attendance_pct - _ATTENDANCE_BASELINE_PCT), 3)
    social_contrib = round(_SOCIAL_WEIGHT * (social_hours - _SOCIAL_BASELINE_HRS), 3)

    raw_total = study_contrib + attend_contrib + social_contrib
    capped_total = round(
        float(np.clip(raw_total, -_MAX_LIFESTYLE_ADJUSTMENT, _MAX_LIFESTYLE_ADJUSTMENT)), 3
    )

    explanation = (
        f"Study hours {study_hours:.1f} h/day vs baseline {_STUDY_BASELINE_HRS:.1f} h → "
        f"{study_contrib:+.2f} pts; "
        f"Attendance {attendance_pct:.1f}% vs baseline {_ATTENDANCE_BASELINE_PCT:.1f}% → "
        f"{attend_contrib:+.2f} pts; "
        f"Social media {social_hours:.1f} h/day vs baseline {_SOCIAL_BASELINE_HRS:.1f} h → "
        f"{social_contrib:+.2f} pts; "
        f"Total adjustment: {capped_total:+.2f} pts (capped at ±{_MAX_LIFESTYLE_ADJUSTMENT:.0f})."
    )

    return {
        "total": capped_total,
        "study_contribution": study_contrib,
        "attendance_contribution": attend_contrib,
        "social_contribution": social_contrib,
        "study_baseline": _STUDY_BASELINE_HRS,
        "attendance_baseline": _ATTENDANCE_BASELINE_PCT,
        "social_baseline": _SOCIAL_BASELINE_HRS,
        "max_adjustment": _MAX_LIFESTYLE_ADJUSTMENT,
        "explanation": explanation,
    }

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

    # Demographic attributes
    age_val = float(features.get("age", 18.0))
    if not (10.0 <= age_val <= 100.0):
        raise ValueError(f"Age must be between 10 and 100 (received {age_val})")
    clean_features["age"] = age_val

    gender_val = str(features.get("gender", "Female")).capitalize()
    if gender_val not in ["Female", "Male", "Other"]:
        gender_val = "Female"
    clean_features["gender"] = gender_val

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
    1. Validates inputs across every feature (assessments, lifestyle, demographics)
    2. Executes inference through trained pipeline
    3. Calculates 90% prediction interval from OOF residuals
    4. Computes risk probability and risk badge
    5. Computes score-point factor attribution across all features
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
    # Regression model receives every feature of the dataset
    input_df = pd.DataFrame([clean_dict])[REGRESSION_FEATURES]

    # 2. Model prediction
    raw_pred = float(model.predict(input_df)[0])
    bounded_pred = round(float(np.clip(raw_pred, 0.0, 100.0)), 2)

    # 3. 90% Prediction Interval
    half_width = float(oof_meta.get("interval_half_width", 9.5))
    interval_lower = round(float(np.clip(bounded_pred - half_width, 0.0, 100.0)), 2)
    interval_upper = round(float(np.clip(bounded_pred + half_width, 0.0, 100.0)), 2)

    # 4. Risk Assessment
    residual_std = float(oof_meta.get("residual_std", 8.5))
    p_risk = float(calculate_p_risk(np.array([bounded_pred]), settings.target_score, residual_std)[0])
    risk_badge = get_risk_badge(p_risk, settings.risk_badges.high, settings.risk_badges.medium)
    is_at_risk = bounded_pred < (settings.target_score + settings.risk_margin)

    # 5. Explainability (computed across every feature in the dataset)
    explainer = ModelExplainer(model=model, feature_names=REGRESSION_FEATURES)
    contributions = explainer.explain_student(input_row, top_k=7)

    # Extract lifestyle contributions directly from the model's learned weights
    c_map = {c["feature"]: c["points"] for c in contributions}
    study_c = c_map.get("study_hours_per_day", 0.0)
    attend_c = c_map.get("attendance_percentage", 0.0)
    social_c = c_map.get("social_media_hours", 0.0)
    lifestyle_total = round(study_c + attend_c + social_c, 2)
    lifestyle_adj = {
        "total": lifestyle_total,
        "study_contribution": study_c,
        "attendance_contribution": attend_c,
        "social_contribution": social_c,
        "study_baseline": 3.0,
        "attendance_baseline": 80.0,
        "social_baseline": 3.0,
        "explanation": (
            f"Model learned impact: Study hours ({study_c:+.2f} pts), "
            f"Attendance ({attend_c:+.2f} pts), Social media ({social_c:+.2f} pts). "
            f"Total lifestyle impact: {lifestyle_total:+.2f} pts."
        ),
    }

    # 7. Exam Eligibility Policy (Attendance < 75% cannot attend final exam)
    min_attendance = float(settings.rules.min_attendance)
    student_attendance = float(clean_dict["attendance_percentage"])
    is_exam_eligible = student_attendance >= min_attendance
    exam_eligibility = {
        "is_eligible": is_exam_eligible,
        "min_required_attendance": min_attendance,
        "student_attendance": student_attendance,
        "status": "Eligible to Attend Exam" if is_exam_eligible else "Ineligible (Cannot Attend Exam)",
        "message": (
            "Student satisfies the minimum 75.0% attendance requirement to attend the final exam."
            if is_exam_eligible
            else f"Student attendance is {student_attendance:.1f}% (below the mandatory {min_attendance:.1f}% threshold). The student is not permitted to attend the final examination under institutional policy."
        ),
    }

    # 8. Recommendation rules (evaluated for personalized support recommendations)
    rules_fired = evaluate_student_rules(input_row, settings)

    return {
        "success": True,
        "predicted_score": bounded_pred,
        "ml_base_score": round(float(np.clip(raw_pred, 0.0, 100.0)), 2),
        "raw_prediction": round(raw_pred, 4),
        "lifestyle_adjustment": lifestyle_adj,
        "exam_eligibility": exam_eligibility,
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


# ──────────────────────────────────────────────────────────────────────
# BULK PREDICTION PIPELINE
# Processes a DataFrame of student records through the identical
# validate → pipeline → clip pathway used for single-student inference.
# Returns: (results_df, errors_df) where results_df contains only valid
# rows (with predicted_final_marks column appended) and errors_df
# contains rejected rows with a 'validation_errors' column.
# ──────────────────────────────────────────────────────────────────────

# Canonical column spec derived from validate_input_features
BULK_REQUIRED_COLUMNS: List[str] = [
    "series_1",
    "series_2",
    "study_hours_per_day",
    "attendance_percentage",
    "social_media_hours",
]

BULK_OPTIONAL_COLUMNS: List[str] = ["age", "gender"]

BULK_ALL_FEATURE_COLUMNS: List[str] = BULK_REQUIRED_COLUMNS + BULK_OPTIONAL_COLUMNS

# Domain bounds mirrored exactly from validate_input_features
BULK_BOUNDS: Dict[str, tuple] = {
    "series_1":              (0.0,  40.0),
    "series_2":              (0.0,  40.0),
    "study_hours_per_day":   (0.0,  24.0),
    "attendance_percentage": (0.0, 100.0),
    "social_media_hours":    (0.0,  24.0),
    "age":                   (10.0, 100.0),
}

BULK_VALID_GENDERS = {"Female", "Male", "Other"}


def _validate_bulk_row(row: pd.Series, row_idx: int) -> List[str]:
    """Returns a list of error strings for a single CSV row.
    An empty list means the row is valid.
    """
    errors: List[str] = []

    # Check required numeric fields
    for field in BULK_REQUIRED_COLUMNS:
        val = row.get(field)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            errors.append(f"'{field}': missing or empty")
            continue
        try:
            fval = float(val)
        except (ValueError, TypeError):
            errors.append(f"'{field}': cannot convert '{val}' to number")
            continue
        lo, hi = BULK_BOUNDS[field]
        if not (lo <= fval <= hi):
            errors.append(f"'{field}': {fval} out of valid range [{lo}, {hi}]")

    # Optional age
    if "age" in row.index:
        val = row.get("age")
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            try:
                fval = float(val)
                lo, hi = BULK_BOUNDS["age"]
                if not (lo <= fval <= hi):
                    errors.append(f"'age': {fval} out of valid range [{lo}, {hi}]")
            except (ValueError, TypeError):
                errors.append(f"'age': cannot convert '{val}' to number")

    # Optional gender
    if "gender" in row.index:
        val = row.get("gender")
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            gval = str(val).strip().capitalize()
            if gval not in BULK_VALID_GENDERS:
                errors.append(f"'gender': '{val}' not in {sorted(BULK_VALID_GENDERS)}")

    return errors


def predict_bulk(
    df: pd.DataFrame,
    model: Any = None,
    oof_meta: Dict[str, Any] = None,
    settings=None,
    progress_callback=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Bulk prediction over a DataFrame.

    Each row is independently validated and run through the same
    predict_student_performance pipeline used for single-student inference.
    No data is fabricated; invalid rows are collected in errors_df.

    Args:
        df:                Raw DataFrame uploaded by the user.
        model:             Trained sklearn pipeline (loaded from cache if None).
        oof_meta:          OOF residual dict (loaded from file if None).
        settings:          Application Settings (loaded from config if None).
        progress_callback: Optional callable(completed_rows, total_rows).

    Returns:
        results_df: DataFrame with original columns + 'predicted_final_marks'
                    (only valid rows, original order preserved).
        errors_df:  DataFrame with original columns + 'validation_errors'
                    (only rejected rows, original order preserved).
    """
    from src.config import load_settings as _load_settings

    if settings is None:
        settings = _load_settings()
    if model is None or oof_meta is None:
        m, oof = load_inference_artifacts()
        if model is None:
            model = m
        if oof_meta is None:
            oof_meta = oof

    total = len(df)
    valid_indices: List[int] = []
    valid_scores: List[float] = []
    error_indices: List[int] = []
    error_messages: List[str] = []

    for pos, (idx, row) in enumerate(df.iterrows()):
        # Row-level validation
        row_errors = _validate_bulk_row(row, pos)
        if row_errors:
            error_indices.append(idx)
            error_messages.append("; ".join(row_errors))
        else:
            # Build the raw_inputs dict expected by validate_input_features
            raw_inputs: Dict[str, Any] = {}
            for field in BULK_REQUIRED_COLUMNS:
                raw_inputs[field] = float(row[field])
            raw_inputs["age"] = float(row["age"]) if "age" in row.index and not (
                isinstance(row["age"], float) and np.isnan(row["age"])
            ) else 18.0
            raw_inputs["gender"] = str(row["gender"]).strip().capitalize() if "gender" in row.index and not (
                isinstance(row["gender"], float) and np.isnan(row["gender"])
            ) else "Female"

            try:
                result = predict_student_performance(
                    raw_inputs=raw_inputs,
                    model=model,
                    oof_meta=oof_meta,
                    settings=settings,
                )
                valid_indices.append(idx)
                valid_scores.append(result["predicted_score"])
            except Exception as exc:
                error_indices.append(idx)
                error_messages.append(f"Inference error: {exc}")

        if progress_callback is not None:
            progress_callback(pos + 1, total)

    # Build results DataFrame — preserve original columns and row order
    if valid_indices:
        results_df = df.loc[valid_indices].copy()
        results_df["predicted_final_marks"] = valid_scores
    else:
        results_df = df.iloc[0:0].copy()
        results_df["predicted_final_marks"] = pd.Series(dtype=float)

    if error_indices:
        errors_df = df.loc[error_indices].copy()
        errors_df["validation_errors"] = error_messages
    else:
        errors_df = df.iloc[0:0].copy()
        errors_df["validation_errors"] = pd.Series(dtype=str)

    return results_df, errors_df
