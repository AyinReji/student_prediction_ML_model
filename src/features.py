import pandas as pd

# ──────────────────────────────────────────────────────────────────────
# REGRESSION FEATURES — authorized predictive inputs for the final
# examination score model.  These are the ONLY features permitted
# in the regression pipeline, its preprocessing, and its prediction
# interval calculations.
# ──────────────────────────────────────────────────────────────────────
REGRESSION_FEATURES = [
    "series_1",
    "series_2",
]

# ──────────────────────────────────────────────────────────────────────
# LIFESTYLE FEATURES — available for deterministic support rules,
# teacher conversations, follow-up planning, and contextual notes.
# These MUST NOT enter the regression model or influence the
# predicted final examination mark in any way.
# ──────────────────────────────────────────────────────────────────────
LIFESTYLE_FEATURES = [
    "study_hours_per_day",
    "attendance_percentage",
    "social_media_hours",
]

# Default feature set for regression models — ONLY series_1 and series_2
DEFAULT_FEATURE_COLS = REGRESSION_FEATURES

# Legacy alias — kept ONLY for backward-compatible rule evaluation,
# priority-queue flag counting, and UI display of student profiles.
# Must NOT be passed to model.fit() or model.predict().
ALL_STUDENT_FEATURES = REGRESSION_FEATURES + LIFESTYLE_FEATURES

# Ablation study features (includes demographics for fairness comparison)
ABLATION_FEATURE_COLS = REGRESSION_FEATURES + LIFESTYLE_FEATURES + ["age", "gender"]

TARGET_COL = "final_exam_score"
PROTECTED_COLS = ["age", "gender"]
METADATA_COLS = ["student_id", "age", "gender"]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds domain features without modifying the original dataframe in place.
    - series_avg: average of series 1 and series 2 (out of 40)
    - series_delta: trajectory between series 1 and 2 (series_2 - series_1)

    Note: series_avg and series_delta are derived for rule evaluation and
    UI display only.  The regression model uses raw series_1 and series_2.
    """
    df_out = df.copy()
    if "series_1" in df_out.columns and "series_2" in df_out.columns:
        df_out["series_avg"] = (df_out["series_1"] + df_out["series_2"]) / 2.0
        df_out["series_delta"] = df_out["series_2"] - df_out["series_1"]
    return df_out
