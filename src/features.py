import pandas as pd

# ──────────────────────────────────────────────────────────────────────
# PREDICTIVE POLICY:
# "Final examination marks are predicted using Series 1, Series 2, and
# relevant lifestyle attributes. Lifestyle attributes may contribute to
# predictions when supported by the training data. These attributes can
# also inform personalized academic support recommendations."
# ──────────────────────────────────────────────────────────────────────

PREDICTIVE_POLICY = (
    "Final examination marks are predicted using Series 1, Series 2, and "
    "relevant lifestyle attributes. Lifestyle attributes may contribute to "
    "predictions when supported by the training data. These attributes can "
    "also inform personalized academic support recommendations."
)

ACADEMIC_FEATURES = [
    "series_1",
    "series_2",
]

LIFESTYLE_FEATURES = [
    "study_hours_per_day",
    "attendance_percentage",
    "social_media_hours",
]

# Complete authorized feature vector for the expanded regression model
EXPANDED_FEATURES = ACADEMIC_FEATURES + LIFESTYLE_FEATURES

# Aliases ensuring backward and forward compatibility across training & inference
REGRESSION_FEATURES = EXPANDED_FEATURES
DEFAULT_FEATURE_COLS = EXPANDED_FEATURES
ALL_STUDENT_FEATURES = EXPANDED_FEATURES

# Ablation study features (includes demographics for fairness comparison)
ABLATION_FEATURE_COLS = EXPANDED_FEATURES + ["age", "gender"]

TARGET_COL = "final_exam_score"
PROTECTED_COLS = ["age", "gender"]
METADATA_COLS = ["student_id", "age", "gender"]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds domain features without modifying the original dataframe in place.
    - series_avg: average of series 1 and series 2 (out of 40)
    - series_delta: trajectory between series 1 and 2 (series_2 - series_1)

    Note: series_avg and series_delta are derived for rule evaluation and
    UI display. Raw series_1 and series_2 are used as model inputs.
    """
    df_out = df.copy()
    if "series_1" in df_out.columns and "series_2" in df_out.columns:
        df_out["series_avg"] = (df_out["series_1"] + df_out["series_2"]) / 2.0
        df_out["series_delta"] = df_out["series_2"] - df_out["series_1"]
    return df_out
