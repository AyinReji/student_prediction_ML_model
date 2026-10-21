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

DEMOGRAPHIC_FEATURES = [
    "age",
    "gender",
]

NUMERIC_FEATURES = [
    "series_1",
    "series_2",
    "study_hours_per_day",
    "attendance_percentage",
    "social_media_hours",
    "age",
]

CATEGORICAL_FEATURES = [
    "gender",
]

# Complete feature vector incorporating every feature of the dataset
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
EXPANDED_FEATURES = ALL_FEATURES

# Aliases ensuring backward and forward compatibility across training & inference
REGRESSION_FEATURES = ALL_FEATURES
DEFAULT_FEATURE_COLS = ALL_FEATURES
ALL_STUDENT_FEATURES = ALL_FEATURES

# Ablation study features
ABLATION_FEATURE_COLS = ALL_FEATURES

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
