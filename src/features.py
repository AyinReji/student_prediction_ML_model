import pandas as pd

DEFAULT_FEATURE_COLS = [
    "study_hours_per_day",
    "attendance_percentage",
    "social_media_hours",
    "series_1",
    "series_2",
    "series_avg",
    "series_delta",
]

ABLATION_FEATURE_COLS = DEFAULT_FEATURE_COLS + ["age", "gender"]

TARGET_COL = "final_exam_score"
PROTECTED_COLS = ["age", "gender"]
METADATA_COLS = ["student_id", "age", "gender"]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds domain features without modifying the original dataframe in place.
    - series_avg: average of series 1 and series 2 (out of 40)
    - series_delta: trajectory between series 1 and 2 (series_2 - series_1)
    """
    df_out = df.copy()
    if "series_1" in df_out.columns and "series_2" in df_out.columns:
        df_out["series_avg"] = (df_out["series_1"] + df_out["series_2"]) / 2.0
        df_out["series_delta"] = df_out["series_2"] - df_out["series_1"]
    return df_out
