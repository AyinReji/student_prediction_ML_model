import numpy as np
import pandas as pd

from src.data import load_raw_data, split_and_save_data
from src.features import DEFAULT_FEATURE_COLS, PROTECTED_COLS, TARGET_COL


def test_target_not_in_features():
    """Hard requirement: final_exam_score is only the target, never an input."""
    assert TARGET_COL not in DEFAULT_FEATURE_COLS
    for col in DEFAULT_FEATURE_COLS:
        assert "final" not in col.lower()


def test_all_features_included_in_model():
    """Hard requirement: Every feature of the dataset is included in model inputs."""
    for f in ["series_1", "series_2", "study_hours_per_day", "attendance_percentage", "social_media_hours", "age", "gender"]:
        assert f in DEFAULT_FEATURE_COLS


def test_train_test_split_ratio_and_disjoint():
    """Hard requirement: 80% train / 20% test split, disjoint, reproducible."""
    df = load_raw_data()
    train_df, test_df = split_and_save_data(df)

    assert len(train_df) == int(len(df) * 0.8)
    assert len(test_df) == int(len(df) * 0.2)

    # Ensure completely disjoint index sets
    train_ids = set(train_df["student_id"])
    test_ids = set(test_df["student_id"])
    assert len(train_ids.intersection(test_ids)) == 0


def test_preprocessing_fitted_on_train_only():
    """Hard requirement: Preprocessing fitted on train data only, test set never seen in fit."""
    import joblib
    from src.features import NUMERIC_FEATURES
    model = joblib.load("models/best_model.joblib")
    prep = model.named_steps.get("prep")
    if prep is not None:
        num_scaler = prep.named_transformers_["num"]
        train_df = pd.read_csv("data/processed/train.csv")
        X_num = train_df[NUMERIC_FEATURES]
        np.testing.assert_allclose(num_scaler.mean_, X_num.mean().values, rtol=1e-3)
