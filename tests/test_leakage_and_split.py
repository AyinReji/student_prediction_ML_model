import numpy as np
import pandas as pd

from src.data import load_raw_data, split_and_save_data
from src.features import DEFAULT_FEATURE_COLS, PROTECTED_COLS, TARGET_COL


def test_target_not_in_features():
    """Hard requirement: final_exam_score is only the target, never an input."""
    assert TARGET_COL not in DEFAULT_FEATURE_COLS
    for col in DEFAULT_FEATURE_COLS:
        assert "final" not in col.lower()


def test_protected_attributes_excluded_by_default():
    """Hard requirement: age and gender are not model inputs by default."""
    for p in PROTECTED_COLS:
        assert p not in DEFAULT_FEATURE_COLS


def test_train_test_split_ratio_and_disjoint():
    """Hard requirement: 80% train / 20% test split, disjoint, reproducible."""
    df = load_raw_data()
    train_df, test_df = split_and_save_data(df)

    assert len(train_df) == 8000
    assert len(test_df) == 2000

    # Ensure completely disjoint index sets
    train_ids = set(train_df["student_id"])
    test_ids = set(test_df["student_id"])
    assert len(train_ids.intersection(test_ids)) == 0


def test_preprocessing_fitted_on_train_only():
    """Hard requirement: Preprocessing fitted on train data only, test set never seen in fit."""
    import joblib
    model = joblib.load("models/best_model.joblib")
    scaler = model.named_steps.get("scaler")
    if scaler is not None:
        train_df = pd.read_csv("data/processed/train.csv")
        X_train = train_df[DEFAULT_FEATURE_COLS]
        # Compare fitted mean with train mean
        np.testing.assert_allclose(scaler.mean_, X_train.mean().values, rtol=1e-3)
