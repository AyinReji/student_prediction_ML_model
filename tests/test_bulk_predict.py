"""
tests/test_bulk_predict.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Tests for the bulk CSV prediction pipeline added to src/predict.py.
Covers:
  - Valid CSV round-trip and consistency with single-student predictions
  - Missing required columns
  - Out-of-range numeric values
  - Non-numeric values in numeric fields
  - Invalid gender categories
  - Missing optional columns (age, gender default)
  - Partial invalid rows (mixed valid/invalid)
  - Empty DataFrame input
  - Extra pass-through columns (student_id etc.) preserved in output
  - Rejected rows carry a validation_errors column
"""

import numpy as np
import pandas as pd
import pytest

from src.predict import (
    BULK_REQUIRED_COLUMNS,
    BULK_VALID_GENDERS,
    _validate_bulk_row,
    predict_bulk,
    predict_student_performance,
    load_inference_artifacts,
)


# ── Shared fixture ─────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def artifacts():
    """Load model + OOF once for all tests in this module."""
    model, oof = load_inference_artifacts()
    return model, oof


def _make_valid_row(**overrides):
    """Return a dict representing a perfectly valid student row."""
    base = {
        "series_1": 25.0,
        "series_2": 28.0,
        "study_hours_per_day": 3.0,
        "attendance_percentage": 85.0,
        "social_media_hours": 2.0,
        "age": 19.0,
        "gender": "Female",
    }
    base.update(overrides)
    return base


# ══════════════════════════════════════════════════════════════════════
# 1. _validate_bulk_row — unit tests (no model needed)
# ══════════════════════════════════════════════════════════════════════

class TestValidateBulkRow:
    def _row(self, **kw):
        return pd.Series(_make_valid_row(**kw))

    def test_valid_row_returns_no_errors(self):
        assert _validate_bulk_row(self._row(), 0) == []

    def test_missing_required_field(self):
        row = self._row()
        row = row.drop("series_1")
        errs = _validate_bulk_row(row, 0)
        assert any("series_1" in e for e in errs)

    def test_nan_required_field(self):
        row = self._row(series_2=float("nan"))
        errs = _validate_bulk_row(row, 0)
        assert any("series_2" in e for e in errs)

    def test_series_above_max(self):
        row = self._row(series_1=45.0)
        errs = _validate_bulk_row(row, 0)
        assert any("series_1" in e for e in errs)

    def test_series_below_min(self):
        row = self._row(series_2=-1.0)
        errs = _validate_bulk_row(row, 0)
        assert any("series_2" in e for e in errs)

    def test_attendance_out_of_range(self):
        row = self._row(attendance_percentage=110.0)
        errs = _validate_bulk_row(row, 0)
        assert any("attendance_percentage" in e for e in errs)

    def test_non_numeric_field(self):
        row = self._row(study_hours_per_day="abc")
        errs = _validate_bulk_row(row, 0)
        assert any("study_hours_per_day" in e for e in errs)

    def test_invalid_gender(self):
        row = self._row(gender="Unknown")
        errs = _validate_bulk_row(row, 0)
        assert any("gender" in e for e in errs)

    def test_valid_gender_case_insensitive(self):
        # 'female' should be capitalised to 'Female' by the validator
        row = self._row(gender="male")
        # _validate_bulk_row capitalises before checking
        errs = _validate_bulk_row(row, 0)
        assert errs == []

    def test_age_out_of_range(self):
        row = self._row(age=5.0)
        errs = _validate_bulk_row(row, 0)
        assert any("age" in e for e in errs)

    def test_multiple_errors_reported_together(self):
        row = self._row(series_1=99.0, attendance_percentage=200.0)
        errs = _validate_bulk_row(row, 0)
        assert len(errs) >= 2

    def test_missing_optional_columns_no_error(self):
        row = pd.Series({
            "series_1": 20.0,
            "series_2": 22.0,
            "study_hours_per_day": 2.0,
            "attendance_percentage": 80.0,
            "social_media_hours": 1.5,
        })
        assert _validate_bulk_row(row, 0) == []


# ══════════════════════════════════════════════════════════════════════
# 2. predict_bulk — integration tests (uses model)
# ══════════════════════════════════════════════════════════════════════

class TestPredictBulk:

    def test_all_valid_rows_no_errors(self, artifacts):
        model, oof = artifacts
        df = pd.DataFrame([_make_valid_row() for _ in range(5)])
        results, errors = predict_bulk(df, model=model, oof_meta=oof)
        assert len(results) == 5
        assert len(errors) == 0
        assert "predicted_final_marks" in results.columns

    def test_predicted_marks_in_valid_range(self, artifacts):
        model, oof = artifacts
        df = pd.DataFrame([_make_valid_row() for _ in range(10)])
        results, _ = predict_bulk(df, model=model, oof_meta=oof)
        assert results["predicted_final_marks"].between(0.0, 100.0).all()

    def test_consistency_with_single_prediction(self, artifacts):
        """Bulk result for a row must equal single-student prediction."""
        model, oof = artifacts
        single_input = _make_valid_row(series_1=30.0, series_2=32.0, study_hours_per_day=4.0,
                                       attendance_percentage=90.0, social_media_hours=1.0,
                                       age=20.0, gender="Male")
        # Single-student prediction
        single_result = predict_student_performance(
            raw_inputs=single_input, model=model, oof_meta=oof
        )
        single_score = single_result["predicted_score"]

        # Bulk prediction of the same row
        df = pd.DataFrame([single_input])
        results, _ = predict_bulk(df, model=model, oof_meta=oof)
        bulk_score = results["predicted_final_marks"].iloc[0]

        assert abs(bulk_score - single_score) < 1e-6, (
            f"Bulk score {bulk_score} != single score {single_score}"
        )

    def test_invalid_rows_go_to_errors_df(self, artifacts):
        model, oof = artifacts
        df = pd.DataFrame([
            _make_valid_row(),                            # valid
            _make_valid_row(series_1=99.0),               # invalid: series_1 out of range
            _make_valid_row(attendance_percentage=150.0), # invalid
            _make_valid_row(),                            # valid
        ])
        results, errors = predict_bulk(df, model=model, oof_meta=oof)
        assert len(results) == 2
        assert len(errors) == 2
        assert "validation_errors" in errors.columns

    def test_row_order_preserved(self, artifacts):
        """Original DataFrame index must map correctly into results."""
        model, oof = artifacts
        rows = [_make_valid_row(series_1=float(i)) for i in range(5, 30, 5)]
        df = pd.DataFrame(rows)
        results, _ = predict_bulk(df, model=model, oof_meta=oof)
        # Scores should increase monotonically as series_1 increases
        scores = results["predicted_final_marks"].tolist()
        assert scores == sorted(scores), "Predictions should increase with series_1"

    def test_extra_columns_preserved(self, artifacts):
        """student_id and other pass-through columns must appear in results."""
        model, oof = artifacts
        rows = [_make_valid_row() for _ in range(3)]
        df = pd.DataFrame(rows)
        df.insert(0, "student_id", [101, 102, 103])
        df["class"] = "10-A"
        results, _ = predict_bulk(df, model=model, oof_meta=oof)
        assert "student_id" in results.columns
        assert "class" in results.columns
        assert list(results["student_id"]) == [101, 102, 103]

    def test_missing_optional_columns_default(self, artifacts):
        """Rows without age/gender columns should use defaults (18 / Female)."""
        model, oof = artifacts
        rows = [{k: v for k, v in _make_valid_row().items()
                 if k not in ("age", "gender")} for _ in range(3)]
        df = pd.DataFrame(rows)
        results, errors = predict_bulk(df, model=model, oof_meta=oof)
        assert len(errors) == 0
        assert len(results) == 3

    def test_empty_dataframe_returns_empty_results(self, artifacts):
        model, oof = artifacts
        df = pd.DataFrame(columns=BULK_REQUIRED_COLUMNS)
        results, errors = predict_bulk(df, model=model, oof_meta=oof)
        assert len(results) == 0
        assert len(errors) == 0

    def test_missing_required_column_raises_no_crash(self, artifacts):
        """Rows missing a required column should all land in errors_df."""
        model, oof = artifacts
        df = pd.DataFrame([{"series_1": 20.0, "series_2": 22.0,
                             "study_hours_per_day": 3.0,
                             # attendance_percentage intentionally absent from this row
                             "social_media_hours": 2.0}])
        results, errors = predict_bulk(df, model=model, oof_meta=oof)
        assert len(results) == 0
        assert len(errors) == 1

    def test_progress_callback_called(self, artifacts):
        model, oof = artifacts
        df = pd.DataFrame([_make_valid_row() for _ in range(4)])
        calls = []

        def cb(done, total):
            calls.append((done, total))

        predict_bulk(df, model=model, oof_meta=oof, progress_callback=cb)
        assert len(calls) == 4
        assert calls[-1] == (4, 4)

    def test_all_invalid_rows(self, artifacts):
        model, oof = artifacts
        df = pd.DataFrame([
            _make_valid_row(series_1=999.0),
            _make_valid_row(series_2=-5.0),
        ])
        results, errors = predict_bulk(df, model=model, oof_meta=oof)
        assert len(results) == 0
        assert len(errors) == 2

    def test_large_cohort_no_crash(self, artifacts):
        """Stress test: 500 rows should complete without error."""
        model, oof = artifacts
        df = pd.DataFrame([_make_valid_row() for _ in range(500)])
        results, errors = predict_bulk(df, model=model, oof_meta=oof)
        assert len(results) == 500
        assert len(errors) == 0

    def test_real_test_set_rows_match_pipeline(self, artifacts):
        """Run 10 rows from the actual test CSV through bulk and compare to direct model call."""
        model, oof = artifacts
        from src.features import DEFAULT_FEATURE_COLS
        test_df = pd.read_csv("data/processed/test.csv").head(10)
        bulk_cols = BULK_REQUIRED_COLUMNS + ["age", "gender"]
        bulk_input = test_df[bulk_cols].copy()
        results, errors = predict_bulk(bulk_input, model=model, oof_meta=oof)

        assert len(errors) == 0, f"Unexpected validation errors: {errors['validation_errors'].tolist()}"

        for i, (_, row) in enumerate(test_df.iterrows()):
            raw_input = {
                "series_1": float(row["series_1"]),
                "series_2": float(row["series_2"]),
                "study_hours_per_day": float(row["study_hours_per_day"]),
                "attendance_percentage": float(row["attendance_percentage"]),
                "social_media_hours": float(row["social_media_hours"]),
                "age": float(row["age"]),
                "gender": str(row["gender"]),
            }
            single_score = predict_student_performance(raw_input, model=model, oof_meta=oof)["predicted_score"]
            bulk_score = results["predicted_final_marks"].iloc[i]
            assert abs(bulk_score - single_score) < 1e-6, (
                f"Row {i}: bulk={bulk_score} vs single={single_score}"
            )
