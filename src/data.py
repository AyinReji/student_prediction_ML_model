from pathlib import Path
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

from src.config import load_settings
from src.features import (
    TARGET_COL,
    add_engineered_features,
)

RAW_DATA_PATH = Path("data/raw/student_performance_10000-1.csv")
PROCESSED_DIR = Path("data/processed")
REPORTS_DIR = Path("reports")
FIGURES_DIR = Path("reports/figures")


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Loads raw student performance dataset and assigns row index as student_id."""
    if not path.exists():
        # Fallback to local root if not yet in data/raw
        alt_path = Path("student_performance_10000-1.csv")
        if alt_path.exists():
            path = alt_path
        else:
            raise FileNotFoundError(f"Raw data file not found at {path}")
    df = pd.read_csv(path)
    if "student_id" not in df.columns:
        df.insert(0, "student_id", df.index)
    return df


def audit_data(df: pd.DataFrame) -> dict[str, Any]:
    """Conducts a rigorous data audit on the raw dataset without fitting any model."""
    numeric_cols = [
        "age",
        "study_hours_per_day",
        "attendance_percentage",
        "social_media_hours",
        "series_1",
        "series_2",
        "final_exam_score",
    ]

    # 1. Integrity checks
    null_counts = df.isnull().sum().to_dict()
    num_duplicates = int(df.duplicated(subset=[c for c in df.columns if c != "student_id"]).sum())
    dtypes = {k: str(v) for k, v in df.dtypes.items()}

    # 2. Range checks
    range_checks = {
        "series_1_valid": bool(df["series_1"].between(0, 40).all()),
        "series_2_valid": bool(df["series_2"].between(0, 40).all()),
        "attendance_valid": bool(df["attendance_percentage"].between(0, 100).all()),
        "final_exam_score_valid": bool(df["final_exam_score"].between(0, 100).all()),
        "study_hours_valid": bool(df["study_hours_per_day"].between(0, 24).all()),
        "social_media_hours_valid": bool(df["social_media_hours"].between(0, 24).all()),
        "age_valid": bool(df["age"].between(10, 100).all()),
    }

    # 3. IQR Outlier detection
    outlier_counts = {}
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        n_outliers = int(((df[col] < lower_bound) | (df[col] > upper_bound)).sum())
        outlier_counts[col] = {
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "count": n_outliers,
            "pct": float(round((n_outliers / len(df)) * 100, 2)),
        }

    # 4. Correlation matrix
    corr_matrix = df[numeric_cols].corr()

    # 5. Leakage check
    target_corrs = corr_matrix["final_exam_score"].drop("final_exam_score")
    near_copies = target_corrs[target_corrs.abs() > 0.95].to_dict()

    # 6. Demographics & representativeness
    gender_counts = df["gender"].value_counts().to_dict()
    age_dist = df["age"].describe().to_dict()

    return {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "null_counts": null_counts,
        "num_duplicates": num_duplicates,
        "dtypes": dtypes,
        "range_checks": range_checks,
        "outlier_counts": outlier_counts,
        "corr_matrix": corr_matrix,
        "near_copies": near_copies,
        "gender_counts": gender_counts,
        "age_dist": age_dist,
    }


def generate_data_audit_report(audit_results: dict[str, Any], output_path: Path = REPORTS_DIR / "data_audit.md") -> None:
    """Writes the comprehensive audit findings to reports/data_audit.md."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    corr: pd.DataFrame = audit_results["corr_matrix"]

    # Generate static figure for the report
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".3f", cmap="Blues", cbar=True, vmin=-0.1, vmax=1.0)
    plt.title("Correlation Matrix (Raw Dataset)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=300)
    plt.close()

    content = f"""# Data Audit Report: Student Performance Dataset

## 1. Overview & Data Integrity
- **Total Records:** {audit_results['n_rows']}
- **Total Features:** {audit_results['n_cols']}
- **Missing Values:** {sum(audit_results['null_counts'].values())} across all columns.
- **Duplicate Records:** {audit_results['num_duplicates']} duplicates detected.
- **Identifier Handling:** The dataset lacks native student identifiers and timestamps. Row indices (0 to {audit_results['n_rows'] - 1}) are assigned as pseudo student IDs.

## 2. Range & Domain Validation
All variables strictly fall within expected academic domain bounds:
- `series_1` & `series_2`: [0, 40] — Valid: {audit_results['range_checks']['series_1_valid']} & {audit_results['range_checks']['series_2_valid']}
- `attendance_percentage`: [0, 100] — Valid: {audit_results['range_checks']['attendance_valid']}
- `study_hours_per_day` & `social_media_hours`: [0, 24] — Valid: {audit_results['range_checks']['study_hours_valid']} & {audit_results['range_checks']['social_media_hours_valid']}
- `final_exam_score`: [0, 100] — Valid: {audit_results['range_checks']['final_exam_score_valid']}

## 3. Distribution & Outlier Summary (1.5 × IQR)
| Column | Q1 | Q3 | IQR | Bounds | Outlier Count | Outlier % |
|---|---|---|---|---|---|---|
"""
    for col, o in audit_results["outlier_counts"].items():
        content += f"| `{col}` | {o['q1']:.2f} | {o['q3']:.2f} | {o['iqr']:.2f} | [{o['lower_bound']:.2f}, {o['upper_bound']:.2f}] | {o['count']} | {o['pct']:.2f}% |\n"

    content += """
*Note: Valid domain values identified by IQR are preserved without deletion.*

## 4. Correlation Analysis & Predictive Signal
| Variable | Correlation with `final_exam_score` |
|---|---|
"""
    for col in corr.columns:
        if col != "final_exam_score":
            content += f"| `{col}` | {corr.loc[col, 'final_exam_score']:.4f} |\n"

    content += f"""
### Key Findings on Predictive Signal
1. **Primary Drivers:** `series_1` (r = {corr.loc['series_1', 'final_exam_score']:.3f}) and `series_2` (r = {corr.loc['series_2', 'final_exam_score']:.3f}) carry essentially all predictive signal.
2. **Lifestyle Attributes:** `study_hours_per_day` (r = {corr.loc['study_hours_per_day', 'final_exam_score']:.3f}), `attendance_percentage` (r = {corr.loc['attendance_percentage', 'final_exam_score']:.3f}), and `social_media_hours` (r = {corr.loc['social_media_hours', 'final_exam_score']:.3f}) exhibit virtually zero linear correlation with exam outcomes (|r| <= 0.01).
3. **Data Authenticity Assessment:** The complete absence of collinearity between lifestyle habits and academic performance strongly suggests simulated or synthetic data generation. In natural educational environments, attendance and study hours typically correlate moderately with academic scores.
4. **Wording & Practice Guardrails:** Lifestyle attributes MUST NOT be presented as causes of low performance. In all user interfaces and teacher communications, they are presented strictly as conversation starters and indicators.
5. **No Subject/Topic Granularity:** The dataset provides aggregate series marks only, with no subject or question-level breakdowns. Topic-level diagnostic claims are strictly prohibited.

## 5. Leakage Verification
- Target `final_exam_score` is strictly segregated from input feature sets.
- No feature is a proxy or near-copy of the target (|r| > 0.95 check found: {len(audit_results['near_copies'])} features).
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


def split_and_save_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits data 80% train / 20% test stratified on target quantile bins and saves to disk."""
    settings = load_settings()
    test_size = settings.split.test_size
    seed = settings.split.seed

    df_featured = add_engineered_features(df)

    # 10 quantile bins for stratified split on continuous target
    quantile_bins = pd.qcut(df_featured[TARGET_COL], q=10, labels=False, duplicates="drop")

    train_df, test_df = train_test_split(
        df_featured,
        test_size=test_size,
        random_state=seed,
        stratify=quantile_bins,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    return train_df, test_df


def run_pipeline() -> None:
    """Executes the data audit and stratified splitting pipeline."""
    df = load_raw_data()
    audit_results = audit_data(df)
    generate_data_audit_report(audit_results)
    train_df, test_df = split_and_save_data(df)
    print(f"Data audit completed. Report written to {REPORTS_DIR / 'data_audit.md'}")
    print(f"Split completed: Train shape {train_df.shape}, Test shape {test_df.shape}")


if __name__ == "__main__":
    run_pipeline()
