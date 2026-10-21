import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import json
from typing import Any

import joblib
import pandas as pd
import streamlit as st

from src.config import Settings, load_settings, save_settings
from src.db import init_db

PRIMARY_COLOR = "#2563EB"

CUSTOM_CSS = """
<style>
/* ── Academic Minimalist Design Tokens ───────────────────────────────── */
:root {
    --text-main: #0F172A;
    --text-muted: #64748B;
    --bg-card: #FFFFFF;
    --border-card: #E2E8F0;
    --accent-blue: #2563EB;
}

/* Metric Card */
.metric-card {
    background-color: var(--bg-card);
    border: 1px solid var(--border-card);
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.metric-title {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 4px;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text-main);
    line-height: 1.2;
}
.metric-subtitle {
    font-size: 0.75rem;
    color: #94A3B8;
    margin-top: 4px;
}

/* Call-to-action hero banner */
.cta-banner {
    background: #F8FAFC;
    border: 1px solid #CBD5E1;
    border-left: 5px solid #2563EB;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 18px;
}
.cta-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #0F172A;
    margin-bottom: 4px;
}
.cta-desc {
    font-size: 0.85rem;
    color: #475569;
}

/* Risk Badges */
.risk-badge-high {
    background-color: #FEE2E2;
    color: #991B1B;
    padding: 3px 8px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.78rem;
    display: inline-block;
}
.risk-badge-medium {
    background-color: #FEF3C7;
    color: #92400E;
    padding: 3px 8px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.78rem;
    display: inline-block;
}
.risk-badge-low {
    background-color: #DCFCE7;
    color: #166534;
    padding: 3px 8px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.78rem;
    display: inline-block;
}

/* Clean Rule Cards */
.rule-card {
    background-color: #F8FAFC;
    border-left: 3px solid #2563EB;
    padding: 10px 14px;
    border-radius: 4px;
    margin-bottom: 8px;
    font-size: 0.85rem;
}
.rule-indicator {
    border-left-color: #D97706 !important;
}

/* Info button & tooltip styling */
.info-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 15px;
    height: 15px;
    border-radius: 50%;
    background-color: #E2E8F0;
    color: #334155;
    font-size: 10px;
    font-weight: 700;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    cursor: help;
    margin-left: 6px;
    vertical-align: middle;
    transition: all 0.2s ease;
    border: 1px solid #CBD5E1;
}
.info-btn:hover {
    background-color: #2563EB;
    color: #FFFFFF;
    border-color: #1D4ED8;
}

/* Info Box */
.info-box {
    background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 6px;
    padding: 10px 14px;
    color: #334155;
    font-size: 0.82rem;
    margin-bottom: 12px;
}
</style>
"""

MODEL_TERMS_GLOSSARY = {
    "predicted_final_score": {
        "term": "Predicted Final Score (ŷ)",
        "definition": "The expected mark (0.0 – 100.0) forecasted by the regression model based on internal assessments, study habits, and demographic attributes.",
    },
    "prediction_interval": {
        "term": "90% Prediction Interval",
        "definition": "The calibrated score band [Lower, Upper] derived from the 90th percentile of empirical out-of-fold residuals (±13.9 pts). There is a 90% probability that the student's true exam score will fall inside this range.",
    },
    "academic_risk_category": {
        "term": "Academic Risk Category",
        "definition": "Categorizes academic vulnerability relative to Target Score (50.0) + Safety Buffer (3.0 pts). Flags whether a student is On Track, Near Target (Borderline), or Below Target.",
    },
    "risk_probability": {
        "term": "Risk Probability P(Risk)",
        "definition": "The statistical likelihood that the student scores below target, computed via the cumulative normal distribution Φ((Target - ŷ) / σ_residual). Categorized as High (≥60%), Medium (30–60%), or Low (<30%).",
    },
    "exam_eligibility": {
        "term": "Exam Eligibility Policy",
        "definition": "Institutional regulation mandating a minimum of 75.0% attendance. Students falling below 75% are disqualified from sitting for the final exam without formal condonation.",
    },
    "target_score": {
        "term": "Target Score",
        "definition": "The institutional benchmark or passing threshold (default 50.0 points out of 100).",
    },
    "risk_margin": {
        "term": "Risk Margin (Buffer)",
        "definition": "Safety buffer (default 3.0 points) added to the target score to proactively identify borderline students before they fail.",
    },
    "teacher_capacity": {
        "term": "Teacher Capacity (N)",
        "definition": "Workload bound on how many students the department can actively follow up with in a single intervention cycle (default top N=25).",
    },
    "priority_score": {
        "term": "Priority Score",
        "definition": "Composite ranking metric balancing risk probability, declining performance between assessment series, and triggered lifestyle indicators: w_risk · P(risk) + w_decline · Decline + w_flags · Flags.",
    },
    "mae": {
        "term": "MAE (Mean Absolute Error)",
        "definition": "Average point discrepancy between predicted and actual final exam marks on unseen test data. Current all-feature test MAE is 6.960 points.",
    },
    "rmse": {
        "term": "RMSE (Root Mean Squared Error)",
        "definition": "Square root of mean squared prediction error on test data (8.498 points). Heavily penalizes large forecasting misses.",
    },
    "r2": {
        "term": "R² (Coefficient of Determination)",
        "definition": "Proportion of variance in final marks explained by model features. Current model R² is 0.650 (65.0% of score variation is captured).",
    },
    "one_se_rule": {
        "term": "1-Standard-Error (1-SE) Rule",
        "definition": "Parsimony principle in machine learning that selects the simplest, most explainable model whose cross-validation error is within 1 standard error of the best candidate.",
    },
    "cross_validation": {
        "term": "5-Fold Cross-Validation",
        "definition": "Rigorous evaluation method partitioning the 4,000-student training dataset into 5 equal folds to tune hyperparameters without test data leakage.",
    },
    "oof_residuals": {
        "term": "Out-of-Fold (OOF) Residuals",
        "definition": "Prediction errors (Actual − Predicted) collected on unseen folds during cross-validation, used to calibrate honest empirical prediction intervals.",
    },
    "feature_attribution": {
        "term": "Score-Point Feature Attribution",
        "definition": "Linear feature breakdown (x_j - mean_j) · β_j displaying how many concrete exam marks each factor adds or subtracts relative to the cohort mean.",
    },
    "series_1": {
        "term": "Series 1 Exam Score",
        "definition": "First internal assessment exam mark (0–40). Contributes +0.838 points to the final exam score per mark.",
    },
    "series_2": {
        "term": "Series 2 Exam Score",
        "definition": "Second internal assessment exam mark (0–40). Contributes +0.800 points to the final exam score per mark.",
    },
    "series_avg": {
        "term": "Average Series Score",
        "definition": "Mean mark of Series 1 and Series 2 ((S1 + S2) / 2 out of 40). Thresholds below 18/40 trigger remedial support rules.",
    },
    "series_delta": {
        "term": "Assessment Trajectory (Series 2 − Series 1)",
        "definition": "Performance shift between exam series. A drop of 4.0 points or more triggers a declining trajectory alert.",
    },
    "study_hours_per_day": {
        "term": "Daily Study Hours",
        "definition": "Reported daily self-study hours. In the 5,000-student model, each additional study hour adds +1.305 points to the final exam score.",
    },
    "attendance_percentage": {
        "term": "Class Attendance Rate (%)",
        "definition": "Overall lecture and lab attendance rate. Adds +0.109 points per percentage point and determines final exam sitting eligibility (≥ 75%).",
    },
    "social_media_hours": {
        "term": "Daily Social Media Hours",
        "definition": "Daily recreational screen time. In the trained model, each additional hour of social media deducts −0.652 points from the final exam score.",
    },
    "age": {
        "term": "Student Age",
        "definition": "Chronological age in years (14–24). Accounts for maturity adjustments (+0.030 points per year).",
    },
    "gender": {
        "term": "Student Gender",
        "definition": "Student gender profile (Female, Male, Other). Monitored for disparate impact and demographic parity.",
    },
    "lifestyle_influence": {
        "term": "Lifestyle & Study Habit Influence",
        "definition": "The net impact of self-study hours, attendance percentage, and social media habits on the student's forecasted score.",
    },
    "linear_regression": {
        "term": "Linear Regression Model",
        "definition": "Interpretable regression pipeline computing ŷ = β₀ + ∑(βⱼ · xⱼ). Selected under the 1-SE rule for superior generalization and transparent point attribution.",
    },
    "permutation_importance": {
        "term": "Permutation Feature Importance",
        "definition": "Measures predictive reliance by calculating the degradation in R² score when values of a specific feature column are randomly shuffled.",
    },
    "residual": {
        "term": "Residual Error (e = y − ŷ)",
        "definition": "Difference between actual exam mark and model prediction. Homoscedastic scatter around 0.0 confirms model unbiasedness across the score range.",
    },
    "fairness_audit": {
        "term": "Fairness & Disparate Impact Audit",
        "definition": "Demographic parity inspection measuring MAE, recall, and precision across gender and age subgroups to guarantee non-discriminatory advising.",
    },
}


def info_btn(term_key: str) -> str:
    """Returns an inline HTML (i) button with hover tooltip for the term."""
    item = MODEL_TERMS_GLOSSARY.get(term_key, {})
    definition = item.get("definition", "").replace('"', '&quot;')
    term = item.get("term", term_key).replace('"', '&quot;')
    return f'<span class="info-btn" title="{term}: {definition}">i</span>'


def render_glossary_popover(button_label: str = "ℹ️ Model Terminology Guide"):
    """Renders a collapsible popover displaying all model terms and definitions."""
    with st.popover(button_label):
        st.markdown("#### 📘 Machine Learning & Evaluation Terminology Guide")
        st.caption("Reference definitions and mathematical formulas for all model metrics, thresholds, and variables.")

        search = st.text_input("🔍 Search terms...", placeholder="e.g. MAE, prediction interval, attendance", label_visibility="collapsed")
        st.markdown("---")

        for key, val in MODEL_TERMS_GLOSSARY.items():
            if search and search.lower() not in val["term"].lower() and search.lower() not in val["definition"].lower():
                continue
            st.markdown(f"**{val['term']}**")
            st.write(val["definition"])
            st.markdown("")


def apply_custom_styles():
    """Injects minimalist academic theme CSS tokens."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_app_model():
    """Loads and caches the trained scikit-learn pipeline."""
    model_path = Path("models/best_model.joblib")
    if not model_path.exists():
        st.error("Model file not found. Please run: python -m src.train")
        return None
    return joblib.load(model_path)


@st.cache_data
def load_app_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads and caches train and test datasets."""
    train_path = Path("data/processed/train.csv")
    test_path = Path("data/processed/test.csv")
    if not train_path.exists() or not test_path.exists():
        st.error("Processed data not found. Please run: python -m src.data")
        return pd.DataFrame(), pd.DataFrame()
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


@st.cache_data
def load_metrics_and_residuals() -> tuple[dict[str, Any], dict[str, Any]]:
    """Loads and caches model evaluation metrics and residual statistics."""
    metrics_path = Path("models/metrics.json")
    res_path = Path("models/oof_residuals.json")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    res = json.loads(res_path.read_text(encoding="utf-8")) if res_path.exists() else {}
    return metrics, res


def render_sidebar_controls() -> Settings:
    """Renders compact teacher controls in the Streamlit sidebar."""
    init_db()
    settings = load_settings()

    st.sidebar.markdown("### Teacher Controls")

    new_target = st.sidebar.slider(
        "Target Pass Score",
        min_value=30.0,
        max_value=70.0,
        value=float(settings.target_score),
        step=1.0,
        help="Benchmark pass score (default: 40).",
    )

    new_margin = st.sidebar.slider(
        "Risk Margin Buffer (pts)",
        min_value=0.0,
        max_value=10.0,
        value=float(settings.risk_margin),
        step=0.5,
        help="Score buffer for identifying borderline at-risk students.",
    )

    new_capacity = st.sidebar.slider(
        "Follow-Up Capacity (N)",
        min_value=5,
        max_value=100,
        value=int(settings.capacity),
        step=5,
        help="Maximum students queued for prioritized intervention.",
    )

    # Persist if changed
    if (
        new_target != settings.target_score
        or new_capacity != settings.capacity
        or new_margin != settings.risk_margin
    ):
        settings.target_score = new_target
        settings.capacity = new_capacity
        settings.risk_margin = new_margin
        save_settings(settings)



    return settings
