import sys
from pathlib import Path

# Ensure workspace root is in sys.path
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

PRIMARY_COLOR = "#2F5BEA"

CUSTOM_CSS = """
<style>
/* Clean modern dashboard styling */
.metric-card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 18px 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    margin-bottom: 12px;
}
.metric-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #1E293B;
}
.metric-subtitle {
    font-size: 0.8rem;
    color: #94A3B8;
    margin-top: 4px;
}
.risk-badge-high {
    background-color: #FEE2E2;
    color: #991B1B;
    padding: 4px 10px;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.8rem;
    display: inline-block;
}
.risk-badge-medium {
    background-color: #FEF3C7;
    color: #92400E;
    padding: 4px 10px;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.8rem;
    display: inline-block;
}
.risk-badge-low {
    background-color: #DCFCE7;
    color: #166534;
    padding: 4px 10px;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 0.8rem;
    display: inline-block;
}
.rule-card {
    background-color: #F8FAFC;
    border-left: 4px solid #2F5BEA;
    padding: 12px 16px;
    border-radius: 4px;
    margin-bottom: 10px;
}
.rule-indicator {
    border-left-color: #F59E0B !important;
}
.disclaimer-box {
    background-color: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 8px;
    padding: 12px 16px;
    color: #1E40AF;
    font-size: 0.85rem;
    margin-bottom: 16px;
}
</style>
"""


def apply_custom_styles():
    """Injects high-quality light-theme CSS tokens."""
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
    """Renders persistent teacher controls in the Streamlit sidebar."""
    init_db()
    settings = load_settings()

    st.sidebar.markdown("### ⚙️ Teacher Controls")
    st.sidebar.markdown("Configure operational parameters dynamically:")

    new_target = st.sidebar.slider(
        "Target Pass Score (0-100)",
        min_value=30.0,
        max_value=70.0,
        value=float(settings.target_score),
        step=1.0,
        help="Students with predicted score below target + margin are flagged as at-risk.",
    )

    new_capacity = st.sidebar.slider(
        "Follow-Up Capacity (N)",
        min_value=5,
        max_value=100,
        value=int(settings.capacity),
        step=5,
        help="Maximum number of student interventions the teacher can execute this cycle.",
    )

    new_margin = st.sidebar.slider(
        "Risk Margin (Points)",
        min_value=0.0,
        max_value=10.0,
        value=float(settings.risk_margin),
        step=0.5,
        help="Buffer added to target score for identifying at-risk boundary.",
    )

    # Check if settings changed
    if (
        new_target != settings.target_score
        or new_capacity != settings.capacity
        or new_margin != settings.risk_margin
    ):
        settings.target_score = new_target
        settings.capacity = new_capacity
        settings.risk_margin = new_margin
        save_settings(settings)

    st.sidebar.divider()
    st.sidebar.info(
        "💡 **Data Reality Notice:**\n"
        "In this dataset, only `series_1` and `series_2` drive final exam score predictions. "
        "Lifestyle factors (study hours, attendance, social media) are indicators for conversation, not causal drivers."
    )

    return settings
