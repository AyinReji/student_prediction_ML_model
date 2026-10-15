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

    st.sidebar.divider()
    st.sidebar.caption(
        "**Predictive Policy:** Final examination marks are predicted using Series 1, Series 2, and "
        "relevant lifestyle attributes. Lifestyle attributes may contribute to predictions when supported "
        "by the training data. These attributes can also inform personalized academic support recommendations."
    )

    return settings
