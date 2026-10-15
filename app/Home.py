import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from app.ui_utils import (
    apply_custom_styles,
    load_app_data,
    load_app_model,
    load_metrics_and_residuals,
    render_sidebar_controls,
)
from src.features import REGRESSION_FEATURES, TARGET_COL
from src.priority import build_priority_queue, compute_priority_components

st.set_page_config(
    page_title="Student Performance Dashboard",
    page_icon="🎓",
    layout="wide",
)

apply_custom_styles()
settings = render_sidebar_controls()

# ── A. Header ──────────────────────────────────────────────────────────
st.title("Student Performance Dashboard")
st.caption("Predict academic outcomes and identify students who may benefit from additional support.")

# ── B. Primary Prediction Action ───────────────────────────────────────
st.markdown("""
<div class="cta-banner">
    <div class="cta-title">Predict Student Marks</div>
    <div class="cta-desc">Enter series examination marks and relevant lifestyle attributes to estimate the final examination score and view academic risk classification.</div>
</div>
""", unsafe_allow_html=True)

btn_col, _ = st.columns([1.5, 3.5])
with btn_col:
    if st.button("Predict Student Marks →", type="primary", use_container_width=True):
        st.switch_page("pages/0_Model_Prediction.py")

model = load_app_model()
train_df, test_df = load_app_data()
metrics, oof_res = load_metrics_and_residuals()

if model is None or test_df.empty:
    st.warning("Model artifacts or processed data not found. Please run training pipeline.")
    st.stop()

# ── C. Summary Metric Cards ───────────────────────────────────────────
X_test = test_df[REGRESSION_FEATURES]
y_pred = model.predict(X_test)
residual_std = oof_res.get("residual_std", 6.5)

scored_cohort = compute_priority_components(
    df=test_df,
    predicted_scores=y_pred,
    target_score=settings.target_score,
    residual_std=residual_std,
    settings=settings,
)

total_students = len(test_df)
at_risk_mask = scored_cohort["predicted_score"] < (settings.target_score + settings.risk_margin)
at_risk_count = int(np.sum(at_risk_mask))
avg_predicted = float(np.mean(y_pred))
chosen_model = metrics.get("chosen_model", "LinearRegression")
test_mae = metrics.get("test_regression_metrics", {}).get(chosen_model, {}).get("mae", 5.19)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Active Cohort</div>
        <div class="metric-value">{total_students:,}</div>
        <div class="metric-subtitle">Evaluated students</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Flagged for Support</div>
        <div class="metric-value" style="color: #DC2626;">{at_risk_count}</div>
        <div class="metric-subtitle">&lt; {settings.target_score + settings.risk_margin:.0f} pts threshold ({round(at_risk_count/total_students*100, 1)}%)</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Mean Predicted Mark</div>
        <div class="metric-value">{avg_predicted:.1f} <span style="font-size: 1rem; color: #64748B;">/ 100</span></div>
        <div class="metric-subtitle">Cohort forecast average</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Model Test MAE</div>
        <div class="metric-value">{test_mae:.2f} <span style="font-size: 1rem; color: #64748B;">pts</span></div>
        <div class="metric-subtitle">Mean absolute error (Test Set)</div>
    </div>
    """, unsafe_allow_html=True)

# ── D. Essential Decision-Supporting Charts ────────────────────────────
st.markdown("---")
c_chart1, c_chart2 = st.columns(2)

with c_chart1:
    st.subheader("Student Risk Distribution")
    risk_counts = scored_cohort["risk_badge"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0).reset_index()
    risk_counts.columns = ["Risk Category", "Student Count"]

    fig_risk = px.bar(
        risk_counts,
        x="Risk Category",
        y="Student Count",
        color="Risk Category",
        color_discrete_map={"High": "#DC2626", "Medium": "#D97706", "Low": "#16A34A"},
        text="Student Count",
    )
    fig_risk.update_layout(
        template="plotly_white",
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    fig_risk.update_traces(textposition="outside")
    st.plotly_chart(fig_risk, use_container_width=True)
    st.caption("Risk categories determined by failure probability relative to configured target pass mark.")

with c_chart2:
    st.subheader("Assessment Scores Overview")
    plot_df = pd.DataFrame({
        "Series 1 (/40)": test_df["series_1"],
        "Series 2 (/40)": test_df["series_2"],
        "Final Exam (/100)": test_df[TARGET_COL],
    }).melt(var_name="Assessment", value_name="Score")

    fig_assess = px.box(
        plot_df,
        x="Assessment",
        y="Score",
        color="Assessment",
        color_discrete_sequence=["#2563EB", "#0284C7", "#475569"],
    )
    fig_assess.update_layout(
        template="plotly_white",
        height=260,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    st.plotly_chart(fig_assess, use_container_width=True)
    st.caption("Observed score distributions across internal series and actual final examinations.")

# ── E. Priority Students Preview ──────────────────────────────────────
st.markdown("---")
col_header, col_nav_btn = st.columns([3, 1])
with col_header:
    st.subheader("Students Requiring Attention")
with col_nav_btn:
    if st.button("View All Students →", use_container_width=True):
        st.switch_page("pages/1_Priority_Queue.py")

queue_df = build_priority_queue(
    df=test_df,
    predicted_scores=y_pred,
    residual_std=residual_std,
    settings=settings,
    capacity=5,
)

if queue_df.empty:
    st.info("No students currently flagged for prioritized academic support.")
else:
    preview_table = queue_df[["rank", "student_id", "predicted_score", "risk_badge", "priority_score", "top_reason"]].copy()
    preview_table.columns = ["Rank", "Student ID", "Predicted Score", "Risk Category", "Priority Score", "Primary Concern"]
    st.dataframe(
        preview_table,
        use_container_width=True,
        hide_index=True,
    )
