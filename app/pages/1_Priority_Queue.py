import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
_app_dir = _repo_root / "app"
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))

import streamlit as st
import pandas as pd
import numpy as np

from app.ui_utils import (
    apply_custom_styles,
    info_btn,
    load_app_data,
    load_app_model,
    load_metrics_and_residuals,
    render_glossary_popover,
    render_sidebar_controls,
)
from src.features import REGRESSION_FEATURES
from src.priority import build_priority_queue

st.set_page_config(page_title="Priority Queue | Student Support", page_icon="📋", layout="wide")
apply_custom_styles()
settings = render_sidebar_controls()

head_c1, head_c2 = st.columns([3, 1])
with head_c1:
    st.title("Teacher Priority Queue")
with head_c2:
    st.write("")
    render_glossary_popover()
st.caption(f"Identifies the top {settings.capacity} students prioritized for intervention within teacher capacity.")

model = load_app_model()
train_df, test_df = load_app_data()
metrics, oof_res = load_metrics_and_residuals()

if model is None or test_df.empty:
    st.warning("Model or data not available. Please ensure model training is complete.")
    st.stop()

X_test = test_df[REGRESSION_FEATURES]
y_pred = model.predict(X_test)
residual_std = oof_res.get("residual_std", 6.5)

# Calculate priority queue
queue_df = build_priority_queue(
    df=test_df,
    predicted_scores=y_pred,
    residual_std=residual_std,
    settings=settings,
    capacity=settings.capacity,
)

# ── Summary Metrics ───────────────────────────────────────────────────
at_risk_count = int(np.sum(y_pred < (settings.target_score + settings.risk_margin)))
high_priority_count = int(np.sum(queue_df["risk_badge"] == "High")) if not queue_df.empty else 0

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Students Flagged {info_btn('academic_risk_category')}</div>
        <div class="metric-value">{at_risk_count}</div>
        <div class="metric-subtitle">Score &lt; {settings.target_score + settings.risk_margin:.0f} pts threshold</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">High Priority in Queue {info_btn('risk_probability')}</div>
        <div class="metric-value" style="color: #DC2626;">{high_priority_count}</div>
        <div class="metric-subtitle">Highest failure probability</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Teacher Capacity (N) {info_btn('teacher_capacity')}</div>
        <div class="metric-value">{settings.capacity}</div>
        <div class="metric-subtitle">Active intervention limit</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

if queue_df.empty:
    st.info("No students currently meet priority intervention thresholds.")
else:
    # ── Quick Student Navigation Action ───────────────────────────────
    col_sel, col_btn = st.columns([2.5, 1.5])
    with col_sel:
        selected_student = st.selectbox(
            "Select Student to Review Detail:",
            options=queue_df["student_id"].tolist(),
            format_func=lambda x: f"Student #{x} (Rank #{queue_df[queue_df['student_id'] == x]['rank'].iloc[0]})",
        )
    with col_btn:
        st.write("")  # spacing
        st.write("")
        if st.button("Open Student Detail →", use_container_width=True):
            st.session_state["selected_student_id"] = int(selected_student)
            st.query_params["student_id"] = str(selected_student)
            st.switch_page("pages/2_Student_Detail.py")

    # ── Compact Priority Table ────────────────────────────────────────
    queue_df["exam_eligibility"] = queue_df["attendance_percentage"].apply(
        lambda a: "Eligible" if a >= float(settings.rules.min_attendance) else "Ineligible (<75%)"  # noqa: E501
    )

    display_df = queue_df[[
        "rank",
        "student_id",
        "predicted_score",
        "risk_badge",
        "exam_eligibility",
        "priority_score",
        "attendance_percentage",
        "series_1",
        "series_2",
        "top_reason",
    ]].copy()

    display_df["attendance_percentage"] = display_df["attendance_percentage"].apply(lambda a: f"{a:.1f}%")

    display_df.columns = [
        "Rank",
        "Student ID",
        "Predicted Score",
        "Risk Category",
        "Exam Eligibility",
        "Priority Score",
        "Attendance",
        "Series 1 (/40)",
        "Series 2 (/40)",
        "Primary Concern",
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    # ── Export & Mathematical Explainer ───────────────────────────────
    col_dl, _ = st.columns([1.5, 3])
    with col_dl:
        csv_bytes = queue_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Priority Queue (CSV)",
            data=csv_bytes,
            file_name=f"priority_queue_top_{settings.capacity}.csv",
            mime="text/csv",
        )

    with st.expander("Priority Calculation Methodology"):
        st.markdown(f"**Composite Priority Score Formula:** {info_btn('priority_score')}", unsafe_allow_html=True)
        st.markdown(f"""
        The queue ranks students within the teacher's capacity ($N={settings.capacity}$) by blending:
        - **Failure Risk Probability ($w={settings.priority_weights.risk}$):** $P(\\text{{Final Score}} < {settings.target_score:.0f})$ {info_btn('risk_probability')}
        - **Trajectory Decline ($w={settings.priority_weights.decline}$):** Score decrease from Series 1 to Series 2 {info_btn('series_delta')}
        - **Pedagogical Rule Alerts ($w={settings.priority_weights.flags}$):** Share of triggered support guidelines
        - **Exam Eligibility:** {info_btn('exam_eligibility')}
        """, unsafe_allow_html=True)
