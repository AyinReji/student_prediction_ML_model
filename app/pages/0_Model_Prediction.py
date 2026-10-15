import sys
from pathlib import Path
from datetime import datetime

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
_app_dir = _repo_root / "app"
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

try:
    from app.ui_utils import (
        apply_custom_styles,
        render_sidebar_controls,
        load_app_model,
        load_metrics_and_residuals,
    )
except ImportError:
    from ui_utils import (
        apply_custom_styles,
        render_sidebar_controls,
        load_app_model,
        load_metrics_and_residuals,
    )

from src.predict import predict_student_performance, validate_input_features
from src.db import log_followup, init_db

st.set_page_config(
    page_title="Model Prediction | Student Support",
    page_icon="🎯",
    layout="wide",
)

apply_custom_styles()
settings = render_sidebar_controls()
init_db()

st.title("🎯 Real-Time Model Prediction Workspace")
st.markdown(
    "Adjust the student's exam scores and routine indicators below. "
    "The **Linear Regression model instantly forecasts the final exam mark** in real time as you adjust the inputs."
)

model = load_app_model()
metrics, oof_res = load_metrics_and_residuals()

if model is None:
    st.error("Prediction model is not available. Please run `python -m src.train` first.")
    st.stop()

# ── Preset Scenarios Helper ───────────────────────────────────────────
if "pred_s1" not in st.session_state:
    st.session_state["pred_s1"] = 22.0
if "pred_s2" not in st.session_state:
    st.session_state["pred_s2"] = 24.5
if "pred_study" not in st.session_state:
    st.session_state["pred_study"] = 3.0
if "pred_att" not in st.session_state:
    st.session_state["pred_att"] = 85.0
if "pred_social" not in st.session_state:
    st.session_state["pred_social"] = 2.0
if "pred_name" not in st.session_state:
    st.session_state["pred_name"] = "Candidate Student"

# Preset scenario buttons
col_preset1, col_preset2, col_preset3, col_preset4 = st.columns(4)
with col_preset1:
    if st.button("⚠️ Load At-Risk Profile", use_container_width=True):
        st.session_state["pred_s1"] = 12.0
        st.session_state["pred_s2"] = 14.0
        st.session_state["pred_study"] = 1.0
        st.session_state["pred_att"] = 62.0
        st.session_state["pred_social"] = 5.5
        st.session_state["pred_name"] = "Struggling Student"
        st.rerun()

with col_preset2:
    if st.button("⚡ Load Average Profile", use_container_width=True):
        st.session_state["pred_s1"] = 22.0
        st.session_state["pred_s2"] = 24.5
        st.session_state["pred_study"] = 3.0
        st.session_state["pred_att"] = 85.0
        st.session_state["pred_social"] = 2.5
        st.session_state["pred_name"] = "Average Student"
        st.rerun()

with col_preset3:
    if st.button("🌟 Load High-Achiever Profile", use_container_width=True):
        st.session_state["pred_s1"] = 34.0
        st.session_state["pred_s2"] = 36.5
        st.session_state["pred_study"] = 5.5
        st.session_state["pred_att"] = 96.0
        st.session_state["pred_social"] = 1.0
        st.session_state["pred_name"] = "Top Candidate"
        st.rerun()

with col_preset4:
    if st.button("🔄 Reset Inputs", use_container_width=True):
        st.session_state["pred_s1"] = 20.0
        st.session_state["pred_s2"] = 20.0
        st.session_state["pred_study"] = 2.5
        st.session_state["pred_att"] = 80.0
        st.session_state["pred_social"] = 2.0
        st.session_state["pred_name"] = "New Candidate"
        st.rerun()

st.markdown("---")

# ── Main Two-Column Side-by-Side Live Workspace ───────────────────────
col_input, col_output = st.columns([1.05, 1.25], gap="large")

with col_input:
    st.subheader("1. Enter Student Parameters")

    # Group A: Predictive Academic Assessments
    st.markdown("##### 📚 Academic Assessments *(Primary Predictive Signal)*")
    st.caption("Internal examination marks (0.0 – 40.0 pts each) that drive final mark prediction:")

    s1_val = st.slider(
        "Series 1 Exam Score (0.0 – 40.0)",
        min_value=0.0,
        max_value=40.0,
        value=float(st.session_state["pred_s1"]),
        step=0.5,
        help="Mandatory. First internal assessment mark.",
        key="slider_s1",
    )
    st.session_state["pred_s1"] = s1_val

    s2_val = st.slider(
        "Series 2 Exam Score (0.0 – 40.0)",
        min_value=0.0,
        max_value=40.0,
        value=float(st.session_state["pred_s2"]),
        step=0.5,
        help="Mandatory. Second internal assessment mark.",
        key="slider_s2",
    )
    st.session_state["pred_s2"] = s2_val

    # Trajectory metric preview
    s_avg = (s1_val + s2_val) / 2.0
    s_delta = s2_val - s1_val
    delta_color = "#16A34A" if s_delta >= 0 else "#DC2626"
    st.markdown(
        f"<div style='background: #F1F5F9; padding: 10px 14px; border-radius: 8px; font-size: 0.9rem; margin-bottom: 15px; border-left: 4px solid #2F5BEA;'>"
        f"📊 <b>Assessment Summary:</b> Average = <b>{s_avg:.1f} / 40</b> | "
        f"Trajectory = <b style='color: {delta_color};'>{s_delta:+.1f} pts</b>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # Group B: Pedagogical Routine & Indicators
    st.markdown("##### ⏱️ Student Routine & Attendance *(Pedagogical Indicators)*")
    st.caption("Used solely for support rules and teacher check-ins; **does NOT alter predicted marks**:")

    study_val = st.slider(
        "Daily Study Time (Hours / Day)",
        min_value=0.0,
        max_value=16.0,
        value=float(st.session_state["pred_study"]),
        step=0.5,
        help="Estimated daily independent study hours.",
        key="slider_study",
    )
    st.session_state["pred_study"] = study_val

    att_val = st.slider(
        "Class Attendance Percentage (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(st.session_state["pred_att"]),
        step=1.0,
        help="Overall percentage of class attendance.",
        key="slider_att",
    )
    st.session_state["pred_att"] = att_val

    social_val = st.slider(
        "Daily Social Media Use (Hours / Day)",
        min_value=0.0,
        max_value=16.0,
        value=float(st.session_state["pred_social"]),
        step=0.5,
        help="Estimated daily recreational screen time.",
        key="slider_social",
    )
    st.session_state["pred_social"] = social_val

    # Group C: Context (Optional)
    with st.expander("👤 Student Profile Details (Optional)"):
        st_name = st.text_input("Student Identifier / Name", value=st.session_state["pred_name"])
        st.session_state["pred_name"] = st_name
        c_age, c_gender = st.columns(2)
        with c_age:
            st_age = st.number_input("Age (Years)", min_value=10, max_value=80, value=19, step=1)
        with c_gender:
            st_gender = st.selectbox("Gender", ["Male", "Female", "Other"])

# ── Dynamic Model Prediction Execution ─────────────────────────────────
raw_payload = {
    "series_1": s1_val,
    "series_2": s2_val,
    "study_hours_per_day": study_val,
    "attendance_percentage": att_val,
    "social_media_hours": social_val,
    "student_label": st.session_state.get("pred_name", "Candidate"),
    "age": 19,
    "gender": "Male",
}

try:
    prediction = predict_student_performance(
        raw_inputs=raw_payload,
        model=model,
        oof_meta=oof_res,
        settings=settings,
    )
except Exception as e:
    st.error(f"Prediction Calculation Error: {e}")
    st.stop()

# ── Right Column: Live Prediction Results ─────────────────────────────
with col_output:
    st.subheader(f"2. Live Prediction for {raw_payload['student_label']}")

    pred_score = prediction["predicted_score"]
    interval = prediction["prediction_interval"]
    risk = prediction["risk_evaluation"]
    risk_badge = risk["risk_badge"]
    badge_class = f"risk-badge-{risk_badge.lower()}"

    # Score color coding
    if pred_score >= settings.target_score + settings.risk_margin:
        score_color = "#16A34A"  # Green
        status_label = "✅ On Track"
        status_color = "#16A34A"
    elif pred_score >= settings.target_score:
        score_color = "#D97706"  # Amber
        status_label = "⚠️ Near Threshold (Borderline)"
        status_color = "#D97706"
    else:
        score_color = "#DC2626"  # Red
        status_label = "🚨 Flagged At-Risk"
        status_color = "#DC2626"

    # Main Metric Cards Row 1
    m1, m2 = st.columns(2)
    with m1:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid {score_color};">
            <div class="metric-title">Predicted Final Exam Score</div>
            <div class="metric-value" style="color: {score_color}; font-size: 2.2rem;">
                {pred_score:.1f} <span style="font-size: 1rem; color: #64748B;">/ 100</span>
            </div>
            <div class="metric-subtitle">Live forecast from Linear Regression</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid #2F5BEA;">
            <div class="metric-title">90% Prediction Interval</div>
            <div class="metric-value" style="font-size: 1.6rem; color: #1E293B;">
                [{interval['lower']:.1f}, {interval['upper']:.1f}]
            </div>
            <div class="metric-subtitle">Expected range (margin ±{interval['half_width']:.1f} pts)</div>
        </div>
        """, unsafe_allow_html=True)

    # Metric Cards Row 2
    m3, m4 = st.columns(2)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">At-Risk Evaluation</div>
            <div class="metric-value"><span class="{badge_class}">{risk_badge} Risk</span></div>
            <div class="metric-subtitle">Risk Probability: <b>{risk['risk_probability_pct']:.1f}%</b></div>
        </div>
        """, unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Intervention Status</div>
            <div class="metric-value" style="color: {status_color}; font-size: 1.3rem;">{status_label}</div>
            <div class="metric-subtitle">Target: <b>{settings.target_score:.0f}</b> pts (+{settings.risk_margin:.0f} buffer)</div>
        </div>
        """, unsafe_allow_html=True)

    # Formula Breakdown Box
    st.markdown(f"""
    <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 10px 14px; margin: 12px 0; font-size: 0.85rem; color: #334155;">
        <b>📐 Model Formula Calculation:</b><br>
        <code>Final Score = 13.93 + (0.962 × Series 1) + (0.929 × Series 2)</code><br>
        <code>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;= 13.93 + (0.962 × {s1_val:.1f}) + (0.929 × {s2_val:.1f}) = <b>{pred_score:.1f} pts</b></code>
    </div>
    """, unsafe_allow_html=True)

    # Factor Attribution Bar Chart
    st.markdown("##### 📊 Factor Attribution (Relative to Cohort Average)")
    contrib_list = prediction["contributions"]
    contrib_df = pd.DataFrame(contrib_list)

    fig_bar = px.bar(
        contrib_df,
        x="points",
        y="display_name",
        orientation="h",
        color="points",
        color_continuous_scale=["#EF4444", "#E2E8F0", "#10B981"],
        text="points",
        labels={"points": "Point Impact", "display_name": "Factor"},
    )
    fig_bar.update_layout(
        template="plotly_white",
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=10, t=10, b=10),
        height=180,
        coloraxis_showscale=False,
    )
    fig_bar.update_traces(texttemplate="%{text:+.1f} pts", textposition="outside")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Action Plan & Support Recommendations
    st.markdown("##### 💡 Triggered Support Recommendations")
    rules_fired = prediction.get("rules_fired", [])

    if not rules_fired:
        st.success("✅ No critical intervention thresholds triggered. Student is tracking solidly.")
    else:
        for rule in rules_fired:
            is_ind = rule.get("is_indicator_only", False)
            card_class = "rule-card rule-indicator" if is_ind else "rule-card"
            badge_text = "Conversational Flag" if is_ind else "Academic Trigger"
            badge_color = "#D97706" if is_ind else "#2563EB"

            st.markdown(f"""
            <div class="{card_class}" style="margin-bottom: 8px;">
                <b>{rule['name']}</b> <span style="color: {badge_color}; font-weight: 600; font-size: 0.8rem;">[{badge_text}]</span><br>
                <div style="font-size: 0.85rem; color: #1E293B; margin-top: 2px;"><b>Suggested Action:</b> {rule['action']}</div>
                <div style="font-size: 0.78rem; color: #64748B; font-style: italic;">ℹ️ {rule['info_note']}</div>
            </div>
            """, unsafe_allow_html=True)

    # Option to Log Follow-up directly to Database
    with st.expander("📝 Log Follow-up / Save to Tracking Database"):
        followup_notes = st.text_area("Teacher Follow-up Plan & Notes", value=f"Assessed marks (S1={s1_val}, S2={s2_val}). Target: {settings.target_score}.")
        if st.button("💾 Save Follow-up Record", type="primary"):
            log_followup(
                student_id=9999,  # Manual evaluation student ID
                note=followup_notes,
                action_type="Teacher Check-in",
                status="Planned",
            )
            st.success(f"Follow-up record logged for {raw_payload['student_label']}!")
