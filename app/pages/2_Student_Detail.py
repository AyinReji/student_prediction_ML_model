import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
_app_dir = _repo_root / "app"
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from app.ui_utils import (
        apply_custom_styles,
        load_app_data,
        load_app_model,
        load_metrics_and_residuals,
        render_sidebar_controls,
    )
except ImportError:
    from ui_utils import (
        apply_custom_styles,
        load_app_data,
        load_app_model,
        load_metrics_and_residuals,
        render_sidebar_controls,
    )
from src.db import get_followups, log_followup
from src.explain import ModelExplainer
from src.features import DEFAULT_FEATURE_COLS
from src.priority import calculate_p_risk, get_risk_badge
from src.recommend import evaluate_student_rules

st.set_page_config(page_title="Student Detail | Student Support", page_icon="🔍", layout="wide")
apply_custom_styles()
settings = render_sidebar_controls()

st.title("🔍 Individual Student Diagnostic & Support Detail")

model = load_app_model()
train_df, test_df = load_app_data()
metrics, oof_res = load_metrics_and_residuals()

if model is None or test_df.empty:
    st.warning("Model or data not ready. Please run train pipeline first.")
    st.stop()

# Student Selector
all_students = test_df.sort_values(by="student_id")
student_ids = all_students["student_id"].tolist()

# Query param support
query_id = st.query_params.get("student_id", None)
default_idx = 0
if query_id and int(query_id) in student_ids:
    default_idx = student_ids.index(int(query_id))

selected_id = st.selectbox(
    "Select Student by ID:",
    options=student_ids,
    index=default_idx,
    format_func=lambda x: f"Student #{x}",
)

student_row = test_df[test_df["student_id"] == selected_id].iloc[0]

# Model prediction and interval
features_vector = student_row[DEFAULT_FEATURE_COLS].to_frame().T
pred_score = float(model.predict(features_vector)[0])
half_width = float(oof_res.get("interval_half_width", 11.12))
lower_bound = max(0.0, pred_score - half_width)
upper_bound = min(100.0, pred_score + half_width)

residual_std = float(oof_res.get("residual_std", 6.5))
p_risk = float(calculate_p_risk(np.array([pred_score]), settings.target_score, residual_std)[0])
risk_badge = get_risk_badge(p_risk, settings.risk_badges.high, settings.risk_badges.medium)

# Rules evaluation
fired_rules = evaluate_student_rules(student_row, settings)

# Explanations
explainer = ModelExplainer(model=model, feature_names=DEFAULT_FEATURE_COLS)
contributions = explainer.explain_student(student_row, top_k=5)

# Header Metric Cards
c1, c2, c3, c4 = st.columns(4)
with c1:
    badge_class = f"risk-badge-{risk_badge.lower()}"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Risk Assessment</div>
        <div class="metric-value"><span class="{badge_class}">{risk_badge} Risk</span></div>
        <div class="metric-subtitle">Risk Probability: {p_risk*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Predicted Final Score</div>
        <div class="metric-value">{pred_score:.1f} <span style="font-size: 1rem; color: #64748B;">/ 100</span></div>
        <div class="metric-subtitle">Target: {settings.target_score:.0f} pts</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">90% Prediction Interval</div>
        <div class="metric-value" style="font-size: 1.5rem;">[{lower_bound:.1f}, {upper_bound:.1f}]</div>
        <div class="metric-subtitle">±{half_width:.1f} pts margin of error</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    actual_score = student_row.get("final_exam_score", None)
    actual_str = f"{actual_score:.1f}" if pd.notnull(actual_score) else "Pending"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Actual Final Score</div>
        <div class="metric-value" style="color: #64748B;">{actual_str}</div>
        <div class="metric-subtitle">Ground truth record</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

col_left, col_right = st.columns([1.2, 1.0])

with col_left:
    st.subheader("📊 Factor Contributions to Score (Points)")
    st.caption("Point impact relative to the typical student average across the cohort.")

    contrib_df = pd.DataFrame(contributions)
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
        margin=dict(l=10, r=10, t=30, b=20),
        coloraxis_showscale=False,
    )
    fig_bar.update_traces(texttemplate="%{text:+.1f} pts", textposition="outside")
    st.plotly_chart(fig_bar, use_container_width=True)

with col_right:
    st.subheader("📈 Assessment Trajectory")
    s1 = float(student_row["series_1"])
    s2 = float(student_row["series_2"])
    delta = s2 - s1

    traj_fig = go.Figure()
    traj_fig.add_trace(go.Scatter(
        x=["Series 1", "Series 2"],
        y=[s1, s2],
        mode="lines+markers+text",
        text=[f"{s1:.1f}/40", f"{s2:.1f}/40"],
        textposition="top center",
        line=dict(color="#2F5BEA", width=3),
        marker=dict(size=10, color="#2F5BEA"),
    ))
    traj_fig.update_layout(
        template="plotly_white",
        yaxis=dict(range=[0, 42], title="Assessment Score (/40)"),
        title=f"Internal Assessment Momentum ({delta:+.1f} points)",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(traj_fig, use_container_width=True)

# Recommended Actions & Triggered Rules
st.subheader("💡 Triggered Rules & Recommended Actions")
if not fired_rules:
    st.success("No critical support rules triggered for this student. Student is performing steadily.")
else:
    for rule in fired_rules:
        is_indicator = rule.get("is_indicator_only", False)
        card_class = "rule-card rule-indicator" if is_indicator else "rule-card"
        label_badge = f"<span style='color: {'#D97706' if is_indicator else '#2563EB'}; font-weight: 600;'>[{rule['label']}]</span>"

        st.markdown(f"""
        <div class="{card_class}">
            <b>{rule['name']}</b> {label_badge}<br>
            <div style="font-size: 0.85rem; color: #475569; margin: 4px 0;"><b>Trigger:</b> {rule['trigger']}</div>
            <div style="font-size: 0.9rem; color: #1E293B; margin: 4px 0;"><b>Suggested Teacher Action:</b> {rule['action']}</div>
            <div style="font-size: 0.8rem; color: #64748B; font-style: italic; margin-top: 4px;">ℹ️ {rule['info_note']}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# Form to Log Follow-Up
st.subheader("📝 Log a Follow-Up Intervention")
with st.form(f"followup_form_{selected_id}"):
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        teacher_name = st.text_input("Teacher Name / Initials", "Faculty Lead")
    with fc2:
        action_type = st.selectbox(
            "Action Type",
            options=[
                "Review Assessment Work",
                "Progress Check-In",
                "Attendance Consultation",
                "Study Routine Discussion",
                "Time Management Support",
                "Other Academic Support",
            ],
        )
    with fc3:
        status_val = st.selectbox("Status", ["Completed", "In Progress", "Scheduled"])

    notes = st.text_area("Observation Notes & Agreed Next Steps", placeholder="Enter specific conversational notes or agreements...")
    submitted = st.form_submit_button("💾 Save Follow-Up Record")

    if submitted:
        log_followup(
            student_id=selected_id,
            teacher=teacher_name,
            action_type=action_type,
            notes=notes,
            status=status_val,
        )
        st.success(f"Follow-up successfully logged for Student #{selected_id}!")

# Past Follow-ups
existing_followups = get_followups(student_id=selected_id)
if existing_followups:
    st.markdown("#### Historical Follow-Up Log for this Student")
    st.dataframe(pd.DataFrame(existing_followups)[["date", "teacher", "action_type", "status", "notes"]], hide_index=True)
