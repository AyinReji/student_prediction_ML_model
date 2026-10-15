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
import streamlit as st

from app.ui_utils import (
    apply_custom_styles,
    load_app_data,
    load_app_model,
    load_metrics_and_residuals,
    render_sidebar_controls,
)
from src.db import get_followups, log_followup
from src.explain import ModelExplainer
from src.features import REGRESSION_FEATURES
from src.priority import calculate_p_risk, get_risk_badge
from src.recommend import evaluate_student_rules

st.set_page_config(page_title="Student Detail | Student Support", page_icon="🔍", layout="wide")
apply_custom_styles()
settings = render_sidebar_controls()

st.title("Student Support Diagnostic")
st.caption("Individual assessment profile, score prediction interval, and intervention tracking.")

model = load_app_model()
train_df, test_df = load_app_data()
metrics, oof_res = load_metrics_and_residuals()

if model is None or test_df.empty:
    st.warning("Model or data not ready. Please run train pipeline first.")
    st.stop()

# ── Student Selection ─────────────────────────────────────────────────
all_students = test_df.sort_values(by="student_id")
student_ids = all_students["student_id"].tolist()

query_id = st.query_params.get("student_id", None)
if query_id is None and "selected_student_id" in st.session_state:
    query_id = st.session_state["selected_student_id"]

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

# ── Inference & Risk Assessment ───────────────────────────────────────
features_vector = student_row[REGRESSION_FEATURES].to_frame().T
pred_score = float(model.predict(features_vector)[0])
pred_score_bounded = round(float(np.clip(pred_score, 0.0, 100.0)), 1)

half_width = float(oof_res.get("interval_half_width", 11.12))
lower_bound = max(0.0, round(pred_score_bounded - half_width, 1))
upper_bound = min(100.0, round(pred_score_bounded + half_width, 1))

residual_std = float(oof_res.get("residual_std", 6.5))
p_risk = float(calculate_p_risk(np.array([pred_score_bounded]), settings.target_score, residual_std)[0])
risk_badge = get_risk_badge(p_risk, settings.risk_badges.high, settings.risk_badges.medium)

if pred_score_bounded < settings.target_score:
    status_text = "Below Target"
    status_color = "#DC2626"
    badge_style = "risk-badge-high"
elif pred_score_bounded < (settings.target_score + settings.risk_margin):
    status_text = "Near Target"
    status_color = "#D97706"
    badge_style = "risk-badge-medium"
else:
    status_text = "On Track"
    status_color = "#16A34A"
    badge_style = "risk-badge-low"

# ── Metric Row ────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card" style="border-top: 4px solid {status_color};">
        <div class="metric-title">Predicted Final Score</div>
        <div class="metric-value" style="color: {status_color};">{pred_score_bounded:.1f} <span style="font-size: 0.9rem; color: #64748B;">/ 100</span></div>
        <div class="metric-subtitle">Forecasted mark</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Prediction Interval (90%)</div>
        <div class="metric-value" style="font-size: 1.45rem;">[{lower_bound:.1f}, {upper_bound:.1f}]</div>
        <div class="metric-subtitle">Margin ±{half_width:.1f} pts</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Risk Level</div>
        <div class="metric-value"><span class="{badge_style}">{risk_badge} Risk</span></div>
        <div class="metric-subtitle">Probability: {p_risk*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Academic Status</div>
        <div class="metric-value" style="color: {status_color}; font-size: 1.3rem;">{status_text}</div>
        <div class="metric-subtitle">Target {settings.target_score:.0f} pts (+{settings.risk_margin:.0f} buffer)</div>
    </div>
    """, unsafe_allow_html=True)

# ── Assessment Profile & Recommendations ──────────────────────────────
st.markdown("---")
col_perf, col_rules = st.columns([1.1, 1.3])

with col_perf:
    st.subheader("Assessment Performance")
    s1 = float(student_row["series_1"])
    s2 = float(student_row["series_2"])
    delta = s2 - s1
    delta_color = "#16A34A" if delta >= 0 else "#DC2626"

    st.markdown(f"""
    - **Series 1 Marks:** `{s1:.1f} / 40.0`
    - **Series 2 Marks:** `{s2:.1f} / 40.0`
    - **Average Series Score:** `{(s1+s2)/2:.1f} / 40.0`
    - **Trajectory Shift:** <span style="color: {delta_color}; font-weight: 600;">{delta:+.1f} pts</span>
    """, unsafe_allow_html=True)

    explainer = ModelExplainer(model=model, feature_names=REGRESSION_FEATURES)
    contribs = explainer.explain_student(student_row, top_k=2)
    if contribs:
        st.markdown("**Assessment Attribution to Prediction:**")
        attr_df = pd.DataFrame([
            {
                "Feature": c["display_name"],
                "Student Mark": f"{c['value']:.1f}/40",
                "Point Impact": f"{c['points']:+.2f} pts",
            }
            for c in contribs
        ])
        st.dataframe(attr_df, use_container_width=True, hide_index=True)

with col_rules:
    st.subheader("Pedagogical Guidelines & Actions")
    fired_rules = evaluate_student_rules(student_row, settings)
    if not fired_rules:
        st.success("No critical warning rules triggered. Student is tracking solidly.")
    else:
        for rule in fired_rules:
            is_ind = rule.get("is_indicator_only", False)
            card_class = "rule-card rule-indicator" if is_ind else "rule-card"
            badge = "Conversational Indicator" if is_ind else "Academic Trigger"
            st.markdown(f"""
            <div class="{card_class}">
                <b>{rule['name']}</b> <span style="font-size: 0.75rem; color: #64748B;">[{badge}]</span><br>
                <span><b>Action:</b> {rule['action']}</span><br>
                <span style="font-size: 0.78rem; color: #64748B; font-style: italic;">{rule['info_note']}</span>
            </div>
            """, unsafe_allow_html=True)

# ── Follow-Up Log & New Record Form ───────────────────────────────────
st.markdown("---")
st.subheader("Intervention History & Log")

all_followups = get_followups(student_id=int(selected_id))
if not all_followups:
    st.info(f"No previous follow-ups recorded for Student #{selected_id}.")
else:
    st.dataframe(
        pd.DataFrame(all_followups)[["id", "date", "teacher", "action_type", "status", "notes"]],
        use_container_width=True,
        hide_index=True,
    )

with st.expander("Record New Follow-Up for this Student"):
    with st.form("new_followup_form"):
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            t_name = st.text_input("Teacher Name / Initials", value="Academic Advisor")
        with fc2:
            act_type = st.selectbox("Action Type", ["Remedial Plan", "1-on-1 Consultation", "Study Skills Check-in", "Attendance Review", "Parent Contact"])
        with fc3:
            f_status = st.selectbox("Status", ["Planned", "In Progress", "Completed"])

        f_notes = st.text_area("Session Notes & Action Items", f"Discussed Series 1 ({s1}) & Series 2 ({s2}) scores. Target: {settings.target_score:.0f}.")
        sub_btn = st.form_submit_button("Save Follow-Up Record")

        if sub_btn:
            log_followup(
                student_id=int(selected_id),
                note=f_notes,
                action_type=act_type,
                status=f_status,
                teacher=t_name,
            )
            st.success(f"Follow-up logged successfully for Student #{selected_id}!")
            st.rerun()
