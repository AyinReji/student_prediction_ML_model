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
    load_app_model,
    load_metrics_and_residuals,
    render_sidebar_controls,
)
from src.db import init_db, log_followup
from src.predict import predict_student_performance

st.set_page_config(
    page_title="Predict Final Marks | Student Support",
    page_icon="🎯",
    layout="wide",
)

apply_custom_styles()
settings = render_sidebar_controls()
init_db()

st.title("Predict Final Examination Marks")
st.caption("Enter academic assessment marks and relevant lifestyle attributes to forecast final examination performance.")

model = load_app_model()
metrics, oof_res = load_metrics_and_residuals()

if model is None:
    st.error("Prediction model not available. Please ensure the model is trained.")
    st.stop()

# ── 1. Compact Two-Section Input Form ─────────────────────────────────
with st.form("mark_prediction_form"):
    col_acad, col_life = st.columns(2)

    with col_acad:
        st.markdown("##### 1. Academic Performance")
        s1 = st.number_input(
            "Series 1 Marks (0.0 – 40.0)",
            min_value=0.0,
            max_value=40.0,
            value=22.0,
            step=0.5,
            help="First internal assessment mark out of 40.",
        )
        s2 = st.number_input(
            "Series 2 Marks (0.0 – 40.0)",
            min_value=0.0,
            max_value=40.0,
            value=24.5,
            step=0.5,
            help="Second internal assessment mark out of 40.",
        )

    with col_life:
        st.markdown("##### 2. Lifestyle and Study Habits")
        study_hrs = st.number_input(
            "Daily Study Time (hrs/day)",
            min_value=0.0,
            max_value=16.0,
            value=3.0,
            step=0.5,
            help="Daily self-study hours.",
        )
        attendance = st.number_input(
            "Class Attendance (%)",
            min_value=0.0,
            max_value=100.0,
            value=85.0,
            step=1.0,
            help="Overall attendance percentage.",
        )
        social_hrs = st.number_input(
            "Social Media Usage (hrs/day)",
            min_value=0.0,
            max_value=16.0,
            value=2.0,
            step=0.5,
            help="Daily recreational screen time.",
        )

    submitted = st.form_submit_button("Predict Final Marks", type="primary", use_container_width=True)

# ── 2. Validation & Inference ─────────────────────────────────────────
if submitted:
    raw_payload = {
        "series_1": s1,
        "series_2": s2,
        "study_hours_per_day": study_hrs,
        "attendance_percentage": attendance,
        "social_media_hours": social_hrs,
    }

    try:
        prediction = predict_student_performance(
            raw_inputs=raw_payload,
            model=model,
            oof_meta=oof_res,
            settings=settings,
        )
        st.session_state["active_prediction"] = prediction
        st.session_state["active_inputs"] = raw_payload
    except ValueError as ve:
        st.error(f"Validation Error: {ve}")
        st.stop()
    except Exception as e:
        st.error(f"Inference Error: {e}")
        st.stop()

# ── 3. Prominent Prediction Results ───────────────────────────────────
prediction = st.session_state.get("active_prediction")
inputs = st.session_state.get("active_inputs")

if prediction is not None and inputs is not None:
    st.markdown("---")
    pred_score = prediction["predicted_score"]
    interval = prediction["prediction_interval"]
    risk = prediction["risk_evaluation"]

    # Academic status based on target and margin
    if pred_score < settings.target_score:
        status_text = "Below Target"
        status_color = "#DC2626"
        badge_style = "risk-badge-high"
    elif pred_score < (settings.target_score + settings.risk_margin):
        status_text = "Near Target (Borderline)"
        status_color = "#D97706"
        badge_style = "risk-badge-medium"
    else:
        status_text = "On Track"
        status_color = "#16A34A"
        badge_style = "risk-badge-low"

    r1, r2, r3 = st.columns(3)

    with r1:
        st.markdown(f"""
        <div class="metric-card" style="border-top: 4px solid {status_color};">
            <div class="metric-title">Predicted Final Score</div>
            <div class="metric-value" style="color: {status_color};">
                {pred_score:.1f} <span style="font-size: 1rem; color: #64748B;">/ 100</span>
            </div>
            <div class="metric-subtitle">Forecasted final examination mark</div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Prediction Interval (90%)</div>
            <div class="metric-value" style="font-size: 1.45rem;">
                [{interval['lower']:.1f}, {interval['upper']:.1f}]
            </div>
            <div class="metric-subtitle">Calibrated expected score range (±{interval['half_width']:.1f} pts)</div>
        </div>
        """, unsafe_allow_html=True)

    with r3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Academic Risk Category</div>
            <div class="metric-value"><span class="{badge_style}">{status_text}</span></div>
            <div class="metric-subtitle">Risk Probability: <b>{risk['risk_probability_pct']:.1f}%</b> against Target {settings.target_score:.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── 4. Contributing Factors Explanation ───────────────────────────
    with st.expander("How was this prediction generated?"):
        st.markdown(f"""
        - **Model Architecture:** `StandardScaler + LinearRegression`
        - **Predictive Inputs:** Series 1 (`{inputs['series_1']:.1f}`), Series 2 (`{inputs['series_2']:.1f}`), Study Hours (`{inputs['study_hours_per_day']:.1f}`), Attendance (`{inputs['attendance_percentage']:.1f}%`), Social Media (`{inputs['social_media_hours']:.1f}h`).
        - **Formula Decomposition:**
          $$\\text{{Final Score}} = 13.78 + (0.962 \\times S_1) + (0.929 \\times S_2) + (0.035 \\times \\text{{Study}}) - (0.001 \\times \\text{{Att}}) + (0.030 \\times \\text{{Social}}) = \\mathbf{{{pred_score:.1f}}}$$
        """)

        # Attribution across academic and lifestyle attributes
        contribs = prediction.get("contributions", [])
        if contribs:
            st.markdown("**Factor Attribution Relative to Cohort Benchmark:**")
            attr_table = []
            for c in contribs:
                attr_table.append({
                    "Feature": c["display_name"],
                    "Input Value": f"{c['value']:.1f}",
                    "Score Impact": f"{c['points']:+.2f} pts",
                    "Direction": "Positive Impact" if c["points"] >= 0 else "Negative Impact",
                })
            st.dataframe(pd.DataFrame(attr_table), use_container_width=True, hide_index=True)

    # ── 5. Relevant Support Recommendations ───────────────────────────
    rules_fired = prediction.get("rules_fired", [])
    if risk["is_at_risk"] or rules_fired:
        st.markdown("---")
        st.subheader("Personalized Academic Support Recommendations")

        if rules_fired:
            for rule in rules_fired:
                is_ind = rule.get("is_indicator_only", False)
                card_class = "rule-card rule-indicator" if is_ind else "rule-card"
                badge = "Lifestyle Habit Note" if is_ind else "Academic Assessment Alert"
                st.markdown(f"""
                <div class="{card_class}">
                    <b>{rule['name']}</b> <span style="font-size: 0.75rem; color: #64748B;">[{badge}]</span><br>
                    <span style="color: #1E293B;"><b>Action:</b> {rule['action']}</span><br>
                    <span style="color: #64748B; font-style: italic; font-size: 0.8rem;">Context: {rule['info_note']}</span>
                </div>
                """, unsafe_allow_html=True)

        with st.expander("Record a Support Follow-Up in Database"):
            note_text = st.text_area(
                "Intervention Notes",
                value=f"Predicted final score: {pred_score:.1f}/100. Series 1: {inputs['series_1']}, Series 2: {inputs['series_2']}, Study: {inputs['study_hours_per_day']}h, Attendance: {inputs['attendance_percentage']}%.",
            )
            if st.button("Save Follow-Up Record"):
                log_followup(
                    student_id=9999,
                    note=note_text,
                    action_type="Academic Consultation",
                    status="Planned",
                )
                st.success("Follow-up successfully saved to tracking database.")
else:
    st.info("Enter the student's academic marks and lifestyle attributes above, then click **Predict Final Marks**.")
