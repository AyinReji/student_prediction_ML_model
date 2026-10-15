import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
_app_dir = _repo_root / "app"
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))

import streamlit as st

from app.ui_utils import apply_custom_styles, render_sidebar_controls
from src.config import load_settings, save_settings

st.set_page_config(page_title="Settings | Student Support", page_icon="⚙️", layout="wide")
apply_custom_styles()
settings = render_sidebar_controls()

st.title("System Configuration & Pedagogical Settings")
st.caption("Adjust operational pass targets, capacity limits, and rule thresholds. Changes persist to `config/settings.yaml`.")

current = load_settings()

with st.form("settings_form"):
    st.subheader("1. Operational Targets & Capacity")
    c1, c2, c3 = st.columns(3)
    with c1:
        target_score = st.number_input(
            "Target Pass Mark",
            min_value=10.0,
            max_value=90.0,
            value=float(current.target_score),
            step=1.0,
            help="Minimum target final exam score for passing (default: 40.0).",
        )
    with c2:
        risk_margin = st.number_input(
            "Risk Buffer Margin (pts)",
            min_value=0.0,
            max_value=15.0,
            value=float(current.risk_margin),
            step=0.5,
            help="Buffer added to target score to identify students near the threshold.",
        )
    with c3:
        capacity = st.number_input(
            "Teacher Follow-Up Capacity (N)",
            min_value=5,
            max_value=200,
            value=int(current.capacity),
            step=5,
            help="Maximum number of students prioritized for intervention.",
        )

    st.markdown("---")
    st.subheader("2. Recommendation Engine Thresholds")
    r1, r2 = st.columns(2)
    with r1:
        low_series_avg = st.number_input("Low Assessment Series Threshold (/40)", min_value=5.0, max_value=35.0, value=float(current.rules.low_series_avg), step=0.5)
        decline_points = st.number_input("Assessment Trajectory Decline Warning (pts)", min_value=1.0, max_value=20.0, value=float(current.rules.decline_points), step=0.5)
        min_attendance = st.number_input("Minimum Attendance Warning (%)", min_value=40.0, max_value=100.0, value=float(current.rules.min_attendance), step=1.0)
    with r2:
        min_study_hours = st.number_input("Minimum Study Hours Warning (hrs/day)", min_value=0.5, max_value=6.0, value=float(current.rules.min_study_hours), step=0.5)
        max_social_hours = st.number_input("Maximum Social Media Warning (hrs/day)", min_value=1.0, max_value=10.0, value=float(current.rules.max_social_hours), step=0.5)

    st.markdown("---")
    st.subheader("3. Priority Queue Scoring Weights (Sum ~ 1.0)")
    w1, w2, w3 = st.columns(3)
    with w1:
        w_risk = st.number_input("Weight: Risk Probability", min_value=0.0, max_value=1.0, value=float(current.priority_weights.risk), step=0.05)
    with w2:
        w_decline = st.number_input("Weight: Trajectory Decline", min_value=0.0, max_value=1.0, value=float(current.priority_weights.decline), step=0.05)
    with w3:
        w_flags = st.number_input("Weight: Fired Rule Share", min_value=0.0, max_value=1.0, value=float(current.priority_weights.flags), step=0.05)

    w_sum = round(w_risk + w_decline + w_flags, 2)
    if w_sum != 1.0:
        st.warning(f"Current weights sum to {w_sum} (Recommended sum = 1.0).")

    st.markdown("---")
    st.subheader("4. Risk Badge Cutoffs")
    b1, b2 = st.columns(2)
    with b1:
        badge_high = st.slider("High Risk Probability Cutoff", min_value=0.4, max_value=0.9, value=float(current.risk_badges.high), step=0.05)
    with b2:
        badge_med = st.slider("Medium Risk Probability Cutoff", min_value=0.1, max_value=0.5, value=float(current.risk_badges.medium), step=0.05)

    submitted = st.form_submit_button("Save Settings", type="primary")
    if submitted:
        current.target_score = target_score
        current.risk_margin = risk_margin
        current.capacity = int(capacity)
        current.rules.low_series_avg = low_series_avg
        current.rules.decline_points = decline_points
        current.rules.min_attendance = min_attendance
        current.rules.min_study_hours = min_study_hours
        current.rules.max_social_hours = max_social_hours
        current.priority_weights.risk = w_risk
        current.priority_weights.decline = w_decline
        current.priority_weights.flags = w_flags
        current.risk_badges.high = badge_high
        current.risk_badges.medium = badge_med

        save_settings(current)
        st.success("Settings saved successfully to `config/settings.yaml`.")
