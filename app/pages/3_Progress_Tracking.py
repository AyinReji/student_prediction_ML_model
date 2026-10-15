import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
_app_dir = _repo_root / "app"
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))

import pandas as pd
import streamlit as st

from app.ui_utils import apply_custom_styles, render_sidebar_controls
from src.db import (
    PROGRESS_CAVEAT,
    get_followups,
    get_student_progress,
    log_assessment,
)

st.set_page_config(page_title="Progress Tracking | Student Support", page_icon="📈", layout="wide")
apply_custom_styles()
settings = render_sidebar_controls()

st.title("Progress Tracking & Follow-Up Log")
st.caption("Monitor logged academic interventions, pending actions, and subsequent student assessment results.")

st.caption(f"**Interpretation Note:** {PROGRESS_CAVEAT}")

# ── Summary Metrics ───────────────────────────────────────────────────
followups = get_followups()
df_fl = pd.DataFrame(followups) if followups else pd.DataFrame()

if not df_fl.empty:
    students_with_followup = df_fl["student_id"].nunique()
    completed_count = int((df_fl["status"] == "Completed").sum())
    pending_count = len(df_fl) - completed_count
else:
    students_with_followup = 0
    completed_count = 0
    pending_count = 0

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Students with Follow-Ups</div>
        <div class="metric-value">{students_with_followup}</div>
        <div class="metric-subtitle">Distinct students engaged</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Completed Interventions</div>
        <div class="metric-value" style="color: #16A34A;">{completed_count}</div>
        <div class="metric-subtitle">Resolved check-ins</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Pending / Planned</div>
        <div class="metric-value" style="color: #D97706;">{pending_count}</div>
        <div class="metric-subtitle">Awaiting review or follow-up</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

tab_log, tab_add_assess, tab_trajectory = st.tabs([
    "Intervention Log",
    "Record Subsequent Assessment",
    "Individual Student Trajectory",
])

with tab_log:
    st.subheader("Logged Student Interventions")
    if df_fl.empty:
        st.info("No interventions logged yet. Teachers can record sessions from Student Detail or Model Prediction.")
    else:
        st.dataframe(
            df_fl[["id", "student_id", "date", "teacher", "action_type", "status", "notes"]],
            use_container_width=True,
            hide_index=True,
        )

with tab_add_assess:
    st.subheader("Record Subsequent Assessment")
    with st.form("add_assessment_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            stu_id = st.number_input("Student ID", min_value=0, max_value=99999, value=0, step=1)
        with c2:
            assess_name = st.selectbox("Assessment Name", ["Series 3", "Remedial Quiz 1", "Practice Test", "Midterm Review"])
        with c3:
            score = st.number_input("Score Obtained", min_value=0.0, max_value=100.0, value=25.0, step=0.5)
        with c4:
            max_score = st.number_input("Max Score", min_value=1.0, max_value=100.0, value=40.0, step=1.0)

        sub_btn = st.form_submit_button("Save Assessment Record")
        if sub_btn:
            log_assessment(
                student_id=int(stu_id),
                assessment_name=assess_name,
                score=float(score),
                max_score=float(max_score),
            )
            st.success(f"Saved {assess_name} ({score}/{max_score}) for Student #{stu_id}!")

with tab_trajectory:
    st.subheader("Student Assessment History")
    track_id = st.number_input("Enter Student ID to inspect:", min_value=0, max_value=99999, value=0, step=1)
    progress_info = get_student_progress(int(track_id))
    assess_list = progress_info.get("assessments", [])

    if not assess_list:
        st.info(f"No subsequent assessment observations recorded for Student #{track_id} yet.")
    else:
        st.dataframe(
            pd.DataFrame(assess_list)[["assessment_name", "score", "max_score", "score_pct", "date"]],
            use_container_width=True,
            hide_index=True,
        )
