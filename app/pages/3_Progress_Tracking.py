import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
_app_dir = _repo_root / "app"
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from app.ui_utils import apply_custom_styles, render_sidebar_controls
except ImportError:
    from ui_utils import apply_custom_styles, render_sidebar_controls
from src.db import (
    PROGRESS_CAVEAT,
    get_followups,
    get_student_progress,
    log_assessment,
)

st.set_page_config(page_title="Progress Tracking | Student Support", page_icon="📈", layout="wide")
apply_custom_styles()
settings = render_sidebar_controls()

st.title("📈 Progress Tracking & Longitudinal Assessments")
st.markdown("Track intervention history, record post-intervention assessments, and observe trajectory shifts.")

# Prominent Research Caveat Box
st.warning(f"⚠️ **Research & Interpretation Caveat:**\n\n{PROGRESS_CAVEAT}")

tab1, tab2, tab3 = st.tabs(["📋 Follow-Up Log", "➕ Record New Assessment", "📊 Student Progress Trajectory"])

with tab1:
    st.subheader("Logged Student Interventions")
    followups = get_followups()
    if not followups:
        st.info("No follow-ups logged yet. You can log follow-ups from the Student Detail page.")
    else:
        df_fl = pd.DataFrame(followups)
        st.dataframe(
            df_fl[["id", "student_id", "date", "teacher", "action_type", "status", "notes"]],
            use_container_width=True,
            hide_index=True,
        )

with tab2:
    st.subheader("Record Subsequent Assessment Result")
    with st.form("new_assessment_form"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            stu_id = st.number_input("Student ID", min_value=0, max_value=9999, value=0, step=1)
        with c2:
            assess_name = st.selectbox(
                "Assessment Title",
                ["Series 3 Exam", "Midterm Assessment", "Remedial Quiz 1", "Practice Test 1", "Topic Review Quiz"],
            )
        with c3:
            score = st.number_input("Score Obtained", min_value=0.0, max_value=100.0, value=25.0, step=0.5)
        with c4:
            max_score = st.number_input("Max Possible Score", min_value=1.0, max_value=100.0, value=40.0, step=1.0)

        submitted = st.form_submit_button("💾 Save Assessment Record")
        if submitted:
            log_assessment(
                student_id=int(stu_id),
                assessment_name=assess_name,
                score=float(score),
                max_score=float(max_score),
            )
            st.success(f"Assessment '{assess_name}' ({score}/{max_score}) successfully saved for Student #{stu_id}!")

with tab3:
    st.subheader("Student Longitudinal Trajectory")
    track_student_id = st.number_input("Enter Student ID to inspect progress:", min_value=0, max_value=9999, value=0, step=1)

    progress = get_student_progress(track_student_id)
    assessments = progress.get("assessments", [])

    if not assessments:
        st.info(f"No assessment records found for Student #{track_student_id}.")
    else:
        # Metric cards for progress
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Baseline Average</div>
                <div class="metric-value">{progress['before_avg_pct']:.1f}%</div>
                <div class="metric-subtitle">Series 1 & 2 baseline score</div>
            </div>
            """, unsafe_allow_html=True)

        with pc2:
            after_str = f"{progress['after_avg_pct']:.1f}%" if progress['after_avg_pct'] is not None else "Pending"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Post-Followup Average</div>
                <div class="metric-value">{after_str}</div>
                <div class="metric-subtitle">Subsequent assessment results</div>
            </div>
            """, unsafe_allow_html=True)

        with pc3:
            delta = progress.get("delta_pct")
            delta_str = f"{delta:+.1f}%" if delta is not None else "N/A"
            color_str = "#10B981" if delta and delta > 0 else ("#DC2626" if delta and delta < 0 else "#64748B")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Observed Percentage Change</div>
                <div class="metric-value" style="color: {color_str};">{delta_str}</div>
                <div class="metric-subtitle">Trajectory difference</div>
            </div>
            """, unsafe_allow_html=True)

        # Plotly chart of assessment scores over time
        df_chart = pd.DataFrame(assessments)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_chart["name"],
            y=df_chart["pct"],
            mode="lines+markers+text",
            text=[f"{p:.1f}% ({s}/{m})" for p, s, m in zip(df_chart["pct"], df_chart["score"], df_chart["max_score"])],
            textposition="top center",
            line=dict(color="#2F5BEA", width=3),
            marker=dict(size=10, color="#2F5BEA"),
            name="Score %",
        ))

        fig.update_layout(
            title=f"Assessment Trajectory for Student #{track_student_id}",
            yaxis=dict(range=[0, 105], title="Score Percentage (%)"),
            template="plotly_white",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.caption(progress["caveat"])
