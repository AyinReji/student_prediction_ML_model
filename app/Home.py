import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
_app_dir = Path(__file__).resolve().parent
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))

import numpy as np
import plotly.express as px
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
from src.db import get_followups
from src.features import DEFAULT_FEATURE_COLS
from src.priority import compute_priority_components

st.set_page_config(
    page_title="Student Performance & Support System",
    page_icon="🎓",
    layout="wide",
)

apply_custom_styles()
settings = render_sidebar_controls()

st.title("🎓 Student Performance & Personalized Support System")
st.markdown(
    "A teacher-facing decision-support dashboard predicting final exam outcomes, "
    "flagging students needing assistance against customizable targets, and prescribing actionable, evidence-based recommendations."
)

model = load_app_model()
train_df, test_df = load_app_data()
metrics, oof_res = load_metrics_and_residuals()

if model is None or test_df.empty:
    st.warning("Model or data not ready. Please execute: `python -m src.data` and `python -m src.train`.")
    st.stop()

# Score the cohort (using test_df as active evaluation cohort or all students)
X_test = test_df[DEFAULT_FEATURE_COLS]
y_pred = model.predict(X_test)
residual_std = oof_res.get("residual_std", 6.5)

scored_cohort = compute_priority_components(
    df=test_df,
    predicted_scores=y_pred,
    target_score=settings.target_score,
    residual_std=residual_std,
    settings=settings,
)

# Follow-ups count
all_followups = get_followups()
total_followups_logged = len(all_followups)

at_risk_mask = scored_cohort["predicted_score"] < (settings.target_score + settings.risk_margin)
at_risk_count = int(np.sum(at_risk_mask))
avg_predicted = float(np.mean(y_pred))
total_students = len(test_df)
high_risk_count = int(np.sum(scored_cohort["risk_badge"] == "High"))

# Metric Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Active Cohort Students</div>
        <div class="metric-value">{total_students:,}</div>
        <div class="metric-subtitle">Held-out test set (20% sample)</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">At-Risk Students</div>
        <div class="metric-value" style="color: #DC2626;">{at_risk_count}</div>
        <div class="metric-subtitle">Target &lt; {settings.target_score:.0f} + {settings.risk_margin:.0f} pts buffer ({round(at_risk_count/total_students*100, 1)}%)</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Mean Predicted Score</div>
        <div class="metric-value">{avg_predicted:.1f} / 100</div>
        <div class="metric-subtitle">Cohort baseline average</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Interventions Logged</div>
        <div class="metric-value" style="color: #2F5BEA;">{total_followups_logged}</div>
        <div class="metric-subtitle">Recorded in tracking database</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

col_left, col_right = st.columns([1.8, 1.2])

with col_left:
    st.subheader("📊 Cohort Predicted Score Distribution & Risk Thresholds")
    fig = px.histogram(
        scored_cohort,
        x="predicted_score",
        nbins=35,
        color="risk_badge",
        color_discrete_map={"High": "#EF4444", "Medium": "#F59E0B", "Low": "#10B981"},
        title="Distribution of Predicted Final Exam Scores by Risk Level",
        labels={"predicted_score": "Predicted Score (Points)", "risk_badge": "Risk Badge"},
    )
    fig.add_vline(
        x=settings.target_score,
        line_dash="dash",
        line_color="#DC2626",
        annotation_text=f"Target ({settings.target_score})",
    )
    fig.add_vline(
        x=settings.target_score + settings.risk_margin,
        line_dash="dot",
        line_color="#F59E0B",
        annotation_text=f"Risk Threshold ({settings.target_score + settings.risk_margin})",
    )
    fig.update_layout(
        template="plotly_white",
        bargap=0.08,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("🤖 Production Model Specification")
    chosen_model_name = metrics.get("chosen_model", "LinearRegression")
    st.markdown(f"""
    - **Active Model:** `{chosen_model_name}` with StandardScaler
    - **Cross-Validation RMSE:** `{metrics.get('cv_results', {}).get(chosen_model_name, {}).get('rmse_mean', 0.0):.3f}`
    - **Held-Out Test MAE:** `{metrics.get('test_regression_metrics', {}).get(chosen_model_name, {}).get('mae', 0.0):.2f}` points
    - **Test R² Variance Explained:** `{metrics.get('test_regression_metrics', {}).get(chosen_model_name, {}).get('r2', 0.0):.3f}`
    - **90% Prediction Interval Width:** `±{oof_res.get('interval_half_width', 0.0):.1f}` points
    - **Empirical Test Coverage:** `{metrics.get('prediction_interval', {}).get('test_coverage_pct', 0.0)}%`
    """)

    st.markdown("""
    <div class="disclaimer-box">
        <b>Model Selection Rationale:</b><br>
        5-fold CV proved that Linear Regression, Random Forest, and Gradient Boosting all achieve similar accuracy (~5.2 MAE, 0.76 R²). 
        Under the 1-standard-error rule, Linear Regression was chosen for absolute transparency and explainability.
    </div>
    """, unsafe_allow_html=True)

st.subheader("🚀 Quick Navigation")
nav1, nav2, nav3, nav4 = st.columns(4)
with nav1:
    st.markdown("🎯 **[Priority Queue](Priority_Queue)**\n\nRank students by intervention need within teacher capacity.")
with nav2:
    st.markdown("🔍 **[Student Detail](Student_Detail)**\n\nDeep-dive with score intervals, point contributions, and follow-up logging.")
with nav3:
    st.markdown("📈 **[Progress Tracking](Progress_Tracking)**\n\nLog subsequent assessments and monitor trajectories over time.")
with nav4:
    st.markdown("🔬 **[Model Evaluation](Model_Evaluation)**\n\nCompare 3 algorithms, fairness audits, and baseline comparisons.")
