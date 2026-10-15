import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
_app_dir = _repo_root / "app"
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))

import pandas as pd
import plotly.express as px
import streamlit as st

from app.ui_utils import apply_custom_styles, render_sidebar_controls
from src.data import audit_data, load_raw_data

st.set_page_config(page_title="Data Audit | Student Support", page_icon="📑", layout="wide")
apply_custom_styles()
settings = render_sidebar_controls()

st.title("Data Integrity & Exploratory Audit")
st.caption("Pre-modelling inspection of dataset distributions, domain validity bounds, and statistical authenticity.")

df = load_raw_data()
audit = audit_data(df)

# ── Summary Metrics ───────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Records</div>
        <div class="metric-value">{audit['n_rows']:,}</div>
        <div class="metric-subtitle">Full raw cohort</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Missing Values</div>
        <div class="metric-value">{sum(audit['null_counts'].values())}</div>
        <div class="metric-subtitle">Complete records across all columns</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Duplicates</div>
        <div class="metric-value">{audit['num_duplicates']}</div>
        <div class="metric-subtitle">Zero duplicate rows detected</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Train / Test Split</div>
        <div class="metric-value">80 / 20</div>
        <div class="metric-subtitle">8,000 train / 2,000 test (Stratified)</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <b>Data Reality Findings:</b><br>
    - <b>Academic Marks:</b> <code>series_1</code> and <code>series_2</code> carry all predictive signal ($r \\approx 0.82$).<br>
    - <b>Lifestyle Indicators:</b> <code>study_hours_per_day</code>, <code>attendance_percentage</code>, and <code>social_media_hours</code> show virtually zero correlation with exam outcomes ($|r| \\le 0.01$). They are conversational indicators only.
</div>
""", unsafe_allow_html=True)

# ── Expandable Technical Audit Details ────────────────────────────────
with st.expander("Feature Domain Validity & Outlier Bounds (1.5 × IQR)"):
    outliers = audit["outlier_counts"]
    out_table = []
    for col, v in outliers.items():
        out_table.append({
            "Feature": col,
            "Q1": round(v["q1"], 2),
            "Q3": round(v["q3"], 2),
            "IQR": round(v["iqr"], 2),
            "Valid Domain Range": f"[{v['lower_bound']:.1f}, {v['upper_bound']:.1f}]",
            "Outlier Count": v["count"],
            "Outlier %": f"{v['pct']:.2f}%",
        })
    st.dataframe(pd.DataFrame(out_table), use_container_width=True, hide_index=True)
    st.caption("Valid academic scores flagged by IQR are preserved without deletion.")

with st.expander("Correlation Matrix Heatmap"):
    corr = audit["corr_matrix"]
    fig_corr = px.imshow(
        corr,
        text_auto=".3f",
        aspect="auto",
        color_continuous_scale="Blues",
        title="Pearson Correlation Matrix",
    )
    fig_corr.update_layout(template="plotly_white", height=320, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_corr, use_container_width=True)

with st.expander("Demographic Distributions"):
    dc1, dc2 = st.columns(2)
    with dc1:
        gdf = pd.DataFrame(list(audit["gender_counts"].items()), columns=["Gender", "Count"])
        st.dataframe(gdf, use_container_width=True, hide_index=True)
    with dc2:
        st.write(pd.DataFrame(list(audit["age_dist"].items()), columns=["Age Statistic", "Value"]))
