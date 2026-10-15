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

try:
    from app.ui_utils import apply_custom_styles, render_sidebar_controls
except ImportError:
    from ui_utils import apply_custom_styles, render_sidebar_controls
from src.data import audit_data, load_raw_data

st.set_page_config(page_title="Data Audit | Student Support", page_icon="📑", layout="wide")
apply_custom_styles()
settings = render_sidebar_controls()

st.title("📑 Exploratory Data & Integrity Audit")
st.markdown("Pre-modelling inspection of distributions, domain boundaries, outlier profiles, and authenticity characteristics.")

df = load_raw_data()
audit = audit_data(df)

# High level overview
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Total Records", f"{audit['n_rows']:,}")
with m2:
    st.metric("Features Checked", f"{audit['n_cols']}")
with m3:
    st.metric("Missing Values", f"{sum(audit['null_counts'].values())}")
with m4:
    st.metric("Duplicate Rows", f"{audit['num_duplicates']}")

st.markdown("---")

# Authenticity & Simulated Data Characteristics Box
st.markdown("""
<div class="disclaimer-box">
    <b>Critical Assessment of Dataset Authenticity:</b><br>
    <ul>
        <li><b>Academic Series Domination:</b> <code>series_1</code> and <code>series_2</code> correlate strongly with the final exam (~0.82–0.83).</li>
        <li><b>Absence of Lifestyle Signal:</b> <code>study_hours_per_day</code> (r = 0.011), <code>attendance_percentage</code> (r = -0.005), <code>social_media_hours</code> (r = 0.002), and <code>age</code> (r = 0.009) show virtually zero correlation with the final exam score or with each other.</li>
        <li><b>Simulated Data Pattern:</b> Real educational data universally displays modest positive correlations between attendance/study time and exam outcomes. The exact independence of these features strongly suggests synthetic/simulated generation.</li>
        <li><b>Identifiers & Granularity:</b> The dataset contains no timestamps, student identifiers, or subject/topic questions. Row indices serve as pseudo IDs, and topic-level diagnosis is strictly disallowed.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

col_corr, col_outliers = st.columns([1.1, 1.2])

with col_corr:
    st.subheader("Correlation Heatmap")
    corr = audit["corr_matrix"]
    fig_corr = px.imshow(
        corr,
        text_auto=".3f",
        aspect="auto",
        color_continuous_scale="Blues",
        title="Pearson Correlation Matrix (Raw Data)",
    )
    fig_corr.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_corr, use_container_width=True)

with col_outliers:
    st.subheader("Distribution & Outlier Bounds (1.5 × IQR)")
    outliers = audit["outlier_counts"]
    out_table = []
    for col, v in outliers.items():
        out_table.append({
            "Feature": col,
            "Q1": round(v["q1"], 2),
            "Q3": round(v["q3"], 2),
            "IQR": round(v["iqr"], 2),
            "Valid Bounds": f"[{v['lower_bound']:.1f}, {v['upper_bound']:.1f}]",
            "Outlier Count": v["count"],
            "Outlier %": f"{v['pct']:.2f}%",
        })
    st.dataframe(pd.DataFrame(out_table), use_container_width=True, hide_index=True)
    st.caption("ℹ️ Note: Outliers represent valid domain points in academic distributions and are retained.")

st.markdown("---")

# Demographics summary
st.subheader("Demographic Distributions")
dc1, dc2 = st.columns(2)
with dc1:
    st.markdown("#### Gender Balance")
    gdf = pd.DataFrame(list(audit["gender_counts"].items()), columns=["Gender", "Count"])
    fig_g = px.pie(gdf, names="Gender", values="Count", color="Gender", color_discrete_sequence=["#2F5BEA", "#10B981", "#F59E0B"], hole=0.4)
    fig_g.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_g, use_container_width=True)

with dc2:
    st.markdown("#### Age Distribution")
    fig_age = px.histogram(df, x="age", nbins=11, color_discrete_sequence=["#2F5BEA"], title="Student Age Spread (14–24)")
    fig_age.update_layout(template="plotly_white", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig_age, use_container_width=True)
