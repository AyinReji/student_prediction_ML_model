import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
_app_dir = _repo_root / "app"
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))

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
from src.features import DEFAULT_FEATURE_COLS
from src.priority import build_priority_queue

st.set_page_config(page_title="Priority Queue | Student Support", page_icon="📋", layout="wide")
apply_custom_styles()
settings = render_sidebar_controls()

st.title("📋 Teacher Priority Queue")
st.markdown(
    f"Identifies the top **{settings.capacity}** students prioritized for academic check-in, "
    "blending estimated exam failure risk, assessment momentum, and rule alerts."
)

model = load_app_model()
train_df, test_df = load_app_data()
metrics, oof_res = load_metrics_and_residuals()

if model is None or test_df.empty:
    st.warning("Model or data not ready. Please train the system first.")
    st.stop()

X_test = test_df[DEFAULT_FEATURE_COLS]
y_pred = model.predict(X_test)
residual_std = oof_res.get("residual_std", 6.5)

# Build priority queue for top N students
queue_df = build_priority_queue(
    df=test_df,
    predicted_scores=y_pred,
    residual_std=residual_std,
    settings=settings,
    capacity=settings.capacity,
)

# Filters
col_f1, col_f2, col_f3 = st.columns([1, 1, 2])
with col_f1:
    risk_filter = st.multiselect(
        "Filter by Risk Level",
        options=["High", "Medium", "Low"],
        default=["High", "Medium", "Low"],
    )
with col_f2:
    search_id = st.text_input("Search Student ID", "")

filtered_queue = queue_df[queue_df["risk_badge"].isin(risk_filter)]
if search_id.strip():
    try:
        sid_int = int(search_id.strip())
        filtered_queue = filtered_queue[filtered_queue["student_id"] == sid_int]
    except ValueError:
        st.warning("Please enter a numeric student ID.")

st.markdown(f"**Displaying {len(filtered_queue)} of {settings.capacity} prioritized students:**")

# Format columns for display
display_cols = [
    "rank",
    "student_id",
    "predicted_score",
    "risk_badge",
    "priority_score",
    "series_1",
    "series_2",
    "series_delta",
    "top_reason",
    "suggested_action",
]

# Rename for clean UI table
table_view = filtered_queue[display_cols].copy()
table_view.columns = [
    "Rank",
    "Student ID",
    "Predicted Score",
    "Risk Badge",
    "Priority Score",
    "Series 1 (/40)",
    "Series 2 (/40)",
    "Series Delta",
    "Primary Reason",
    "Recommended Action",
]

st.dataframe(
    table_view,
    use_container_width=True,
    hide_index=True,
)

# CSV Export
csv_data = filtered_queue.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Export Priority Queue (CSV)",
    data=csv_data,
    file_name=f"teacher_priority_queue_top_{settings.capacity}.csv",
    mime="text/csv",
)

st.markdown("---")
st.info(
    "💡 **Next Steps:** Note a student's ID and navigate to **Student Detail** in the sidebar to review detailed "
    "point-attribution charts and log support sessions."
)
