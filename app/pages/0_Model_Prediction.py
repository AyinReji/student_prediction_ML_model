import io
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
    info_btn,
    load_app_model,
    load_metrics_and_residuals,
    render_glossary_popover,
    render_sidebar_controls,
)
from src.db import init_db, log_followup
from src.predict import (
    BULK_ALL_FEATURE_COLUMNS,
    BULK_BOUNDS,
    BULK_OPTIONAL_COLUMNS,
    BULK_REQUIRED_COLUMNS,
    BULK_VALID_GENDERS,
    predict_bulk,
    predict_student_performance,
)

st.set_page_config(
    page_title="Predict Final Marks | Student Support",
    page_icon="🎯",
    layout="wide",
)

apply_custom_styles()
settings = render_sidebar_controls()
init_db()

col_head, col_info = st.columns([3.8, 1.4])
with col_head:
    st.title("Predict Final Examination Marks")
    st.caption(
        "Enter academic assessment marks, lifestyle habits, and demographic attributes "
        "to forecast final examination performance."
    )
with col_info:
    st.write("")
    st.write("")
    render_glossary_popover()

model = load_app_model()
metrics, oof_res = load_metrics_and_residuals()

if model is None:
    st.error("Prediction model not available. Please ensure the model is trained.")
    st.stop()

# ── Mode Toggle ──────────────────────────────────────────────────────────
tab_single, tab_bulk = st.tabs(["🎓 Single Student", "📂 Bulk CSV Prediction"])

# ═══════════════════════════════════════════════════════════════════════
# TAB 1 — SINGLE STUDENT  (original UI, untouched)
# ═══════════════════════════════════════════════════════════════════════
with tab_single:
    # ── 1. Compact Two-Section Input Form ─────────────────────────────────
    with st.form("mark_prediction_form"):
        col_acad, col_life, col_demo = st.columns(3)

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
            st.markdown("##### 2. Lifestyle & Study Habits")
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

        with col_demo:
            st.markdown("##### 3. Student Demographics")
            age = st.number_input(
                "Age (years)",
                min_value=10,
                max_value=100,
                value=18,
                step=1,
                help="Student's age in years.",
            )
            gender = st.selectbox(
                "Gender",
                options=["Female", "Male", "Other"],
                index=0,
                help="Student gender profile.",
            )

        submitted = st.form_submit_button(
            "Predict Final Marks", type="primary", use_container_width=True
        )

    # ── 2. Validation & Inference ─────────────────────────────────────────
    if submitted:
        raw_payload = {
            "series_1": s1,
            "series_2": s2,
            "study_hours_per_day": study_hrs,
            "attendance_percentage": attendance,
            "social_media_hours": social_hrs,
            "age": float(age),
            "gender": str(gender),
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

        # Exam eligibility check (Attendance < 75%)
        eligibility = prediction.get("exam_eligibility", {})
        is_eligible = eligibility.get("is_eligible", True)

        if not is_eligible:
            st.error(
                f"🚫 **EXAM INELIGIBILITY ALERT:** Attendance is **{inputs['attendance_percentage']:.1f}%** "
                f"(below the mandatory {eligibility.get('min_required_attendance', 75.0):.0f}% requirement). "
                "**Under institutional policy, this student cannot attend the final examination.**"
            )

        r1, r2, r3, r4 = st.columns(4)

        with r1:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid {status_color};">
                <div class="metric-title">Predicted Final Score {info_btn('predicted_final_score')}</div>
                <div class="metric-value" style="color: {status_color};">
                    {pred_score:.1f} <span style="font-size: 1rem; color: #64748B;">/ 100</span>
                </div>
                <div class="metric-subtitle">Forecasted final examination mark</div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Prediction Interval (90%) {info_btn('prediction_interval')}</div>
                <div class="metric-value" style="font-size: 1.45rem;">
                    [{interval['lower']:.1f}, {interval['upper']:.1f}]
                </div>
                <div class="metric-subtitle">Calibrated expected range (±{interval['half_width']:.1f} pts)</div>
            </div>
            """, unsafe_allow_html=True)

        with r3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Academic Risk Category {info_btn('academic_risk_category')}</div>
                <div class="metric-value"><span class="{badge_style}">{status_text}</span></div>
                <div class="metric-subtitle">Risk Probability: <b>{risk['risk_probability_pct']:.1f}%</b> {info_btn('risk_probability')}</div>
            </div>
            """, unsafe_allow_html=True)

        with r4:
            elig_color = "#16A34A" if is_eligible else "#DC2626"
            elig_badge = "risk-badge-low" if is_eligible else "risk-badge-high"
            elig_title = "Eligible to Sit Exam" if is_eligible else "Ineligible to Sit Exam"
            elig_sub = (
                f"{inputs['attendance_percentage']:.1f}% attendance (≥ 75%)"
                if is_eligible
                else f"{inputs['attendance_percentage']:.1f}% attendance (&lt; 75% policy)"
            )
            st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid {elig_color};">
                <div class="metric-title">Exam Eligibility {info_btn('exam_eligibility')}</div>
                <div class="metric-value" style="font-size: 1.25rem;"><span class="{elig_badge}">{elig_title}</span></div>
                <div class="metric-subtitle" style="color: {elig_color}; font-weight: 500;">{elig_sub}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── 4. Lifestyle Adjustment Breakdown ─────────────────────────────
        life_adj = prediction.get("lifestyle_adjustment", {})
        ml_base = prediction.get("ml_base_score", pred_score)
        if life_adj:
            adj_total = life_adj.get("total", 0)
            study_c = life_adj.get("study_contribution", 0)
            attend_c = life_adj.get("attendance_contribution", 0)
            social_c = life_adj.get("social_contribution", 0)
            adj_color = "#16A34A" if adj_total >= 0 else "#DC2626"
            adj_sign = "+" if adj_total >= 0 else ""

            st.markdown("---")
            st.markdown(f"##### 📊 Lifestyle & Study Habit Influence on Score {info_btn('lifestyle_influence')}", unsafe_allow_html=True)
            lc1, lc2, lc3, lc4 = st.columns(4)

            with lc1:
                st.markdown(f"""
                <div class="metric-card" style="border-top: 4px solid {adj_color};">
                    <div class="metric-title">Total Lifestyle Impact {info_btn('lifestyle_influence')}</div>
                    <div class="metric-value" style="color: {adj_color}; font-size: 1.6rem;">
                        {adj_sign}{adj_total:+.1f} pts
                    </div>
                    <div class="metric-subtitle">Net effect on final score</div>
                </div>
                """, unsafe_allow_html=True)

            study_color = "#16A34A" if study_c >= 0 else "#DC2626"
            with lc2:
                st.markdown(f"""
                <div class="metric-card" style="border-top: 3px solid {study_color};">
                    <div class="metric-title">📚 Study Hours {info_btn('study_hours_per_day')}</div>
                    <div class="metric-value" style="color: {study_color}; font-size: 1.4rem;">
                        {study_c:+.2f} pts
                    </div>
                    <div class="metric-subtitle">{inputs['study_hours_per_day']:.1f} h/day (+1.30 pts/hr)</div>
                </div>
                """, unsafe_allow_html=True)

            attend_color = "#16A34A" if attend_c >= 0 else "#DC2626"
            with lc3:
                st.markdown(f"""
                <div class="metric-card" style="border-top: 3px solid {attend_color};">
                    <div class="metric-title">🏫 Attendance {info_btn('attendance_percentage')}</div>
                    <div class="metric-value" style="color: {attend_color}; font-size: 1.4rem;">
                        {attend_c:+.2f} pts
                    </div>
                    <div class="metric-subtitle">{inputs['attendance_percentage']:.1f}% (+0.11 pts/%)</div>
                </div>
                """, unsafe_allow_html=True)

            social_color = "#16A34A" if social_c >= 0 else "#DC2626"
            with lc4:
                st.markdown(f"""
                <div class="metric-card" style="border-top: 3px solid {social_color};">
                    <div class="metric-title">📱 Social Media {info_btn('social_media_hours')}</div>
                    <div class="metric-value" style="color: {social_color}; font-size: 1.4rem;">
                        {social_c:+.2f} pts
                    </div>
                    <div class="metric-subtitle">{inputs['social_media_hours']:.1f} h/day (−0.65 pts/hr)</div>
                </div>
                """, unsafe_allow_html=True)

        # ── 5. Contributing Factors Explanation ───────────────────────────
        with st.expander("How was this prediction generated?"):
            life_adj = prediction.get("lifestyle_adjustment", {})
            ml_base = prediction.get("ml_base_score", pred_score)
            adj_total = life_adj.get("total", 0) if life_adj else 0
            st.markdown(f"""
            - **Model Architecture {info_btn('one_se_rule')}:** Multi-feature pipeline (`ColumnTransformer + LinearRegression`) fitted and tuned on 5,000 student records.
            - **Final Predicted Score {info_btn('predicted_final_score')}:** **{pred_score:.1f} / 100**
            - **Linear Model Equation Weights:**
              - **Series 1 Exam {info_btn('series_1')}:** `+0.838 pts` per mark
              - **Series 2 Exam {info_btn('series_2')}:** `+0.800 pts` per mark
              - **Daily Study Time {info_btn('study_hours_per_day')}:** `+1.305 pts` per hr/day
              - **Class Attendance {info_btn('attendance_percentage')}:** `+0.109 pts` per %
              - **Social Media Usage {info_btn('social_media_hours')}:** `−0.652 pts` per hr/day
              - **Student Age {info_btn('age')}:** `+0.030 pts` per year
              - **Gender {info_btn('gender')}:** Subgroup adjustment (+0.30 Male / -0.80 Other vs Female baseline)
            """, unsafe_allow_html=True)

            # Attribution across academic, lifestyle, and demographic attributes
            contribs = prediction.get("contributions", [])
            if contribs:
                st.markdown(f"**Score-Point Factor Attribution Relative to Cohort Average {info_btn('feature_attribution')}:**", unsafe_allow_html=True)
                attr_table = []
                for c in contribs:
                    val = c.get("value")
                    if isinstance(val, (int, float)):
                        val_str = f"{val:.1f}"
                    else:
                        try:
                            val_str = f"{float(val):.1f}"
                        except (ValueError, TypeError):
                            val_str = str(val)

                    attr_table.append({
                        "Feature": c["display_name"],
                        "Input Value": val_str,
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
        st.info(
            "Enter the student's academic marks and lifestyle attributes above, "
            "then click **Predict Final Marks**."
        )


# ═══════════════════════════════════════════════════════════════════════
# TAB 2 — BULK CSV PREDICTION
# ═══════════════════════════════════════════════════════════════════════
with tab_bulk:
    # ── Schema reference card ────────────────────────────────────────────
    st.markdown("#### 📋 Required CSV Schema")

    schema_rows = [
        {"Column": "series_1",              "Type": "float",  "Required": "✅ Yes", "Valid Range / Values": "0.0 – 40.0",              "Notes": "Series 1 internal assessment mark"},
        {"Column": "series_2",              "Type": "float",  "Required": "✅ Yes", "Valid Range / Values": "0.0 – 40.0",              "Notes": "Series 2 internal assessment mark"},
        {"Column": "study_hours_per_day",   "Type": "float",  "Required": "✅ Yes", "Valid Range / Values": "0.0 – 24.0",              "Notes": "Daily self-study hours"},
        {"Column": "attendance_percentage", "Type": "float",  "Required": "✅ Yes", "Valid Range / Values": "0.0 – 100.0",             "Notes": "Overall class attendance %"},
        {"Column": "social_media_hours",    "Type": "float",  "Required": "✅ Yes", "Valid Range / Values": "0.0 – 24.0",              "Notes": "Daily social media hours"},
        {"Column": "age",                   "Type": "float",  "Required": "⬜ No",  "Valid Range / Values": "10.0 – 100.0",            "Notes": "Age in years (default 18.0)"},
        {"Column": "gender",                "Type": "string", "Required": "⬜ No",  "Valid Range / Values": "Female / Male / Other",   "Notes": "Case-insensitive (default Female)"},
    ]
    st.dataframe(pd.DataFrame(schema_rows), use_container_width=True, hide_index=True)

    st.info(
        "ℹ️ **Optional columns** (`age`, `gender`) default to 18 / Female when absent or blank. "
        "Any extra columns (e.g. `student_id`, `name`) are preserved in the output."
    )

    # ── Sample template download ─────────────────────────────────────────
    st.markdown("#### ⬇️ Download Sample Template")
    sample_data = pd.DataFrame([
        {"student_id": 1, "series_1": 32.5, "series_2": 29.0, "study_hours_per_day": 4.0,
         "attendance_percentage": 88.0, "social_media_hours": 1.5, "age": 19, "gender": "Female"},
        {"student_id": 2, "series_1": 15.0, "series_2": 12.5, "study_hours_per_day": 1.5,
         "attendance_percentage": 62.0, "social_media_hours": 5.0, "age": 17, "gender": "Male"},
        {"student_id": 3, "series_1": 25.0, "series_2": 27.5, "study_hours_per_day": 3.0,
         "attendance_percentage": 76.0, "social_media_hours": 2.5, "age": 20, "gender": "Other"},
        {"student_id": 4, "series_1": 38.0, "series_2": 36.0, "study_hours_per_day": 6.0,
         "attendance_percentage": 95.0, "social_media_hours": 0.5, "age": 21, "gender": "Female"},
        {"student_id": 5, "series_1": 20.0, "series_2": 18.0, "study_hours_per_day": 2.0,
         "attendance_percentage": 80.0, "social_media_hours": 3.0, "age": 18, "gender": "Male"},
    ])
    template_csv = sample_data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Sample Template (CSV)",
        data=template_csv,
        file_name="bulk_prediction_template.csv",
        mime="text/csv",
        key="dl_template",
    )

    st.markdown("---")
    st.markdown("#### 📤 Upload Student CSV")

    # ── File uploader ────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload a CSV file with student data. Maximum recommended size: 50,000 rows.",
        key="bulk_csv_uploader",
    )

    if uploaded_file is not None:
        # ── Parse & preview ──────────────────────────────────────────────
        try:
            raw_df = pd.read_csv(uploaded_file)
        except Exception as parse_err:
            st.error(f"❌ Could not parse CSV file: {parse_err}")
            st.stop()

        n_rows = len(raw_df)
        n_cols = len(raw_df.columns)
        st.markdown(f"**Preview** — {n_rows:,} rows × {n_cols} columns uploaded")
        st.dataframe(raw_df.head(10), use_container_width=True)
        if n_rows > 10:
            st.caption(f"Showing first 10 of {n_rows:,} rows.")

        # ── Column presence check (required only) ────────────────────────
        missing_cols = [c for c in BULK_REQUIRED_COLUMNS if c not in raw_df.columns]
        if missing_cols:
            st.error(
                f"❌ **Missing required columns:** `{', '.join(missing_cols)}`\n\n"
                f"Your CSV has columns: `{', '.join(raw_df.columns.tolist())}`\n\n"
                "Please fix the CSV and re-upload."
            )
            st.stop()

        st.success(f"✅ All required columns present. {n_rows:,} rows ready for validation.")

        # ── Predict All button ───────────────────────────────────────────
        if st.button("🚀 Predict All", type="primary", key="bulk_predict_btn", use_container_width=True):
            progress_bar = st.progress(0, text="Validating and predicting…")
            status_text = st.empty()

            def _progress(done: int, total: int):
                pct = int(done / total * 100) if total else 100
                progress_bar.progress(pct, text=f"Processing row {done:,} / {total:,}…")

            try:
                results_df, errors_df = predict_bulk(
                    df=raw_df,
                    model=model,
                    oof_meta=oof_res,
                    settings=settings,
                    progress_callback=_progress,
                )
            except Exception as bulk_err:
                progress_bar.empty()
                st.error(f"❌ Bulk prediction failed unexpectedly: {bulk_err}")
                st.stop()

            progress_bar.progress(100, text="Done!")
            st.session_state["bulk_results_df"] = results_df
            st.session_state["bulk_errors_df"] = errors_df
            st.session_state["bulk_n_total"] = n_rows

    # ── Results ──────────────────────────────────────────────────────────
    if "bulk_results_df" in st.session_state:
        results_df: pd.DataFrame = st.session_state["bulk_results_df"]
        errors_df: pd.DataFrame = st.session_state["bulk_errors_df"]
        n_total = st.session_state.get("bulk_n_total", len(results_df) + len(errors_df))
        n_valid = len(results_df)
        n_errors = len(errors_df)

        st.markdown("---")
        st.markdown("#### 📊 Prediction Results")

        # Summary metrics
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid #16A34A;">
                <div class="metric-title">Rows Processed Successfully</div>
                <div class="metric-value" style="color: #16A34A;">{n_valid:,}</div>
                <div class="metric-subtitle">of {n_total:,} uploaded rows</div>
            </div>
            """, unsafe_allow_html=True)
        with mc2:
            err_color = "#DC2626" if n_errors > 0 else "#64748B"
            st.markdown(f"""
            <div class="metric-card" style="border-top: 4px solid {err_color};">
                <div class="metric-title">Rows Rejected (Validation Errors)</div>
                <div class="metric-value" style="color: {err_color};">{n_errors:,}</div>
                <div class="metric-subtitle">See error report below</div>
            </div>
            """, unsafe_allow_html=True)
        with mc3:
            if n_valid > 0:
                avg_score = results_df["predicted_final_marks"].mean()
                at_risk_count = (results_df["predicted_final_marks"] < settings.target_score).sum()
                st.markdown(f"""
                <div class="metric-card" style="border-top: 4px solid #2563EB;">
                    <div class="metric-title">Cohort Avg Predicted Score</div>
                    <div class="metric-value" style="color: #2563EB;">{avg_score:.1f}</div>
                    <div class="metric-subtitle">{at_risk_count:,} students below target ({settings.target_score:.0f})</div>
                </div>
                """, unsafe_allow_html=True)

        if n_valid > 0:
            st.markdown("##### ✅ Valid Predictions")

            # Colour-code predicted_final_marks column
            def _style_score(val):
                if val < settings.target_score:
                    return "background-color: #FEE2E2; color: #991B1B; font-weight: 600;"
                elif val < settings.target_score + settings.risk_margin:
                    return "background-color: #FEF3C7; color: #92400E; font-weight: 600;"
                else:
                    return "background-color: #DCFCE7; color: #166534; font-weight: 600;"

            styled = results_df.style.map(_style_score, subset=["predicted_final_marks"])
            st.dataframe(styled, use_container_width=True)

            # Download button for results
            results_csv = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Predictions CSV",
                data=results_csv,
                file_name="bulk_predictions_results.csv",
                mime="text/csv",
                key="dl_results",
            )

        if n_errors > 0:
            st.markdown("##### ❌ Rejected Rows — Validation Errors")
            st.warning(
                f"{n_errors:,} row(s) were rejected due to validation errors. "
                "Fix the issues listed below and re-upload."
            )
            st.dataframe(
                errors_df[["validation_errors"] + [c for c in errors_df.columns if c != "validation_errors"]],
                use_container_width=True,
            )
            errors_csv = errors_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Rejected Rows CSV",
                data=errors_csv,
                file_name="bulk_predictions_rejected.csv",
                mime="text/csv",
                key="dl_errors",
            )

        if n_valid == 0 and n_errors == 0:
            st.info("No rows were in the uploaded file.")

    elif uploaded_file is None:
        st.info(
            "Upload a CSV file above, then click **🚀 Predict All** to generate "
            "bulk final marks predictions for your entire student cohort."
        )
