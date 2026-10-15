# Student Performance Prediction & Personalized Support System

A production-grade, teacher-facing machine learning system that predicts students' final exam scores, flags individuals at academic risk against customizable passing thresholds, explains feature contributions in concrete score points, provides rule-based pedagogical intervention recommendations, ranks students in a capacity-bounded priority queue, and tracks student progress longitudinally in an SQLite database.

---

## 1. Quick Start & Setup

### Environment Setup
```bash
# Clone and enter directory
cd Student_Perfomance_Prediction_v2

# Create virtual environment (optional)
python -m venv .venv
# Windows: .venv\Scripts\activate | macOS/Linux: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Execution Pipeline
Execute the modules in strict sequence:

```bash
# 1. Audit raw data, generate reports/data_audit.md, and perform 80/20 stratified split
python -m src.data

# 2. Train 3 algorithms, run 5-fold CV, apply 1-SE model selection, fit OOF residuals, and evaluate test set
python -m src.train

# 3. Initialize SQLite follow-up database and verify tests
pytest -v

# 4. Launch the Streamlit dashboard
streamlit run app/Home.py
```

Access the dashboard at `http://localhost:8501`.

---

## 2. Dataset Reality & Technical Limits

The raw dataset (`student_performance_10000-1.csv`) contains 10,000 student records and 8 variables:
- **Zero Missing Values & Zero Duplicates**: All variables adhere to expected bounds.
- **Predominant Predictive Signal**: `series_1` ($r = 0.828$) and `series_2` ($r = 0.822$) carry essentially 100% of the predictive signal for `final_exam_score`.
- **Near-Zero Lifestyle Correlation**: `study_hours_per_day` ($r = 0.011$), `attendance_percentage` ($r = -0.005$), `social_media_hours` ($r = 0.002$), and `age` ($r = 0.009$) have virtually zero correlation with the final exam score or with each other.
- **Synthetic/Simulated Characteristics**: Real educational distributions exhibit moderate positive correlation between attendance/study time and academic scores. The complete statistical independence of lifestyle attributes strongly indicates that the data was generated synthetically.
- **No Temporal or Topic Granularity**: There are no timestamps, student identifiers, or subject/topic question breakdowns. Row indices are used as pseudo IDs, and subject/topic diagnostic claims are strictly prohibited in the system.
- **Wording Guardrail**: Lifestyle factors are categorized as **"Indicator only"** conversation starters, never as causal drivers.

---

## 3. Real Empirical Evaluation Results (Held-Out 20% Test Set)

The 80% train / 20% test split was executed using `random_state=42`, stratified across 10 quantile bins of `final_exam_score`. The test set was touched exactly once after model selection.

### Model Selection (5-Fold CV on 80% Training Set)
| Algorithm | CV RMSE (Mean ± SE) | CV MAE | CV R² | Selection Decision |
|---|---|---|---|---|
| **Linear Regression** | **6.765 ± 0.087** | **5.391** | **0.747** | **Selected (1-SE Rule Winner)** |
| **Random Forest** | 6.819 ± 0.090 | 5.438 | 0.743 | Within 1 SE, but higher complexity |
| **Gradient Boosting** | 6.799 ± 0.093 | 5.422 | 0.744 | Within 1 SE, but higher complexity |

*Decision Rationale:* Under the 1-Standard-Error rule, Linear Regression was chosen because its CV RMSE ($6.765$) is well within 1 SE of all models, offering full coefficient transparency, instant training, and direct score-point explainability.

### Test Set Regression Performance (2,000 Samples)
| Model | Test MAE | Test RMSE | Test R² |
|---|---|---|---|
| **Linear Regression** (Selected) | **5.188** | **6.534** | **0.763** |
| Random Forest | 5.211 | 6.567 | 0.761 |
| Gradient Boosting | 5.200 | 6.549 | 0.762 |
| **Ablation (+ Age & Gender)** | **5.189** | **6.535** | **0.763** |

*Ablation Finding:* Adding protected demographic attributes (`age` and `gender`) yielded virtually zero change in prediction error ($\Delta \text{MAE} = +0.001$), confirming that demographic attributes add no predictive signal beyond assessment marks.

### 90% Prediction Interval Calibration
- **Half-Width**: $\pm 11.123$ points (computed as the 90th percentile of absolute out-of-fold residuals).
- **Nominal Coverage**: 90.0%
- **Empirical Test Set Coverage**: **91.15%** (2,000 test students).

### At-Risk Classification Evaluation
Flags students predicted to score below `target + risk_margin` (default target = 40, margin = 3 pts):
| Target Score | Margin | Prevalence | Recall | Precision | F1 Score | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|---|
| **40.0** | +3.0 pts | 11.1% (222/2000) | **0.631** | **0.586** | **0.607** | 140 | 99 | 82 | 1679 |
| **50.0** | +3.0 pts | 31.3% (626/2000) | **0.867** | **0.719** | **0.786** | 543 | 212 | 83 | 1162 |

### Demographic Fairness Audit
| Demographic Subgroup | Group Count (% Share) | MAE | Mean Residual (Actual − Pred) | At-Risk Recall | At-Risk Precision |
|---|---|---|---|---|---|
| **Female** | 909 (45.5%) | 5.133 | +0.156 | 0.629 | 0.555 |
| **Male** | 1022 (51.1%) | 5.213 | -0.234 | 0.633 | 0.608 |
| **Other** | 69 (3.5%) | 5.547 | -0.351 | 0.615 | 0.615 |
| **Age $\le$ 16** | 550 (27.5%) | 5.422 | -0.374 | 0.652 | 0.573 |
| **Age 17–19** | 557 (27.9%) | 5.376 | -0.020 | 0.614 | 0.662 |
| **Age 20–22** | 549 (27.5%) | 4.932 | +0.190 | 0.638 | 0.597 |
| **Age 23+** | 344 (17.2%) | 4.918 | -0.027 | 0.607 | 0.460 |

*Fairness Conclusion:* Mean residuals remain near zero across all groups ($|\text{bias}| < 0.4$ pts), indicating no systemic under- or over-prediction by gender or age. The "Other" group represents 3.5% of the cohort, exhibiting slightly higher sample variance.

### Priority Queue vs. Ranking Baselines (Honest Evaluation)
Evaluated across follow-up capacities $N$ on the test set:
| Capacity $N$ | System Hits (Recall@N) | System Precision@N | Pred Baseline Recall@N | Pred Baseline Precision@N | Overlap with Baseline |
|---|---|---|---|---|---|
| **$N = 10$** | 10 / 222 (0.045) | **1.000** | 10 / 222 (0.045) | **1.000** | 10.0% |
| **$N = 25$** | 22 / 222 (0.099) | **0.880** | 23 / 222 (0.104) | **0.920** | 32.0% |
| **$N = 50$** | 41 / 222 (0.185) | **0.820** | 45 / 222 (0.203) | **0.900** | 74.0% |
| **$N = 100$** | 74 / 222 (0.333) | **0.740** | 78 / 222 (0.351) | **0.780** | 83.0% |

*Honest Research Finding:* Because lifestyle features have no measurable relationship with final exam scores in this dataset, incorporating rule flags into priority scoring introduces minor variance relative to pure score sorting. Consequently, the combined priority queue achieves comparable or marginally lower pure Recall@N than sorting exclusively by predicted score (e.g. 0.88 vs 0.92 precision at $N=25$). The primary value of the priority queue is not purely ranking efficiency, but providing workload bounding ($N$), concrete explanations, and actionable teacher recommendations.

---

## 4. Software Architecture

```
student-support-system/
├── app/
│   ├── Home.py                  # Streamlit entry point: metrics & cohort distribution
│   ├── ui_utils.py              # Shared styling, caching, and sidebar controls
│   └── pages/
│       ├── 1_Priority_Queue.py   # Top-N table, risk badges, filters, CSV export
│       ├── 2_Student_Detail.py   # Point contributions, intervals, rules, follow-up form
│       ├── 3_Progress_Tracking.py# Intervention log, new assessments, trajectory chart
│       ├── 4_Model_Evaluation.py # 3-model CV/test tables, ablation, plots, fairness
│       ├── 5_Data_Audit.py       # Data integrity, correlation heatmap, IQR outliers
│       └── 6_Settings.py         # Config editor for thresholds, rules, and weights
├── config/
│   └── settings.yaml            # Operational parameters & priority weights
├── data/
│   ├── raw/                     # Original student performance CSV
│   ├── processed/               # Stratified train.csv and test.csv
│   └── student_support.db       # SQLite database (students, followups, assessments)
├── models/
│   ├── best_model.joblib        # Fitted pipeline
│   ├── metrics.json             # Comprehensive test set metrics
│   └── oof_residuals.json       # Out-of-fold residual statistics
├── reports/
│   ├── data_audit.md            # Detailed data integrity & authenticity audit
│   └── figures/                 # Correlation heatmap
├── src/
│   ├── config.py                # Pydantic schema validation & YAML persistence
│   ├── data.py                  # Loading, range audits, outlier detection, split
│   ├── features.py              # Feature definitions & engineering (series_avg, delta)
│   ├── train.py                 # 5-fold CV, tuning, 1-SE selection, test evaluation
│   ├── evaluate.py              # Metrics, prediction interval, fairness, baselines
│   ├── explain.py               # Linear coefficient & Tree SHAP point attribution
│   ├── recommend.py             # Deterministic rules with wording guardrails
│   ├── priority.py              # Priority score formulation & ranking engine
│   └── db.py                    # SQLAlchemy SQLite database models & progress tracking
└── tests/                       # Complete pytest suite (13 tests passing)
```

---

## 5. Testing & Verification

Run the automated test suite:
```bash
python -m pytest -v
```
All 13 tests verify:
- Feature segregation (target `final_exam_score` is strictly excluded from `X`).
- 80/20 train/test split disjointness and reproducibility.
- Preprocessing scaler fitted on training data only.
- Positive and counter-example triggers for all 5 recommendation rules.
- Priority queue returns exactly $N$ rows in strictly descending order.
- Absence of causal claims ("because", "causes") and prohibited topic claims.
- Presence of required disclaimers for indicator-only rules and progress tracking.
