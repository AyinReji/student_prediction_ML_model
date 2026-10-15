# BUILD.md — Student Performance Prediction & Personalized Support System

A teacher-facing system that predicts each student's final exam score, flags students at risk of falling below a configurable target, explains *why*, suggests follow-up actions, ranks students by a limited follow-up capacity, and tracks follow-ups over time.

---

## 0. What the dataset actually says (read this first)

File: `student_performance_10000-1.csv` — 10,000 rows, 8 columns, no missing values, no duplicate rows.

| Column | Type | Range | Correlation with `final_exam_score` |
|---|---|---|---|
| age | int | 16–24 | 0.01 |
| gender | category (Male 4893, Female 4754, Other 353) | — | no meaningful mean difference |
| study_hours_per_day | float | 0.5–6.0 | 0.01 |
| attendance_percentage | float | 40–100 | −0.005 |
| social_media_hours | float | 0–6 | 0.002 |
| series_1 | float | 0–40 | **0.83** |
| series_2 | float | 0–40 | **0.82** |
| final_exam_score | float (target) | 12.7–96.0 | — |

Quick baseline (default settings, 80/20 split, `random_state=42`): Linear Regression MAE ≈ 5.4 / R² ≈ 0.74, Gradient Boosting MAE ≈ 5.4 / R² ≈ 0.73, Random Forest MAE ≈ 5.6 / R² ≈ 0.72. A model using only `series_1` + `series_2` scores the same as all features.

**What this means for the project**

1. The two assessment series carry essentially all the predictive signal. Study hours, attendance, social media and age show no relationship with the final score, and they are also uncorrelated with each other and with the series marks. Real student data rarely looks like this; the pattern is consistent with simulated data. The brief asks you to check authenticity, so report this plainly in the app and in your write-up.
2. The three algorithms will perform very similarly. Pick on cross-validation evidence, and prefer the simpler model if the difference is within noise.
3. Lifestyle-based recommendations (attendance, study hours, social media) cannot be justified by this data as predictors of the score. In the UI, label them as **conversation starters to discuss with the student**, never as causes or proven drivers.
4. The data has no topic-level results, no student IDs and no dates. Do not claim subject or topic diagnosis. Use the row index as a pseudo student ID.

---

## 1. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Standard ML ecosystem |
| Data | `pandas`, `numpy` | Loading, cleaning, feature engineering |
| ML | `scikit-learn` | Linear Regression, Random Forest, Gradient Boosting, pipelines, CV, metrics |
| Explainability | `shap` (+ model coefficients for Linear Regression) | Per-student contribution of each feature |
| Persistence of model | `joblib` | Save and load the trained pipeline |
| Follow-up storage | `SQLite` via `SQLAlchemy` | Zero-setup database for follow-up actions and later assessments |
| Config | `PyYAML` + `pydantic` | Teacher-configurable thresholds and rule settings, validated |
| UI | **Streamlit** | Fastest route to a clean multi-page dashboard in pure Python |
| Charts | `plotly` | Interactive, clean charts (gauges, bars, scatter) |
| Static plots (report) | `matplotlib`, `seaborn` | Residual and EDA figures for the write-up |
| Testing | `pytest` | Leakage, split and rule tests |
| Quality | `ruff` | Lint and format |
| Notebook (optional) | `jupyterlab` | EDA before moving code into modules |

### Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -U pip
pip install "pandas>=2.2" "numpy>=1.26" "scikit-learn>=1.4" shap joblib \
            "streamlit>=1.35" plotly matplotlib seaborn \
            sqlalchemy pyyaml pydantic pytest ruff jupyterlab
pip freeze > requirements.txt
```

Alternative UI path (only if you need a multi-user deployment later): FastAPI backend + React frontend. Not needed for this project; Streamlit is enough.

---

## 2. Project structure

```
student-support-system/
├── build.md
├── README.md
├── requirements.txt
├── config/
│   └── settings.yaml            # target score, risk margin, capacity, rule thresholds, weights
├── data/
│   ├── raw/student_performance_10000-1.csv
│   └── processed/               # train.csv, test.csv (written once, after the split)
├── src/
│   ├── config.py                # loads and validates settings.yaml
│   ├── data.py                  # load, validate, audit, split
│   ├── features.py              # feature engineering (series_avg, series_delta)
│   ├── train.py                 # trains 3 models, CV, picks best, evaluates on test
│   ├── evaluate.py              # regression, at-risk, fairness, priority-queue metrics
│   ├── explain.py               # SHAP / coefficient contributions per student
│   ├── recommend.py             # rule engine
│   ├── priority.py              # priority score and queue
│   └── db.py                    # SQLite models and helpers
├── app/
│   ├── Home.py                  # Streamlit entry
│   └── pages/
│       ├── 1_Priority_Queue.py
│       ├── 2_Student_Detail.py
│       ├── 3_Progress_Tracking.py
│       ├── 4_Model_Evaluation.py
│       ├── 5_Data_Audit.py
│       └── 6_Settings.py
├── models/                      # best_model.joblib, metrics.json, oof_residuals.json
├── reports/figures/
├── tests/
└── .streamlit/config.toml       # clean theme
```

---

## 3. Build steps

Complete each phase and its acceptance check before moving on.

### Phase 1 — Setup
1. Create the folder structure, virtual environment and `requirements.txt`.
2. Put the CSV in `data/raw/`.
3. Create `config/settings.yaml` (see section 5).

**Done when:** `python -c "import sklearn, shap, streamlit"` runs without error.

### Phase 2 — Data audit (before any modelling)
Using the full raw file, only to describe it, never to fit anything:
1. Missing values, duplicates, dtypes.
2. Range checks: `series_1`, `series_2` within 0–40; `attendance_percentage` within 0–100; `final_exam_score` within 0–100; study and social media hours within 0–24 and plausible.
3. Outliers: IQR flags per numeric column. Report them; do not delete valid values.
4. Distributions and correlation matrix (including correlations between features).
5. Leakage check: confirm `final_exam_score` is never in the feature list. Check that no feature is a near-copy of the target (|corr| > 0.95).
6. Representativeness and authenticity check: gender balance, age spread, whether lifestyle features are uncorrelated with everything, round-number patterns. Write the findings to `reports/data_audit.md` and show them on the **Data Audit** page.

**Done when:** `reports/data_audit.md` exists and states the limits from section 0.

### Phase 3 — Split first, then preprocess
1. Split **80% train / 20% test**, `random_state=42`. Stratify on target quantile bins (e.g. 10 bins from `pd.qcut`) so both sets have a similar score distribution.
2. Save `train.csv` and `test.csv`. The test set is not touched again until Phase 5.
3. Build everything else inside a scikit-learn `Pipeline` / `ColumnTransformer` so that scaling and encoding are fitted on training data only.

**Features used as model input (default):** `study_hours_per_day`, `attendance_percentage`, `social_media_hours`, `series_1`, `series_2`, plus engineered `series_avg` and `series_delta` (`series_2 − series_1`).
**Not used as input:** `final_exam_score` (target), `age` and `gender` (kept for profile and fairness audit). Run one ablation that adds them, to show whether they change anything.

**Done when:** a test asserts the test set never appears in any `fit` call and the target is absent from `X`.

### Phase 4 — Train three algorithms
Models (all in the same pipeline shape):

| Model | Notes |
|---|---|
| `LinearRegression` | Scale numeric inputs with `StandardScaler`. Fully transparent coefficients. |
| `RandomForestRegressor` | Tune `n_estimators`, `max_depth`, `min_samples_leaf`. |
| `GradientBoostingRegressor` | Tune `n_estimators`, `learning_rate`, `max_depth`, `subsample`. |

Process:
1. 5-fold cross-validation **on the 80% training set only**, with `GridSearchCV` or `RandomizedSearchCV` for the two tree models (small grids).
2. Primary selection metric: CV RMSE. Also record MAE and R².
3. **Selection rule:** choose the lowest CV RMSE. If two models are within one standard error of each other, choose the simpler / more explainable one.
4. Refit the chosen model on the full 80% training set.
5. Record out-of-fold residuals on training data; these power the prediction interval and risk probability.

**Done when:** `models/best_model.joblib`, `models/metrics.json` and a model comparison table exist.

### Phase 5 — Evaluate on the 20% test set (once)
1. **Regression:** MAE, RMSE, R² for all three models, shown side by side. The chosen model is the one picked in Phase 4 using CV, not the one that looks best on test.
2. **Prediction interval:** 90% interval = prediction ± the 90th percentile of absolute out-of-fold residuals. Check empirical coverage on the test set (should be near 90%).
3. **At-risk identification:** ground truth = `final_exam_score < target`. Flag = `predicted_score < target + risk_margin`. Report precision, recall, F1, confusion matrix at the default target, and at 2 other targets (e.g. 40 and 50). Note the class balance: about 11% of students score under 40 and about 31% under 50, so use recall and precision rather than accuracy.
4. **Fairness audit:** by gender and by age band, report MAE, mean residual (bias), and at-risk recall and precision. Flag any group with a materially worse error or recall, and show group sizes (the "Other" group is small, about 3.5%, so say when results are too noisy to conclude).
5. **Baselines for the at-risk and priority tasks:**
   - rank by predicted score only
   - rank by average of `series_1` and `series_2` only
   - rank by the full priority score (Phase 7)

**Done when:** `models/metrics.json` holds all of the above and the Model Evaluation page renders it.

### Phase 6 — Explainability and recommendations
**Explainability**
- Linear Regression: contribution = coefficient × (student value − training mean), in score points.
- Tree models: `shap.TreeExplainer` per-student SHAP values.
- Output per student: predicted score, interval, top 3–5 contributing factors with direction and size in points, expressed in plain language.

**Rule-based recommendation engine** (`src/recommend.py`, thresholds in `settings.yaml`):

| Rule | Trigger (defaults, editable) | Suggested teacher action | Label shown |
|---|---|---|---|
| Low assessment marks | `series_avg` below `low_series_avg` (e.g. 18/40) | Review recent assessment work with the student; offer extra practice | Supported by data |
| Declining marks | `series_delta` ≤ `−decline_points` (e.g. −4) | Check in on what changed between the two series | Supported by data |
| Attendance concern | attendance < `min_attendance` (e.g. 75%) | Have a conversation about barriers to attending | Indicator only |
| Low study time | study hours < `min_study_hours` (e.g. 2) | Discuss study routine and support | Indicator only |
| High social media use | social media hours > `max_social_hours` (e.g. 4) | Discuss time management | Indicator only |

Wording rules:
- Never write "because the student studies little". Write "study hours are below the threshold; worth discussing".
- Rules marked "Indicator only" show an info note: "In this dataset this factor shows no measurable relationship with final scores."
- Never mention a subject or topic weakness.
- Every recommendation shows the rule that fired and the value that triggered it, so a teacher can see why.

### Phase 7 — Teacher priority queue
Compute for each student:

- `p_risk` = probability that the final score is below target, using a normal approximation `Φ((target − predicted) / σ)` where σ is the out-of-fold residual standard deviation.
- `decline_flag` = 1 if the declining-marks rule fired.
- `flag_share` = fired rules / total rules.

```
priority = w_risk * p_risk + w_decline * decline_flag + w_flags * flag_share
```

Default weights in `settings.yaml`: `w_risk=0.7, w_decline=0.2, w_flags=0.1`. The teacher sets **capacity N** (how many follow-ups they can do), and the queue shows the top N with the reason for each.

**Evaluating whether the combined system beats plain ranking by predicted score** (this is the project's main contribution, so do it honestly):
- Metrics at capacity N (try N = 10, 25, 50, 100): how many truly at-risk students are captured (recall@N), precision@N, and overlap with the baseline queue.
- Choose weights using out-of-fold predictions on the **training** data, then report the final comparison on the test set once.
- Because lifestyle flags have no demonstrated link to the score in this data, the full priority score may not beat the predicted-score baseline. If so, say so. A clear negative or neutral result is a valid finding. What the combined system adds is explanation, action suggestions and workload control, which you can describe separately from the accuracy comparison.
- State clearly that no data exists on whether follow-ups improve outcomes, so "useful" here means catching at-risk students within capacity and giving usable explanations, not proven improvement.

### Phase 8 — Follow-up database
SQLite tables:

```
students(student_id PK, age, gender, ...)                -- loaded from the dataset, row index as id
followups(id PK, student_id FK, date, teacher, action_type, notes, status)
assessments(id PK, student_id FK, assessment_name, date, score, max_score)
```

Features:
- Log a follow-up action against a student (action type from the rule categories or "other", plus notes).
- Record later assessment results.
- Progress view: for each student, compare assessment scores before and after the first follow-up, shown as a simple line chart and a change value.
- Caveat in the UI: before/after changes can be due to chance (including regression to the mean) and do not show that the follow-up caused the change.

### Phase 9 — Streamlit UI
Keep it clean: one accent colour, white/light-grey surfaces, generous spacing, large metric cards, no clutter.

`.streamlit/config.toml`:
```toml
[theme]
base = "light"
primaryColor = "#2F5BEA"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F5F7FB"
textColor = "#1B2430"
font = "sans serif"
```

| Page | Contents |
|---|---|
| **Home** | Metric cards: students, at-risk count, average predicted score, follow-ups logged. Sidebar controls: target score slider, capacity N. Short note on the model in use. |
| **Priority Queue** | Table of the top N students: rank, ID, predicted score, risk badge (High / Medium / Low), top reason, suggested action. Filters by risk level. CSV export. |
| **Student Detail** | Predicted score with interval, risk badge, bar chart of contributing factors (points up/down), fired rules with suggested actions and the info notes, assessment trend (series 1 to series 2), form to log a follow-up. |
| **Progress Tracking** | Follow-up log, add later assessment results, before/after comparison per student and for the group, with the caveat shown. |
| **Model Evaluation** | 3-model comparison table (CV and test), predicted vs actual scatter, residual plot, at-risk precision/recall/F1 and confusion matrix, fairness table, priority-queue vs baseline chart (recall@N). |
| **Data Audit** | Findings from Phase 2: ranges, outliers, correlation heatmap, and the authenticity/representativeness note. |
| **Settings** | Edit target, risk margin, capacity, rule thresholds, priority weights. Saves to `settings.yaml`. |

Risk badge logic (editable): High if `p_risk ≥ 0.6`, Medium if `0.3 ≤ p_risk < 0.6`, Low otherwise.

Performance: wrap model loading and data loading in `st.cache_resource` / `st.cache_data`; recompute predictions only when inputs or settings change.

### Phase 10 — Testing and delivery
Tests (`pytest`):
- Target column is not in the feature matrix.
- Train/test split is 80/20, disjoint, and reproducible with the seed.
- Preprocessing is fitted on train only.
- Each rule fires on a hand-made example and not on a counter-example.
- Priority queue returns exactly N rows, sorted descending.
- No recommendation text contains topic/subject claims or causal wording (simple string check on templates).

Run order:
```bash
python -m src.data        # audit + split
python -m src.train       # train 3 models, CV, pick best, evaluate on test
streamlit run app/Home.py
```

`README.md` should cover setup, how to run, the dataset limits from section 0, and the evaluation results.

---

## 4. Research and ethics guardrails (enforce in code and UI)

- `final_exam_score` is only ever the target.
- Split before preprocessing and before model selection.
- Low study hours and high social media use are **indicators for discussion**, not causes.
- No topic-level or subject-level claims: the data has none.
- Do not use gender or age as model inputs by default; use them only for the fairness audit.
- Predictions are decision support for a teacher, not a label for the student. Show the interval, not only a single number.
- State dataset limits (likely simulated, weak lifestyle signal, no IDs or dates) on the Data Audit page.
- If real student data replaces this file later: remove names and direct identifiers, restrict access, and get the school's approval.

---

## 5. `config/settings.yaml` (starter)

```yaml
target_score: 40          # teacher-configurable pass / support target
risk_margin: 3            # flag if predicted < target + margin
capacity: 25              # follow-ups a teacher can handle
interval_level: 0.90
rules:
  low_series_avg: 18      # out of 40
  decline_points: 4
  min_attendance: 75
  min_study_hours: 2
  max_social_hours: 4
priority_weights:
  risk: 0.7
  decline: 0.2
  flags: 0.1
risk_badges:
  high: 0.6
  medium: 0.3
split:
  test_size: 0.2
  seed: 42
cv_folds: 5
```

---

## 6. Deliverables checklist

- [ ] `reports/data_audit.md`
- [ ] Trained and saved best model plus `metrics.json`
- [ ] 3-model comparison (CV and held-out test)
- [ ] At-risk precision / recall / F1 at several targets
- [ ] Fairness table by gender and age band
- [ ] Priority queue vs baselines at several capacities
- [ ] Streamlit app with all pages above
- [ ] SQLite follow-up and progress tracking
- [ ] Tests passing
- [ ] README with limits and honest results
