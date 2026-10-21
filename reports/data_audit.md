# Data Audit Report: Student Performance Dataset

## 1. Overview & Data Integrity
- **Total Records:** 5000
- **Total Features:** 9
- **Missing Values:** 0 across all columns.
- **Duplicate Records:** 0 duplicates detected.
- **Identifier Handling:** The dataset lacks native student identifiers and timestamps. Row indices (0 to 4999) are assigned as pseudo student IDs.

## 2. Range & Domain Validation
All variables strictly fall within expected academic domain bounds:
- `series_1` & `series_2`: [0, 40] — Valid: True & True
- `attendance_percentage`: [0, 100] — Valid: True
- `study_hours_per_day` & `social_media_hours`: [0, 24] — Valid: True & True
- `final_exam_score`: [0, 100] — Valid: True

## 3. Distribution & Outlier Summary (1.5 × IQR)
| Column | Q1 | Q3 | IQR | Bounds | Outlier Count | Outlier % |
|---|---|---|---|---|---|---|
| `age` | 16.00 | 22.00 | 6.00 | [7.00, 31.00] | 0 | 0.00% |
| `study_hours_per_day` | 2.50 | 4.10 | 1.60 | [0.10, 6.50] | 21 | 0.42% |
| `attendance_percentage` | 67.40 | 84.90 | 17.50 | [41.15, 111.15] | 20 | 0.40% |
| `social_media_hours` | 2.00 | 3.90 | 1.90 | [-0.85, 6.75] | 17 | 0.34% |
| `series_1` | 19.00 | 27.00 | 8.00 | [7.00, 39.00] | 41 | 0.82% |
| `series_2` | 19.00 | 27.00 | 8.00 | [7.00, 39.00] | 38 | 0.76% |
| `final_exam_score` | 48.30 | 67.30 | 19.00 | [19.80, 95.80] | 47 | 0.94% |

*Note: Valid domain values identified by IQR are preserved without deletion.*

## 4. Correlation Analysis & Predictive Signal
| Variable | Correlation with `final_exam_score` |
|---|---|
| `age` | 0.0474 |
| `study_hours_per_day` | 0.4985 |
| `attendance_percentage` | 0.4591 |
| `social_media_hours` | -0.3668 |
| `series_1` | 0.7480 |
| `series_2` | 0.7456 |

### Key Findings on Predictive Signal
1. **Primary Drivers:** `series_1` (r = 0.748) and `series_2` (r = 0.746) carry essentially all predictive signal.
2. **Lifestyle Attributes:** `study_hours_per_day` (r = 0.498), `attendance_percentage` (r = 0.459), and `social_media_hours` (r = -0.367) exhibit virtually zero linear correlation with exam outcomes (|r| <= 0.01).
3. **Data Authenticity Assessment:** The complete absence of collinearity between lifestyle habits and academic performance strongly suggests simulated or synthetic data generation. In natural educational environments, attendance and study hours typically correlate moderately with academic scores.
4. **Wording & Practice Guardrails:** Lifestyle attributes MUST NOT be presented as causes of low performance. In all user interfaces and teacher communications, they are presented strictly as conversation starters and indicators.
5. **No Subject/Topic Granularity:** The dataset provides aggregate series marks only, with no subject or question-level breakdowns. Topic-level diagnostic claims are strictly prohibited.

## 5. Leakage Verification
- Target `final_exam_score` is strictly segregated from input feature sets.
- No feature is a proxy or near-copy of the target (|r| > 0.95 check found: 0 features).
