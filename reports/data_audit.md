# Data Audit Report: Student Performance Dataset

## 1. Overview & Data Integrity
- **Total Records:** 10000
- **Total Features:** 9
- **Missing Values:** 0 across all columns.
- **Duplicate Records:** 0 duplicates detected.
- **Identifier Handling:** The dataset lacks native student identifiers and timestamps. Row indices (0 to 9999) are assigned as pseudo student IDs.

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
| `study_hours_per_day` | 1.90 | 4.60 | 2.70 | [-2.15, 8.65] | 0 | 0.00% |
| `attendance_percentage` | 54.90 | 84.60 | 29.70 | [10.35, 129.15] | 0 | 0.00% |
| `social_media_hours` | 1.50 | 4.50 | 3.00 | [-3.00, 9.00] | 0 | 0.00% |
| `series_1` | 18.00 | 27.00 | 9.00 | [4.50, 40.50] | 11 | 0.11% |
| `series_2` | 18.50 | 27.00 | 8.50 | [5.75, 39.75] | 74 | 0.74% |
| `final_exam_score` | 47.30 | 66.20 | 18.90 | [18.95, 94.55] | 9 | 0.09% |

*Note: Valid domain values identified by IQR are preserved without deletion.*

## 4. Correlation Analysis & Predictive Signal
| Variable | Correlation with `final_exam_score` |
|---|---|
| `age` | 0.0085 |
| `study_hours_per_day` | 0.0115 |
| `attendance_percentage` | -0.0050 |
| `social_media_hours` | 0.0020 |
| `series_1` | 0.8278 |
| `series_2` | 0.8220 |

### Key Findings on Predictive Signal
1. **Primary Drivers:** `series_1` (r = 0.828) and `series_2` (r = 0.822) carry essentially all predictive signal.
2. **Lifestyle Attributes:** `study_hours_per_day` (r = 0.011), `attendance_percentage` (r = -0.005), and `social_media_hours` (r = 0.002) exhibit virtually zero linear correlation with exam outcomes (|r| <= 0.01).
3. **Data Authenticity Assessment:** The complete absence of collinearity between lifestyle habits and academic performance strongly suggests simulated or synthetic data generation. In natural educational environments, attendance and study hours typically correlate moderately with academic scores.
4. **Wording & Practice Guardrails:** Lifestyle attributes MUST NOT be presented as causes of low performance. In all user interfaces and teacher communications, they are presented strictly as conversation starters and indicators.
5. **No Subject/Topic Granularity:** The dataset provides aggregate series marks only, with no subject or question-level breakdowns. Topic-level diagnostic claims are strictly prohibited.

## 5. Leakage Verification
- Target `final_exam_score` is strictly segregated from input feature sets.
- No feature is a proxy or near-copy of the target (|r| > 0.95 check found: 0 features).
