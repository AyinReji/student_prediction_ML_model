from typing import Any

import pandas as pd

from src.config import Settings, load_settings

INDICATOR_DISCLAIMER = "In this dataset this factor shows no measurable relationship with final scores."

RULES_METADATA = {
    "low_assessment_marks": {
        "name": "Low assessment marks",
        "category": "academic",
        "label": "Supported by data",
        "action": "Review recent assessment work with the student; offer extra practice.",
        "info_note": "Assessment series performance correlates strongly with final exam outcomes.",
    },
    "declining_marks": {
        "name": "Declining marks",
        "category": "academic",
        "label": "Supported by data",
        "action": "Check in on what changed between the two series.",
        "info_note": "Trajectory between series 1 and series 2 indicates momentum changes.",
    },
    "attendance_concern": {
        "name": "Attendance Concern (Exam Ineligibility)",
        "category": "lifestyle",
        "label": "Indicator only",
        "action": "Attendance is below 75%; student cannot attend the final examination under institutional policy. Arrange urgent review.",
        "info_note": INDICATOR_DISCLAIMER,
    },
    "low_study_time": {
        "name": "Low study time",
        "category": "lifestyle",
        "label": "Indicator only",
        "action": "Discuss study routine and support; study hours are below the threshold; worth discussing.",
        "info_note": INDICATOR_DISCLAIMER,
    },
    "high_social_media": {
        "name": "High social media use",
        "category": "lifestyle",
        "label": "Indicator only",
        "action": "Discuss time management; social media use is above the threshold; worth discussing.",
        "info_note": INDICATOR_DISCLAIMER,
    },
}

TOTAL_RULES_COUNT = 5


def evaluate_student_rules(student_row: pd.Series | dict[str, Any], settings: Settings | None = None) -> list[dict[str, Any]]:
    """Evaluates the 5 deterministic support rules for a single student record.
    Adheres strictly to wording guardrails (no causality, no topic/subject claims).
    """
    if settings is None:
        settings = load_settings()

    rules_cfg = settings.rules
    fired_rules: list[dict[str, Any]] = []

    # 1. Low assessment marks
    series_avg = float(student_row.get("series_avg", (student_row.get("series_1", 0) + student_row.get("series_2", 0)) / 2.0))
    if series_avg < rules_cfg.low_series_avg:
        meta = RULES_METADATA["low_assessment_marks"]
        fired_rules.append({
            "rule_id": "low_assessment_marks",
            "name": meta["name"],
            "trigger": f"Average assessment series is {series_avg:.1f}/40 (threshold < {rules_cfg.low_series_avg:.1f})",
            "action": meta["action"],
            "label": meta["label"],
            "info_note": meta["info_note"],
            "is_indicator_only": False,
        })

    # 2. Declining marks
    series_delta = float(student_row.get("series_delta", student_row.get("series_2", 0) - student_row.get("series_1", 0)))
    if series_delta <= -rules_cfg.decline_points:
        meta = RULES_METADATA["declining_marks"]
        fired_rules.append({
            "rule_id": "declining_marks",
            "name": meta["name"],
            "trigger": f"Assessment delta is {series_delta:+.1f} points (threshold <= -{rules_cfg.decline_points:.1f})",
            "action": meta["action"],
            "label": meta["label"],
            "info_note": meta["info_note"],
            "is_indicator_only": False,
        })

    # 3. Attendance concern
    attendance = float(student_row.get("attendance_percentage", 100))
    if attendance < rules_cfg.min_attendance:
        meta = RULES_METADATA["attendance_concern"]
        fired_rules.append({
            "rule_id": "attendance_concern",
            "name": meta["name"],
            "trigger": f"Attendance is {attendance:.1f}% (threshold < {rules_cfg.min_attendance:.1f}% mandatory minimum; student cannot attend exam)",
            "action": meta["action"],
            "label": meta["label"],
            "info_note": meta["info_note"],
            "is_indicator_only": True,
        })

    # 4. Low study time
    study_hours = float(student_row.get("study_hours_per_day", 0))
    if study_hours < rules_cfg.min_study_hours:
        meta = RULES_METADATA["low_study_time"]
        fired_rules.append({
            "rule_id": "low_study_time",
            "name": meta["name"],
            "trigger": f"Reported study time is {study_hours:.1f} hrs/day (threshold < {rules_cfg.min_study_hours:.1f} hrs/day)",
            "action": meta["action"],
            "label": meta["label"],
            "info_note": meta["info_note"],
            "is_indicator_only": True,
        })

    # 5. High social media use
    social_hours = float(student_row.get("social_media_hours", 0))
    if social_hours > rules_cfg.max_social_hours:
        meta = RULES_METADATA["high_social_media"]
        fired_rules.append({
            "rule_id": "high_social_media",
            "name": meta["name"],
            "trigger": f"Reported social media is {social_hours:.1f} hrs/day (threshold > {rules_cfg.max_social_hours:.1f} hrs/day)",
            "action": meta["action"],
            "label": meta["label"],
            "info_note": meta["info_note"],
            "is_indicator_only": True,
        })

    return fired_rules


def batch_evaluate_rules(df: pd.DataFrame, settings: Settings | None = None) -> list[list[dict[str, Any]]]:
    """Evaluates rules for all rows in a dataframe."""
    return [evaluate_student_rules(row, settings) for _, row in df.iterrows()]
