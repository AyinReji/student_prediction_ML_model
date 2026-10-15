from src.config import Settings
from src.recommend import evaluate_student_rules


def test_rule_low_assessment_marks():
    settings = Settings()
    # Trigger: series_avg < 18.0
    positive_case = {"series_1": 15.0, "series_2": 17.0, "attendance_percentage": 90, "study_hours_per_day": 3.0, "social_media_hours": 1.0}
    negative_case = {"series_1": 25.0, "series_2": 25.0, "attendance_percentage": 90, "study_hours_per_day": 3.0, "social_media_hours": 1.0}

    fired_pos = [r["rule_id"] for r in evaluate_student_rules(positive_case, settings)]
    fired_neg = [r["rule_id"] for r in evaluate_student_rules(negative_case, settings)]

    assert "low_assessment_marks" in fired_pos
    assert "low_assessment_marks" not in fired_neg


def test_rule_declining_marks():
    settings = Settings()
    # Trigger: series_delta <= -4.0
    positive_case = {"series_1": 30.0, "series_2": 24.0, "attendance_percentage": 90, "study_hours_per_day": 3.0, "social_media_hours": 1.0}
    negative_case = {"series_1": 24.0, "series_2": 24.0, "attendance_percentage": 90, "study_hours_per_day": 3.0, "social_media_hours": 1.0}

    fired_pos = [r["rule_id"] for r in evaluate_student_rules(positive_case, settings)]
    fired_neg = [r["rule_id"] for r in evaluate_student_rules(negative_case, settings)]

    assert "declining_marks" in fired_pos
    assert "declining_marks" not in fired_neg


def test_rule_attendance_concern():
    settings = Settings()
    # Trigger: attendance < 75.0%
    positive_case = {"series_1": 25.0, "series_2": 25.0, "attendance_percentage": 65.0, "study_hours_per_day": 3.0, "social_media_hours": 1.0}
    negative_case = {"series_1": 25.0, "series_2": 25.0, "attendance_percentage": 85.0, "study_hours_per_day": 3.0, "social_media_hours": 1.0}

    fired_pos = [r["rule_id"] for r in evaluate_student_rules(positive_case, settings)]
    fired_neg = [r["rule_id"] for r in evaluate_student_rules(negative_case, settings)]

    assert "attendance_concern" in fired_pos
    assert "attendance_concern" not in fired_neg


def test_rule_low_study_time():
    settings = Settings()
    # Trigger: study_hours < 2.0
    positive_case = {"series_1": 25.0, "series_2": 25.0, "attendance_percentage": 90.0, "study_hours_per_day": 1.0, "social_media_hours": 1.0}
    negative_case = {"series_1": 25.0, "series_2": 25.0, "attendance_percentage": 90.0, "study_hours_per_day": 3.5, "social_media_hours": 1.0}

    fired_pos = [r["rule_id"] for r in evaluate_student_rules(positive_case, settings)]
    fired_neg = [r["rule_id"] for r in evaluate_student_rules(negative_case, settings)]

    assert "low_study_time" in fired_pos
    assert "low_study_time" not in fired_neg


def test_rule_high_social_media():
    settings = Settings()
    # Trigger: social_media > 4.0
    positive_case = {"series_1": 25.0, "series_2": 25.0, "attendance_percentage": 90.0, "study_hours_per_day": 3.0, "social_media_hours": 5.5}
    negative_case = {"series_1": 25.0, "series_2": 25.0, "attendance_percentage": 90.0, "study_hours_per_day": 3.0, "social_media_hours": 2.0}

    fired_pos = [r["rule_id"] for r in evaluate_student_rules(positive_case, settings)]
    fired_neg = [r["rule_id"] for r in evaluate_student_rules(negative_case, settings)]

    assert "high_social_media" in fired_pos
    assert "high_social_media" not in fired_neg
