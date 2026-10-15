from src.db import PROGRESS_CAVEAT
from src.recommend import INDICATOR_DISCLAIMER, RULES_METADATA


def test_no_causal_or_topic_claims_in_rules():
    """Hard requirement: Never claim causality ('because the student studies little')
    or topic/subject diagnoses ('algebra', 'physics', etc.).
    """
    forbidden_causal_phrases = [
        "because the student",
        "causes low scores",
        "caused low",
        "leads to failure",
        "due to poor study",
    ]

    forbidden_topic_words = [
        "algebra",
        "calculus",
        "physics",
        "chemistry",
        "biology",
        "topic diagnosis",
        "subject weakness",
    ]

    for rule_id, meta in RULES_METADATA.items():
        text_block = f"{meta['action']} {meta['info_note']} {meta['name']}".lower()

        for phrase in forbidden_causal_phrases:
            assert phrase not in text_block, f"Causal phrase '{phrase}' found in rule '{rule_id}'"

        for word in forbidden_topic_words:
            assert word not in text_block, f"Topic/subject claim '{word}' found in rule '{rule_id}'"


def test_indicator_only_rules_include_disclaimer():
    """Hard requirement: Rules marked 'Indicator only' must include the disclaimer that
    this factor shows no measurable relationship with final scores in this dataset.
    """
    for rule_id, meta in RULES_METADATA.items():
        if meta["label"] == "Indicator only":
            assert INDICATOR_DISCLAIMER in meta["info_note"]


def test_progress_tracking_caveat_present():
    """Hard requirement: Progress tracking must carry caveat that changes do not prove causality."""
    assert "do not prove" in PROGRESS_CAVEAT.lower()
    assert "regression to the mean" in PROGRESS_CAVEAT.lower()
