import numpy as np
import pandas as pd
from scipy.stats import norm

from src.config import Settings, load_settings
from src.recommend import TOTAL_RULES_COUNT, evaluate_student_rules


def calculate_p_risk(predicted_scores: np.ndarray, target: float, residual_std: float) -> np.ndarray:
    """Calculates risk probability using normal approximation Φ((target - predicted) / σ).
    Higher p_risk means higher probability that actual final_exam_score < target.
    """
    if residual_std <= 0:
        residual_std = 1.0
    z = (target - predicted_scores) / residual_std
    return norm.cdf(z)


def get_risk_badge(p_risk: float, high_thresh: float = 0.6, med_thresh: float = 0.3) -> str:
    """Returns risk category badge: High, Medium, or Low."""
    if p_risk >= high_thresh:
        return "High"
    elif p_risk >= med_thresh:
        return "Medium"
    else:
        return "Low"


def compute_priority_components(
    df: pd.DataFrame,
    predicted_scores: np.ndarray,
    target_score: float,
    residual_std: float,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Calculates p_risk, decline_flag, flag_share, priority_score, and risk_badge for each student."""
    if settings is None:
        settings = load_settings()

    p_risk = calculate_p_risk(predicted_scores, target_score, residual_std)

    decline_flags = []
    flag_shares = []
    top_reasons = []
    suggested_actions = []

    for idx, (_, row) in enumerate(df.iterrows()):
        rules = evaluate_student_rules(row, settings)
        n_fired = len(rules)
        flag_shares.append(n_fired / TOTAL_RULES_COUNT)

        has_decline = any(r["rule_id"] == "declining_marks" for r in rules)
        decline_flags.append(1.0 if has_decline else 0.0)

        if rules:
            top_reasons.append(rules[0]["name"])
            suggested_actions.append(rules[0]["action"])
        else:
            top_reasons.append("Projected performance review")
            suggested_actions.append("General check-in")

    p_risk_arr = np.array(p_risk)
    decline_arr = np.array(decline_flags)
    flags_arr = np.array(flag_shares)

    w = settings.priority_weights
    priority_scores = w.risk * p_risk_arr + w.decline * decline_arr + w.flags * flags_arr

    badges = [
        get_risk_badge(p, settings.risk_badges.high, settings.risk_badges.medium)
        for p in p_risk_arr
    ]

    out_df = df.copy()
    out_df["predicted_score"] = np.round(predicted_scores, 2)
    out_df["p_risk"] = np.round(p_risk_arr, 4)
    out_df["decline_flag"] = decline_arr
    out_df["flag_share"] = np.round(flags_arr, 4)
    out_df["priority_score"] = np.round(priority_scores, 4)
    out_df["risk_badge"] = badges
    out_df["top_reason"] = top_reasons
    out_df["suggested_action"] = suggested_actions

    return out_df


def build_priority_queue(
    df: pd.DataFrame,
    predicted_scores: np.ndarray,
    residual_std: float,
    settings: Settings | None = None,
    capacity: int | None = None,
) -> pd.DataFrame:
    """Builds and returns the top-N prioritized students sorted descending by priority_score."""
    if settings is None:
        settings = load_settings()
    if capacity is None:
        capacity = settings.capacity

    scored_df = compute_priority_components(
        df=df,
        predicted_scores=predicted_scores,
        target_score=settings.target_score,
        residual_std=residual_std,
        settings=settings,
    )

    # Sort descending by priority score; tie-break by predicted score ascending
    sorted_df = scored_df.sort_values(
        by=["priority_score", "predicted_score"],
        ascending=[False, True]
    ).reset_index(drop=True)

    sorted_df["rank"] = range(1, len(sorted_df) + 1)

    return sorted_df.head(capacity)
