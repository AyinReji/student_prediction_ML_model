from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)

from src.features import TARGET_COL


def evaluate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Computes standard regression metrics MAE, RMSE, and R²."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
    }


def evaluate_prediction_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    half_width: float,
) -> dict[str, float]:
    """Checks empirical coverage and average width of the prediction interval."""
    lower = y_pred - half_width
    upper = y_pred + half_width
    covered = (y_true >= lower) & (y_true <= upper)
    coverage = float(np.mean(covered))
    return {
        "half_width": round(half_width, 4),
        "coverage": round(coverage, 4),
        "coverage_pct": round(coverage * 100, 2),
    }


def evaluate_at_risk(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_score: float,
    risk_margin: float,
) -> dict[str, Any]:
    """Evaluates at-risk classification:
    - Ground truth: y_true < target_score
    - Flagged: y_pred < (target_score + risk_margin)
    """
    actual_at_risk = (y_true < target_score).astype(int)
    predicted_at_risk = (y_pred < (target_score + risk_margin)).astype(int)

    n_total = len(y_true)
    n_actual_at_risk = int(np.sum(actual_at_risk))
    prevalence = float(n_actual_at_risk / n_total) if n_total > 0 else 0.0

    cm = confusion_matrix(actual_at_risk, predicted_at_risk, labels=[0, 1])
    # cm: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    prec, rec, f1, _ = precision_recall_fscore_support(
        actual_at_risk, predicted_at_risk, average="binary", zero_division=0
    )

    return {
        "target_score": float(target_score),
        "risk_margin": float(risk_margin),
        "total_students": n_total,
        "actual_at_risk_count": n_actual_at_risk,
        "prevalence": round(prevalence, 4),
        "prevalence_pct": round(prevalence * 100, 2),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        },
    }


def evaluate_fairness(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    target_score: float,
    risk_margin: float,
) -> dict[str, list[dict[str, Any]]]:
    """Audits model performance and bias across gender and age demographic slices.
    Mean residual is computed as actual - predicted (positive = underprediction).
    """
    df_eval = df.copy()
    df_eval["y_true"] = df_eval[TARGET_COL].values
    df_eval["y_pred"] = y_pred
    df_eval["residual"] = df_eval["y_true"] - df_eval["y_pred"]
    df_eval["actual_at_risk"] = (df_eval["y_true"] < target_score).astype(int)
    df_eval["predicted_at_risk"] = (df_eval["y_pred"] < (target_score + risk_margin)).astype(int)

    # 1. Gender breakdown
    gender_rows = []
    for group_name, grp in df_eval.groupby("gender"):
        n = len(grp)
        mae = float(mean_absolute_error(grp["y_true"], grp["y_pred"]))
        mean_res = float(grp["residual"].mean())
        prec, rec, f1, _ = precision_recall_fscore_support(
            grp["actual_at_risk"], grp["predicted_at_risk"], average="binary", zero_division=0
        )
        small_sample_note = "Sample size < 5% of cohort; treat estimates with caution" if n < (0.05 * len(df_eval)) else "Adequate sample size"
        gender_rows.append({
            "group": str(group_name),
            "count": n,
            "share_pct": round(n / len(df_eval) * 100, 2),
            "mae": round(mae, 4),
            "mean_residual": round(mean_res, 4),
            "at_risk_recall": round(float(rec), 4),
            "at_risk_precision": round(float(prec), 4),
            "notes": small_sample_note,
        })

    # 2. Age bands
    bins = [0, 16, 19, 22, 100]
    labels = ["<= 16", "17 - 19", "20 - 22", "23+"]
    df_eval["age_band"] = pd.cut(df_eval["age"], bins=bins, labels=labels, right=True)

    age_rows = []
    for group_name, grp in df_eval.groupby("age_band", observed=False):
        n = len(grp)
        if n == 0:
            continue
        mae = float(mean_absolute_error(grp["y_true"], grp["y_pred"]))
        mean_res = float(grp["residual"].mean())
        prec, rec, f1, _ = precision_recall_fscore_support(
            grp["actual_at_risk"], grp["predicted_at_risk"], average="binary", zero_division=0
        )
        age_rows.append({
            "group": str(group_name),
            "count": n,
            "share_pct": round(n / len(df_eval) * 100, 2),
            "mae": round(mae, 4),
            "mean_residual": round(mean_res, 4),
            "at_risk_recall": round(float(rec), 4),
            "at_risk_precision": round(float(prec), 4),
            "notes": "Adequate sample size",
        })

    return {
        "gender": gender_rows,
        "age_band": age_rows,
    }


def evaluate_priority_queue_baselines(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    priority_scores: np.ndarray,
    target_score: float,
    capacities: list[int] = [10, 25, 50, 100],
) -> list[dict[str, Any]]:
    """Compares the combined Priority Queue against two standard baselines:
    - Baseline 1: Rank purely ascending by predicted exam score
    - Baseline 2: Rank purely ascending by average assessment marks (series_avg)
    Evaluates recall@N, precision@N, and ranking overlap.
    """
    y_true = df[TARGET_COL].values
    truly_at_risk_mask = y_true < target_score
    total_truly_at_risk = int(np.sum(truly_at_risk_mask))

    series_avg = df.get("series_avg", (df["series_1"] + df["series_2"]) / 2.0).values

    # System ranking: priority score descending, break ties with pred ascending
    sys_order = np.lexsort((y_pred, -priority_scores))

    # Baseline 1: Predicted score ascending
    base_pred_order = np.argsort(y_pred)

    # Baseline 2: Series average ascending
    base_series_order = np.argsort(series_avg)

    results = []
    for cap in capacities:
        n = min(cap, len(df))

        sys_indices = set(sys_order[:n])
        pred_indices = set(base_pred_order[:n])
        series_indices = set(base_series_order[:n])

        # System metrics
        sys_hits = sum(1 for i in sys_indices if truly_at_risk_mask[i])
        sys_rec = sys_hits / total_truly_at_risk if total_truly_at_risk > 0 else 0.0
        sys_prec = sys_hits / n if n > 0 else 0.0

        # Baseline 1 metrics (pred score)
        pred_hits = sum(1 for i in pred_indices if truly_at_risk_mask[i])
        pred_rec = pred_hits / total_truly_at_risk if total_truly_at_risk > 0 else 0.0
        pred_prec = pred_hits / n if n > 0 else 0.0

        # Baseline 2 metrics (series avg)
        series_hits = sum(1 for i in series_indices if truly_at_risk_mask[i])
        series_rec = series_hits / total_truly_at_risk if total_truly_at_risk > 0 else 0.0
        series_prec = series_hits / n if n > 0 else 0.0

        overlap_pred = len(sys_indices.intersection(pred_indices)) / n if n > 0 else 0.0
        overlap_series = len(sys_indices.intersection(series_indices)) / n if n > 0 else 0.0

        results.append({
            "capacity": n,
            "total_at_risk_in_cohort": total_truly_at_risk,
            "system_queue": {
                "hits": sys_hits,
                "recall_at_n": round(sys_rec, 4),
                "precision_at_n": round(sys_prec, 4),
            },
            "baseline_predicted_score": {
                "hits": pred_hits,
                "recall_at_n": round(pred_rec, 4),
                "precision_at_n": round(pred_prec, 4),
            },
            "baseline_series_avg": {
                "hits": series_hits,
                "recall_at_n": round(series_rec, 4),
                "precision_at_n": round(series_prec, 4),
            },
            "overlap_with_pred_baseline": round(overlap_pred, 4),
            "overlap_with_series_baseline": round(overlap_series, 4),
        })

    return results
