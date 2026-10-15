import numpy as np
import pandas as pd

from src.config import Settings
from src.priority import build_priority_queue


def test_priority_queue_returns_n_rows_sorted():
    """Hard requirement: Priority queue returns exactly N rows, sorted descending by priority_score."""
    settings = Settings(capacity=20, target_score=40.0)

    # Synthetic sample of 100 students
    np.random.seed(42)
    n_sample = 100
    df = pd.DataFrame({
        "student_id": range(n_sample),
        "series_1": np.random.uniform(5, 35, n_sample),
        "series_2": np.random.uniform(5, 35, n_sample),
        "attendance_percentage": np.random.uniform(50, 100, n_sample),
        "study_hours_per_day": np.random.uniform(0.5, 6.0, n_sample),
        "social_media_hours": np.random.uniform(0, 6, n_sample),
        "series_avg": np.random.uniform(5, 35, n_sample),
        "series_delta": np.random.uniform(-10, 10, n_sample),
    })

    pred_scores = np.random.uniform(20, 80, n_sample)
    residual_std = 6.5

    capacity = 15
    queue = build_priority_queue(
        df=df,
        predicted_scores=pred_scores,
        residual_std=residual_std,
        settings=settings,
        capacity=capacity,
    )

    assert len(queue) == capacity
    assert "rank" in queue.columns
    assert list(queue["rank"]) == list(range(1, capacity + 1))

    # Assert descending order of priority scores
    p_scores = queue["priority_score"].values
    assert all(p_scores[i] >= p_scores[i + 1] for i in range(len(p_scores) - 1))
