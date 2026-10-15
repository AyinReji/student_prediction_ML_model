from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_DISPLAY_NAMES = {
    "series_1": "Series 1 Exam Score",
    "series_2": "Series 2 Exam Score",
    "series_avg": "Average Series Score",
    "series_delta": "Assessment Trajectory (Series 2 - Series 1)",
    "study_hours_per_day": "Study Hours / Day (Indicator)",
    "attendance_percentage": "Attendance Rate (Indicator)",
    "social_media_hours": "Social Media Hours / Day (Indicator)",
    "age": "Student Age (Ablation)",
    "gender_Female": "Gender (Female - Ablation)",
    "gender_Male": "Gender (Male - Ablation)",
    "gender_Other": "Gender (Other - Ablation)",
}


class ModelExplainer:
    def __init__(self, model: Any, feature_names: list[str], training_data: pd.DataFrame | np.ndarray | None = None):
        self.model = model
        self.feature_names = feature_names
        self.training_data = training_data
        self.explainer_type = "unknown"
        self._tree_explainer = None
        self._feature_means = None
        self._unscaled_coefs = None
        self._intercept = None

        self._initialize_explainer()

    def _initialize_explainer(self) -> None:
        # Check if LinearRegression in Pipeline
        if isinstance(self.model, Pipeline) and isinstance(self.model.named_steps.get("regressor"), LinearRegression):
            self.explainer_type = "linear"
            scaler: StandardScaler = self.model.named_steps.get("scaler")
            regressor: LinearRegression = self.model.named_steps.get("regressor")

            if scaler is not None and hasattr(scaler, "scale_"):
                self._feature_means = scaler.mean_
                self._unscaled_coefs = regressor.coef_ / scaler.scale_
            else:
                self._feature_means = np.zeros(len(self.feature_names))
                self._unscaled_coefs = regressor.coef_
            self._intercept = regressor.intercept_

        elif isinstance(self.model, LinearRegression):
            self.explainer_type = "linear"
            self._feature_means = np.mean(self.training_data, axis=0) if self.training_data is not None else np.zeros(len(self.feature_names))
            self._unscaled_coefs = self.model.coef_
            self._intercept = self.model.intercept_

        else:
            # Tree-based model (RandomForest or GradientBoosting)
            self.explainer_type = "tree"
            tree_model = self.model
            if isinstance(self.model, Pipeline):
                tree_model = self.model.named_steps.get("regressor", self.model)
            try:
                self._tree_explainer = shap.TreeExplainer(tree_model)
            except Exception:
                self._tree_explainer = None

    def explain_student(
        self,
        student_features: pd.Series | dict[str, Any] | np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Returns feature contributions in score points relative to average student benchmark."""
        if isinstance(student_features, dict):
            values = np.array([float(student_features.get(f, 0.0)) for f in self.feature_names])
        elif isinstance(student_features, pd.Series):
            values = np.array([float(student_features[f]) for f in self.feature_names])
        else:
            values = np.array(student_features).flatten()

        contributions: list[dict[str, Any]] = []

        if self.explainer_type == "linear" and self._unscaled_coefs is not None:
            # contribution_j = (x_j - mean_j) * beta_j
            delta_x = values - self._feature_means
            points = delta_x * self._unscaled_coefs

            for feat, val, pt in zip(self.feature_names, values, points):
                contributions.append({
                    "feature": feat,
                    "display_name": FEATURE_DISPLAY_NAMES.get(feat, feat),
                    "value": float(val),
                    "points": float(np.round(pt, 2)),
                    "direction": "positive" if pt >= 0 else "negative",
                    "abs_impact": float(abs(pt)),
                })

        elif self.explainer_type == "tree" and self._tree_explainer is not None:
            shap_values = self._tree_explainer.shap_values(values.reshape(1, -1))
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            shap_arr = np.array(shap_values).flatten()

            for feat, val, pt in zip(self.feature_names, values, shap_arr):
                contributions.append({
                    "feature": feat,
                    "display_name": FEATURE_DISPLAY_NAMES.get(feat, feat),
                    "value": float(val),
                    "points": float(np.round(pt, 2)),
                    "direction": "positive" if pt >= 0 else "negative",
                    "abs_impact": float(abs(pt)),
                })
        else:
            # Fallback heuristic using unit weights
            for feat, val in zip(self.feature_names, values):
                contributions.append({
                    "feature": feat,
                    "display_name": FEATURE_DISPLAY_NAMES.get(feat, feat),
                    "value": float(val),
                    "points": 0.0,
                    "direction": "neutral",
                    "abs_impact": 0.0,
                })

        # Sort descending by absolute impact in score points
        contributions.sort(key=lambda x: x["abs_impact"], reverse=True)
        return contributions[:top_k]
