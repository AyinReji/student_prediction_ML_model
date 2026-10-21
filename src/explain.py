from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_DISPLAY_NAMES = {
    "series_1": "Series 1 Exam Score",
    "series_2": "Series 2 Exam Score",
    "series_avg": "Average Series Score",
    "series_delta": "Assessment Trajectory (Series 2 - Series 1)",
    "study_hours_per_day": "Daily Study Hours",
    "attendance_percentage": "Attendance Rate (%)",
    "social_media_hours": "Daily Social Media Hours",
    "age": "Student Age",
    "gender": "Gender",
    "gender_Female": "Gender (Female)",
    "gender_Male": "Gender (Male)",
    "gender_Other": "Gender (Other)",
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
        self._is_col_transformer = False
        self._prep = None
        self._trans_names = []
        self._trans_means = None
        self._coefs = None

        self._initialize_explainer()

    def _initialize_explainer(self) -> None:
        regressor = None
        if isinstance(self.model, Pipeline):
            regressor = self.model.named_steps.get("regressor")
        elif hasattr(self.model, "coef_"):
            regressor = self.model

        if regressor is not None and hasattr(regressor, "coef_"):
            self.explainer_type = "linear"
            self._intercept = float(getattr(regressor, "intercept_", 0.0))
            prep = self.model.named_steps.get("prep") if isinstance(self.model, Pipeline) else None
            scaler = self.model.named_steps.get("scaler") if isinstance(self.model, Pipeline) else None

            if prep is not None and isinstance(prep, ColumnTransformer):
                self._is_col_transformer = True
                self._prep = prep
                self._coefs = regressor.coef_
                # Compute transformed feature names
                trans_names = []
                for name, trans, cols in prep.transformers_:
                    if name == "remainder" and trans == "drop":
                        continue
                    if hasattr(trans, "get_feature_names_out"):
                        trans_names.extend(trans.get_feature_names_out(cols).tolist())
                    else:
                        trans_names.extend(cols)
                self._trans_names = trans_names

                # Estimate transformed means from training data or scaler
                if self.training_data is not None:
                    train_df = self.training_data if isinstance(self.training_data, pd.DataFrame) else pd.DataFrame(self.training_data, columns=self.feature_names)
                    self._trans_means = np.mean(prep.transform(train_df), axis=0)
                else:
                    self._trans_means = np.zeros(len(self._trans_names))

            elif scaler is not None and hasattr(scaler, "scale_"):
                self._is_col_transformer = False
                self._feature_means = scaler.mean_
                self._unscaled_coefs = regressor.coef_ / scaler.scale_
            else:
                self._is_col_transformer = False
                if self.training_data is not None:
                    self._feature_means = np.mean(self.training_data, axis=0)
                else:
                    self._feature_means = np.zeros(len(self.feature_names))
                self._unscaled_coefs = regressor.coef_

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
        top_k: int = 7,
    ) -> list[dict[str, Any]]:
        """Returns feature contributions in score points relative to average student benchmark."""
        contributions: list[dict[str, Any]] = []

        if self._is_col_transformer and self._prep is not None and self._coefs is not None:
            # Prepare single-row DataFrame
            if isinstance(student_features, dict):
                row_df = pd.DataFrame([student_features])
            elif isinstance(student_features, pd.Series):
                row_df = student_features.to_frame().T
            else:
                row_df = pd.DataFrame([student_features], columns=self.feature_names)

            # Ensure all expected columns exist
            for col in self.feature_names:
                if col not in row_df.columns:
                    row_df[col] = 0.0 if col != "gender" else "Female"

            trans_val = self._prep.transform(row_df[self.feature_names])[0]
            points = (trans_val - self._trans_means) * self._coefs

            # Aggregate sub-features (e.g., one-hot gender columns)
            combined_contribs: dict[str, dict[str, Any]] = {}
            for fn, pt in zip(self._trans_names, points):
                if fn.startswith("gender_") or fn == "gender":
                    val_str = str(row_df["gender"].iloc[0]) if "gender" in row_df.columns else "Recorded"
                    if "gender" not in combined_contribs:
                        combined_contribs["gender"] = {
                            "feature": "gender",
                            "display_name": "Gender",
                            "value": val_str,
                            "points": 0.0,
                        }
                    combined_contribs["gender"]["points"] += float(pt)
                else:
                    raw_val = float(row_df[fn].iloc[0]) if fn in row_df.columns else 0.0
                    combined_contribs[fn] = {
                        "feature": fn,
                        "display_name": FEATURE_DISPLAY_NAMES.get(fn, fn),
                        "value": raw_val,
                        "points": float(pt),
                    }

            for k, item in combined_contribs.items():
                pt = round(item["points"], 2)
                contributions.append({
                    "feature": item["feature"],
                    "display_name": item["display_name"],
                    "value": item["value"],
                    "points": pt,
                    "direction": "positive" if pt >= 0 else "negative",
                    "abs_impact": abs(pt),
                })

        elif self.explainer_type == "linear" and self._unscaled_coefs is not None:
            if isinstance(student_features, dict):
                values = np.array([float(student_features.get(f, 0.0)) for f in self.feature_names])
            elif isinstance(student_features, pd.Series):
                values = np.array([float(student_features[f]) for f in self.feature_names])
            else:
                values = np.array(student_features).flatten()

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
            if isinstance(student_features, dict):
                values = np.array([float(student_features.get(f, 0.0)) for f in self.feature_names if f != "gender"])
            elif isinstance(student_features, pd.Series):
                values = np.array([float(student_features[f]) for f in self.feature_names if f != "gender"])
            else:
                values = np.array(student_features).flatten()

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
            # Fallback
            for feat in self.feature_names:
                contributions.append({
                    "feature": feat,
                    "display_name": FEATURE_DISPLAY_NAMES.get(feat, feat),
                    "value": 0.0,
                    "points": 0.0,
                    "direction": "neutral",
                    "abs_impact": 0.0,
                })

        # Sort descending by absolute impact in score points
        contributions.sort(key=lambda x: x["abs_impact"], reverse=True)
        return contributions[:top_k]
