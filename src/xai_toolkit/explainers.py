"""Pure SHAP computation — no MCP imports, no narrative generation (ADR-001).

This module computes explainability artifacts from models and data.
All functions return structured data (Pydantic models or dicts),
never English text. Narrative generation is handled by narrators.py.
"""

import numpy as np
import pandas as pd
import shap

from xai_toolkit.schemas import ShapResult


def compute_shap_values(
    model,
    X: pd.DataFrame,
    sample_index: int,
    target_names: list[str] | None = None,
) -> ShapResult:
    """Compute SHAP values for a single sample.

    Args:
        model: A fitted model with a .predict_proba() method.
        X: Feature matrix (test set). Rows are samples, columns are features.
        sample_index: Which row in X to explain.
        target_names: Human-readable class names, e.g. ["malignant", "benign"].
            If None, uses ["class_0", "class_1"].

    Returns:
        ShapResult with prediction info, SHAP values, and feature values.

    Raises:
        IndexError: If sample_index is out of range.

    Example:
        >>> result = compute_shap_values(model, X_test, sample_index=42)
        >>> result.prediction_label
        'benign'
        >>> result.shap_values["worst_radius"]
        0.28
    """
    if sample_index < 0 or sample_index >= len(X):
        raise IndexError(
            f"Sample index {sample_index} is out of range. "
            f"Dataset has {len(X)} samples (valid indices: 0–{len(X) - 1})."
        )

    if target_names is None:
        target_names = ["class_0", "class_1"]

    # Get prediction for this sample
    sample = X.iloc[[sample_index]]
    prediction = int(model.predict(sample)[0])
    probability = float(model.predict_proba(sample)[0, 1])
    prediction_label = target_names[prediction]

    # Compute SHAP values. We pass model.predict_proba as a callable
    # to avoid version-specific model introspection issues between
    # shap and xgboost. The background sample provides the baseline.
    background = X.sample(n=min(50, len(X)), random_state=42)
    explainer = shap.Explainer(model.predict_proba, background)
    shap_explanation = explainer(X.iloc[[sample_index]])

    # predict_proba returns shape (n_samples, n_classes), so SHAP values
    # have shape (1, n_features, n_classes). Take positive class (index 1).
    shap_vals = shap_explanation.values[0]
    if shap_vals.ndim > 1:
        shap_vals = shap_vals[:, 1]

    base_value = shap_explanation.base_values[0]
    if isinstance(base_value, (list, np.ndarray)):
        base_value = float(base_value[1])  # positive class
    else:
        base_value = float(base_value)

    # Build feature name → SHAP value mapping
    feature_names = list(X.columns)
    shap_dict = {
        name: round(float(val), 6) for name, val in zip(feature_names, shap_vals)
    }
    feature_value_dict = {
        name: round(float(sample[name].iloc[0]), 6) for name in feature_names
    }

    return ShapResult(
        prediction=prediction,
        prediction_label=prediction_label,
        probability=round(probability, 4),
        base_value=round(base_value, 6),
        shap_values=shap_dict,
        feature_values=feature_value_dict,
        feature_names=feature_names,
    )
