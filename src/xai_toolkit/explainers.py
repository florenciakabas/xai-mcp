"""Pure SHAP computation — no MCP imports, no narrative generation (ADR-001).

This module computes explainability artifacts from models and data.
All functions return structured data (Pydantic models or dicts),
never English text. Narrative generation is handled by narrators.py.
"""

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import partial_dependence

from xai_toolkit.schemas import (
    DatasetDescription,
    FeatureImportance,
    ModelSummary,
    PartialDependenceResult,
    ShapResult,
)


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


def compute_global_feature_importance(
    model,
    X: pd.DataFrame,
    target_names: list[str] | None = None,
) -> list[FeatureImportance]:
    """Compute mean absolute SHAP values across all samples (global importance).

    This is more expensive than local explanations — it runs SHAP on every
    sample in X. For large datasets in production, use pre-computed artifacts.

    Args:
        model: A fitted model with a .predict_proba() method.
        X: Feature matrix to explain across.
        target_names: Human-readable class names.

    Returns:
        List of FeatureImportance, sorted by importance (highest first).

    Example:
        >>> importances = compute_global_feature_importance(model, X_test)
        >>> importances[0].name
        'worst_radius'
        >>> importances[0].importance
        0.15
    """
    background = X.sample(n=min(50, len(X)), random_state=42)
    explainer = shap.Explainer(model.predict_proba, background)

    # Compute SHAP for all samples (this is the expensive call)
    shap_explanation = explainer(X)

    # Shape: (n_samples, n_features, n_classes) — take positive class
    shap_vals = shap_explanation.values
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1]

    # Mean absolute SHAP = global importance; mean signed SHAP = direction
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    mean_signed = np.mean(shap_vals, axis=0)

    feature_names = list(X.columns)
    importances = []
    for i, name in enumerate(feature_names):
        importances.append(
            FeatureImportance(
                name=name,
                importance=round(float(mean_abs[i]), 6),
                direction="positive" if mean_signed[i] > 0 else "negative",
                mean_shap=round(float(mean_signed[i]), 6),
            )
        )

    # Sort by importance (highest first)
    importances.sort(key=lambda f: f.importance, reverse=True)
    return importances


def compute_model_summary(
    model,
    X: pd.DataFrame,
    metadata: dict,
    top_n: int = 5,
) -> ModelSummary:
    """Compute a summary of model behavior and top features.

    Args:
        model: A fitted model.
        X: Feature matrix (test set).
        metadata: Model metadata dict (from registry).
        top_n: Number of top features to include.

    Returns:
        ModelSummary with model info and top feature importances.
    """
    target_names = metadata.get("target_names", ["class_0", "class_1"])
    importances = compute_global_feature_importance(model, X, target_names)

    return ModelSummary(
        model_type=metadata.get("model_type", "unknown"),
        accuracy=metadata.get("accuracy", 0.0),
        n_features=len(X.columns),
        n_train_samples=metadata.get("n_train_samples", 0),
        n_test_samples=metadata.get("n_test_samples", len(X)),
        target_names=target_names,
        top_features=importances[:top_n],
    )


def compute_partial_dependence(
    model,
    X: pd.DataFrame,
    feature_name: str,
    grid_resolution: int = 50,
) -> PartialDependenceResult:
    """Compute partial dependence of prediction on a single feature.

    Partial dependence shows how the average prediction changes as one
    feature varies, while all other features are held at their observed values.

    Args:
        model: A fitted model with a .predict_proba() method.
        X: Feature matrix.
        feature_name: Name of the feature to analyze.
        grid_resolution: Number of grid points to evaluate.

    Returns:
        PartialDependenceResult with grid values and predictions.

    Raises:
        ValueError: If feature_name is not in X.columns.
    """
    if feature_name not in X.columns:
        # Find closest match for helpful error message
        from difflib import get_close_matches

        close = get_close_matches(feature_name, list(X.columns), n=3, cutoff=0.4)
        available = list(X.columns)
        raise ValueError(
            f"Feature '{feature_name}' not found. "
            f"Did you mean: {close}? "
            f"Available features: {available}"
        )

    feature_index = list(X.columns).index(feature_name)

    # Compute both average (PDP) and individual (ICE) curves.
    # PDP = average effect across all samples.
    # ICE = per-sample effect — shows heterogeneity the average hides.
    result_avg = partial_dependence(
        model,
        X,
        features=[feature_index],
        kind="average",
        grid_resolution=grid_resolution,
    )
    result_ice = partial_dependence(
        model,
        X,
        features=[feature_index],
        kind="individual",
        grid_resolution=grid_resolution,
    )

    grid_values = result_avg["grid_values"][0]
    avg_predictions = result_avg["average"][0]

    # ICE: result_ice["individual"] shape (n_samples, n_grid_points)
    ice_raw = result_ice["individual"][0]  # (n_samples, n_grid_points)
    ice_curves = [
        [round(float(val), 6) for val in row]
        for row in ice_raw
    ]

    return PartialDependenceResult(
        feature_name=feature_name,
        feature_values=[round(float(v), 6) for v in grid_values],
        predictions=[round(float(p), 6) for p in avg_predictions],
        ice_curves=ice_curves,
        feature_min=round(float(grid_values.min()), 6),
        feature_max=round(float(grid_values.max()), 6),
        prediction_min=round(float(avg_predictions.min()), 6),
        prediction_max=round(float(avg_predictions.max()), 6),
    )


def compute_dataset_description(
    X: pd.DataFrame,
    y: pd.Series,
    target_names: list[str] | None = None,
) -> DatasetDescription:
    """Compute descriptive statistics about a dataset.

    Pure computation — no narrative, no side effects.

    Args:
        X: Feature matrix.
        y: Target variable.
        target_names: Human-readable class names.

    Returns:
        DatasetDescription with shape, stats, and class distribution.
    """
    if target_names is None:
        target_names = [f"class_{i}" for i in sorted(y.unique())]

    # Class distribution: map numeric labels to names
    class_counts = y.value_counts().sort_index()
    class_distribution = {
        target_names[int(label)]: int(count)
        for label, count in class_counts.items()
    }

    # Per-feature stats
    feature_stats = {}
    for col in X.columns:
        feature_stats[col] = {
            "mean": round(float(X[col].mean()), 4),
            "std": round(float(X[col].std()), 4),
            "min": round(float(X[col].min()), 4),
            "max": round(float(X[col].max()), 4),
        }

    return DatasetDescription(
        n_samples=len(X),
        n_features=len(X.columns),
        feature_names=list(X.columns),
        class_distribution=class_distribution,
        missing_values=int(X.isna().sum().sum()),
        feature_stats=feature_stats,
    )
