"""Pure SHAP computation — no MCP imports, no narrative generation (ADR-001).

This module computes explainability artifacts from models and data.
All functions return structured data (Pydantic models or dicts),
never English text. Narrative generation is handled by narrators.py.

Two modes of operation:
  1. ON-THE-FLY (PoC): Compute SHAP directly from a model + data.
     Used during the sprint for interactive exploration.
  2. FROM PIPELINE (Production path): Read pre-computed SHAP artifacts
     produced by the Kedro explainability pipeline (xai-xgboost-clf repo).
     The pipeline handles batch computation, background selection, and
     MLflow logging. Our toolkit reads its outputs and translates them
     into English narratives via narrators.py.

For production use, SHAP values should be pre-computed by the Kedro
explainability pipeline. The on-the-fly computation here is for PoC
and interactive exploration only.
"""

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import partial_dependence

logger = logging.getLogger(__name__)

from xai_toolkit.schemas import (
    DatasetDescription,
    FeatureImportance,
    ModelSummary,
    PartialDependenceResult,
    PredictionComparison,
    ShapResult,
    SingleModelPrediction,
)


def _validate_binary_target_names(target_names: list[str]) -> None:
    """Require exactly two target names for the supported binary contract."""
    if len(target_names) != 2:
        raise ValueError(
            "Only binary classification is supported. "
            f"Expected exactly 2 target_names, got {len(target_names)}."
        )


def _validate_numeric_columns(X: pd.DataFrame, name: str) -> None:
    """Require all columns to be numeric for SHAP and JSON serialization."""
    non_numeric = [
        col for col in X.columns
        if not pd.api.types.is_numeric_dtype(X[col])
    ]
    if non_numeric:
        raise ValueError(
            f"Only numeric feature matrices are supported. "
            f"{name} has non-numeric columns: {non_numeric}."
        )


def _validate_background_data(
    X: pd.DataFrame,
    background_data: pd.DataFrame | None,
) -> None:
    """Ensure background data matches the explained matrix exactly."""
    _validate_numeric_columns(X, "X")
    if background_data is None:
        return
    if background_data.empty:
        raise ValueError("background_data must contain at least 1 row.")
    if list(background_data.columns) != list(X.columns):
        raise ValueError(
            "background_data columns must exactly match X columns in the same order."
        )
    _validate_numeric_columns(background_data, "background_data")


def _validate_binary_prediction(prediction: object) -> int:
    """Require predictions to be numeric binary labels 0/1."""
    if not isinstance(prediction, (int, np.integer, bool, np.bool_)):
        raise ValueError(
            "Only binary classifiers with numeric class labels 0 and 1 are supported."
        )
    prediction_int = int(prediction)
    if prediction_int not in (0, 1):
        raise ValueError(
            "Only binary classifiers with numeric class labels 0 and 1 are supported."
        )
    return prediction_int


def _validate_binary_probabilities(probabilities: object) -> np.ndarray:
    """Require predict_proba outputs with shape (n_samples, 2)."""
    probabilities_array = np.asarray(probabilities)
    if probabilities_array.ndim != 2 or probabilities_array.shape[1] != 2:
        raise ValueError(
            "Only binary classifiers with predict_proba output shape "
            "(n_samples, 2) are supported."
        )
    return probabilities_array


def _extract_local_binary_shap_values(shap_explanation: object, n_features: int) -> np.ndarray:
    """Extract per-feature SHAP values for the supported binary contract."""
    shap_values = np.asarray(shap_explanation.values)
    if shap_values.ndim == 3:
        if shap_values.shape[2] != 2:
            raise ValueError(
                "Only binary SHAP outputs with 2 class dimensions are supported."
            )
        sample_values = shap_values[0, :, 1]
    elif shap_values.ndim == 2:
        sample_values = shap_values[0]
    else:
        raise ValueError(
            "Unsupported SHAP output shape. "
            "Expected 2D or 3D SHAP values for binary classification."
        )
    if len(sample_values) != n_features:
        raise ValueError(
            "SHAP output width does not match the explained feature matrix."
        )
    return sample_values


def _extract_local_binary_base_value(shap_explanation: object) -> float:
    """Extract the base value for the supported binary contract."""
    base_values = np.asarray(shap_explanation.base_values)
    if base_values.ndim == 0:
        return float(base_values)
    if base_values.ndim == 1:
        if len(base_values) == 1:
            return float(base_values[0])
        if len(base_values) == 2:
            return float(base_values[1])
    if base_values.ndim == 2:
        if base_values.shape[1] == 1:
            return float(base_values[0, 0])
        if base_values.shape[1] == 2:
            return float(base_values[0, 1])
    raise ValueError(
        "Unsupported SHAP base_values shape for binary classification."
    )


def _extract_global_binary_shap_values(shap_explanation: object, n_features: int) -> np.ndarray:
    """Extract global SHAP values for the supported binary contract."""
    shap_values = np.asarray(shap_explanation.values)
    if shap_values.ndim == 3:
        if shap_values.shape[2] != 2:
            raise ValueError(
                "Only binary SHAP outputs with 2 class dimensions are supported."
            )
        shap_values = shap_values[:, :, 1]
    elif shap_values.ndim != 2:
        raise ValueError(
            "Unsupported SHAP output shape. "
            "Expected 2D or 3D SHAP values for binary classification."
        )
    if shap_values.shape[1] != n_features:
        raise ValueError(
            "SHAP output width does not match the explained feature matrix."
        )
    return shap_values


def _require_pipeline_feature_names(metadata: dict) -> list[str]:
    """Validate pipeline metadata contains a consistent feature list."""
    feature_names = metadata.get("feature_names")
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError(
            "Pipeline metadata must include a non-empty 'feature_names' list."
        )
    n_features = metadata.get("n_features")
    if n_features is not None and int(n_features) != len(feature_names):
        raise ValueError(
            "Pipeline metadata 'n_features' does not match 'feature_names' length."
        )
    return feature_names


def extract_intrinsic_importances(
    model: object,
    feature_names: list[str],
) -> tuple[list[FeatureImportance], str] | None:
    """Extract intrinsic feature importances from models that expose them directly.

    Adapted from Tamas's _handle_intrinsically_explainable_model() in
    xai-xgboost-clf/src/xgboost_clf/pipelines/model_explanation/nodes.py
    (lines 116-162).

    Supports:
    - Linear models with coef_ (logistic regression, linear SVM, etc.)
    - Tree-based models with feature_importances_ (random forest, gradient boosting, etc.)

    Unlike the original which saves to disk (.npy + metadata JSON), we return
    in-memory Pydantic schemas per ADR-001 (separation of concerns).

    Args:
        model: A fitted model object.
        feature_names: List of feature names matching model's input columns.

    Returns:
        Tuple of (list of FeatureImportance sorted by absolute importance,
        source_attr string indicating "coef_" or "feature_importances_"),
        or None if the model is not intrinsically explainable.
    """
    importance_array = None
    source_attr = None

    if hasattr(model, "coef_"):
        importance_array = np.asarray(getattr(model, "coef_")).flatten()
        source_attr = "coef_"
        logger.debug("Extracted coefficients from intrinsically explainable model.")
    elif hasattr(model, "feature_importances_"):
        importance_array = np.asarray(getattr(model, "feature_importances_"))
        source_attr = "feature_importances_"
        logger.debug("Extracted feature importances from intrinsically explainable model.")

    if importance_array is None:
        logger.debug("Model is not intrinsically explainable (no coef_ or feature_importances_).")
        return None

    if len(importance_array) != len(feature_names):
        logger.warning(
            "Intrinsic importance length (%d) doesn't match feature count (%d); skipping.",
            len(importance_array), len(feature_names),
        )
        return None

    importances = []
    for i, name in enumerate(feature_names):
        val = float(importance_array[i])
        if source_attr == "coef_":
            # Coefficients have meaningful sign: positive = pushes toward positive class
            direction = "positive" if val >= 0 else "negative"
            imp = abs(val)
            mean_shap = val  # Use raw coefficient as "mean_shap" analogue
        else:
            # feature_importances_ are magnitude-based (always non-negative)
            direction = "positive"
            imp = val
            mean_shap = val

        importances.append(
            FeatureImportance(
                name=name,
                importance=round(imp, 6),
                direction=direction,
                mean_shap=round(mean_shap, 6),
            )
        )

    importances.sort(key=lambda f: f.importance, reverse=True)
    return importances, source_attr


def compute_shap_values(
    model,
    X: pd.DataFrame,
    sample_index: int,
    target_names: list[str] | None = None,
    background_data: pd.DataFrame | None = None,
) -> ShapResult:
    """Compute SHAP values for a single sample.

    Args:
        model: A fitted model with a .predict_proba() method.
        X: Feature matrix (test set). Rows are samples, columns are features.
        sample_index: Which row in X to explain.
        target_names: Human-readable class names, e.g. ["malignant", "benign"].
            If None, uses ["class_0", "class_1"].
        background_data: Optional training data to use as SHAP background
            distribution. When provided, follows Tamas's correct methodology
            (X_train for background, X_test for samples to explain). When None,
            falls back to sampling from X (test data) — see TD-14.

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
    _validate_binary_target_names(target_names)
    _validate_background_data(X, background_data)

    # Get prediction for this sample
    sample = X.iloc[[sample_index]]
    prediction = _validate_binary_prediction(model.predict(sample)[0])
    probabilities = _validate_binary_probabilities(model.predict_proba(sample))
    probability = float(probabilities[0, 1])
    prediction_label = target_names[prediction]

    # Compute SHAP values. We pass model.predict_proba as a callable
    # to avoid version-specific model introspection issues between
    # shap and xgboost. The background sample provides the baseline.
    # When background_data is provided (training data), use it per SHAP best
    # practices (Lundberg's recommendation, Tamas's explainability_node).
    if background_data is not None:
        background = background_data.sample(n=min(50, len(background_data)), random_state=42)
        logger.debug(
            "Computing SHAP for sample %d (background=%d from training data)",
            sample_index, len(background),
        )
    else:
        background = X.sample(n=min(50, len(X)), random_state=42)
        logger.debug(
            "Computing SHAP for sample %d (background=%d from test data — TD-14)",
            sample_index, len(background),
        )
    explainer = shap.Explainer(model.predict_proba, background)
    shap_explanation = explainer(X.iloc[[sample_index]])

    shap_vals = _extract_local_binary_shap_values(
        shap_explanation,
        n_features=len(X.columns),
    )
    base_value = _extract_local_binary_base_value(shap_explanation)

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


def compute_shap_values_batch(
    model,
    X: pd.DataFrame,
    sample_indices: list[int],
    target_names: list[str] | None = None,
    background_data: pd.DataFrame | None = None,
) -> list[ShapResult]:
    """Compute SHAP values for multiple samples with explainer reuse.

    Creates the SHAP explainer once and loops over sampled indices.
    Per-sample errors are caught, logged, and skipped — the caller
    receives only successful results.

    Args:
        model: A fitted model with predict_proba().
        X: Feature matrix (all rows available for explanation).
        sample_indices: Which rows in X to explain.
        target_names: Human-readable class names.
        background_data: Optional training data for SHAP background.

    Returns:
        List of ShapResult for successful samples. May be shorter than
        sample_indices if some samples failed.

    Raises:
        IndexError: If any index is out of range (validated upfront).
    """
    if target_names is None:
        target_names = ["class_0", "class_1"]
    _validate_binary_target_names(target_names)
    _validate_background_data(X, background_data)

    # Validate all indices upfront
    n_rows = len(X)
    for idx in sample_indices:
        if idx < 0 or idx >= n_rows:
            raise IndexError(
                f"Sample index {idx} is out of range. "
                f"Dataset has {n_rows} samples (valid indices: 0–{n_rows - 1})."
            )

    # Create explainer once
    bg_source = background_data if background_data is not None else X
    background = bg_source.sample(n=min(50, len(bg_source)), random_state=42)
    explainer = shap.Explainer(model.predict_proba, background)

    feature_names = list(X.columns)
    results: list[ShapResult] = []

    for idx in sample_indices:
        try:
            sample = X.iloc[[idx]]
            prediction = _validate_binary_prediction(model.predict(sample)[0])
            probabilities = _validate_binary_probabilities(model.predict_proba(sample))
            probability = float(probabilities[0, 1])
            prediction_label = target_names[prediction]

            shap_explanation = explainer(sample)
            shap_vals = _extract_local_binary_shap_values(
                shap_explanation, n_features=len(feature_names),
            )
            base_value = _extract_local_binary_base_value(shap_explanation)

            shap_dict = {
                name: round(float(val), 6)
                for name, val in zip(feature_names, shap_vals)
            }
            feature_value_dict = {
                name: round(float(sample[name].iloc[0]), 6)
                for name in feature_names
            }

            results.append(ShapResult(
                prediction=prediction,
                prediction_label=prediction_label,
                probability=round(probability, 4),
                base_value=round(base_value, 6),
                shap_values=shap_dict,
                feature_values=feature_value_dict,
                feature_names=feature_names,
            ))
        except Exception:
            logger.warning(
                "SHAP computation failed for sample %d, skipping.", idx,
                exc_info=True,
            )

    logger.info(
        "Batch SHAP: %d/%d samples succeeded.",
        len(results), len(sample_indices),
    )
    return results


def compute_global_feature_importance(
    model,
    X: pd.DataFrame,
    target_names: list[str] | None = None,
    background_data: pd.DataFrame | None = None,
) -> list[FeatureImportance]:
    """Compute mean absolute SHAP values across all samples (global importance).

    This is more expensive than local explanations — it runs SHAP on every
    sample in X. For large datasets in production, use pre-computed artifacts.

    Args:
        model: A fitted model with a .predict_proba() method.
        X: Feature matrix to explain across.
        target_names: Human-readable class names.
        background_data: Optional training data for SHAP background (TD-14).

    Returns:
        List of FeatureImportance, sorted by importance (highest first).

    Example:
        >>> importances = compute_global_feature_importance(model, X_test)
        >>> importances[0].name
        'worst_radius'
        >>> importances[0].importance
        0.15
    """
    _validate_background_data(X, background_data)
    _validate_binary_probabilities(model.predict_proba(X.iloc[:1]))
    bg_source = background_data if background_data is not None else X
    background = bg_source.sample(n=min(50, len(bg_source)), random_state=42)
    logger.debug(
        "Computing global SHAP importance (samples=%d, background=%d, source=%s)",
        len(X), len(background),
        "training_data" if background_data is not None else "test_data",
    )
    explainer = shap.Explainer(model.predict_proba, background)

    # Compute SHAP for all samples (this is the expensive call)
    shap_explanation = explainer(X)

    shap_vals = _extract_global_binary_shap_values(
        shap_explanation,
        n_features=len(X.columns),
    )

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
    background_data: pd.DataFrame | None = None,
) -> ModelSummary:
    """Compute a summary of model behavior and top features.

    Args:
        model: A fitted model.
        X: Feature matrix (test set).
        metadata: Model metadata dict (from registry).
        top_n: Number of top features to include.
        background_data: Optional training data for SHAP background (TD-14).

    Returns:
        ModelSummary with model info and top feature importances.
    """
    target_names = metadata.get("target_names", ["class_0", "class_1"])
    importances = compute_global_feature_importance(model, X, target_names, background_data=background_data)

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


def compute_data_hash(X: pd.DataFrame, sample_index: int | None = None) -> str:
    """Compute a SHA256 hash of a DataFrame for audit purposes (D3-S2).

    The hash captures the exact data values used to generate an explanation,
    creating an auditable link between output and input. Same data → same hash,
    always. Useful for confirming that two explanations were generated from
    identical inputs.

    Args:
        X: The feature matrix.
        sample_index: If provided, hash only that single row.
                      If None, hash the entire matrix.

    Returns:
        SHA256 hex digest (64 hex characters).

    Example:
        >>> h = compute_data_hash(X_test, sample_index=42)
        >>> len(h)
        64
        >>> h == compute_data_hash(X_test, sample_index=42)  # always True
        True
    """
    data = X.iloc[[sample_index]] if sample_index is not None else X
    # to_csv gives deterministic text serialization regardless of DataFrame
    # internal memory layout. index=False ensures row numbers don't affect hash.
    csv_bytes = data.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


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


def _extract_top_features(
    shap_result: ShapResult,
    top_n: int = 3,
) -> list[dict]:
    """Extract top N SHAP contributors from a ShapResult.

    Sorted by absolute SHAP magnitude. Each entry includes the feature
    name, its SHAP value, actual feature value, and direction.

    Args:
        shap_result: Output from compute_shap_values().
        top_n: Number of top features to return.

    Returns:
        List of dicts with keys: name, shap_value, feature_value, direction.
    """
    sorted_features = sorted(
        shap_result.shap_values.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    top = []
    for name, shap_val in sorted_features[:top_n]:
        top.append({
            "name": name,
            "shap_value": shap_val,
            "feature_value": shap_result.feature_values.get(name, 0.0),
            "direction": "positive" if shap_val > 0 else "negative",
        })
    return top


def compute_prediction_comparison(
    models: list[tuple[str, object, dict]],
    X: pd.DataFrame,
    sample_index: int,
    top_n: int = 3,
    background_data: pd.DataFrame | None = None,
) -> PredictionComparison:
    """Compare predictions from two models on the same sample.

    Computes SHAP values for each model, then analyzes agreement,
    confidence gap, and shared/divergent feature drivers.

    Args:
        models: List of (model_id, fitted_model, metadata) tuples.
        X: Shared feature matrix (test set). Both models must use the same data.
        sample_index: Which row in X to compare.
        top_n: Number of top SHAP contributors per model (default: 3).
        background_data: Optional training data for SHAP background (TD-14).

    Returns:
        PredictionComparison with per-model results and agreement analysis.

    Raises:
        IndexError: If sample_index is out of range.
        ValueError: If fewer than 2 models are provided.

    Example:
        >>> models = [
        ...     ("xgboost_breast_cancer", xgb_model, xgb_meta),
        ...     ("rf_breast_cancer", rf_model, rf_meta),
        ... ]
        >>> comparison = compute_prediction_comparison(models, X_test, 42)
        >>> comparison.agreement
        True
    """
    if len(models) < 2:
        raise ValueError(
            f"compare_predictions requires at least 2 models, got {len(models)}."
        )

    if sample_index < 0 or sample_index >= len(X):
        raise IndexError(
            f"Sample index {sample_index} is out of range. "
            f"Dataset has {len(X)} samples (valid indices: 0\u2013{len(X) - 1})."
        )

    per_model: list[SingleModelPrediction] = []

    for model_id, model, metadata in models:
        target_names = metadata.get("target_names", ["class_0", "class_1"])
        shap_result = compute_shap_values(
            model=model,
            X=X,
            sample_index=sample_index,
            target_names=target_names,
            background_data=background_data,
        )
        top_features = _extract_top_features(shap_result, top_n=top_n)

        per_model.append(SingleModelPrediction(
            model_id=model_id,
            model_type=metadata.get("model_type", "unknown"),
            predicted_class=shap_result.prediction,
            predicted_label=shap_result.prediction_label,
            probability=shap_result.probability,
            top_features=top_features,
        ))

    # Agreement analysis
    agreement = per_model[0].predicted_class == per_model[1].predicted_class
    confidence_gap = round(abs(per_model[0].probability - per_model[1].probability), 4)

    # Shared vs divergent drivers
    sets = {
        entry.model_id: {f["name"] for f in entry.top_features}
        for entry in per_model
    }
    all_sets = list(sets.values())
    shared = sorted(all_sets[0] & all_sets[1])
    divergent = {
        mid: sorted(s - all_sets[1 - i])
        for i, (mid, s) in enumerate(sets.items())
    }

    return PredictionComparison(
        per_model=per_model,
        agreement=agreement,
        confidence_gap=confidence_gap,
        shared_top_features=shared,
        divergent_features=divergent,
    )


# ---------------------------------------------------------------------------
# Pipeline bridge: read pre-computed artifacts from Kedro explainability
# pipeline (xai-xgboost-clf repo). This is the PRODUCTION path.
# ---------------------------------------------------------------------------


def _load_pipeline_metadata(artifacts_dir: Path) -> dict:
    """Read and validate the pipeline's shap_metadata.json.

    Args:
        artifacts_dir: Directory containing pipeline output artifacts.

    Returns:
        Parsed metadata dictionary.

    Raises:
        FileNotFoundError: If shap_metadata.json doesn't exist.
    """
    meta_path = artifacts_dir / "shap_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Pipeline metadata not found: {meta_path}. "
            f"Expected shap_metadata.json in '{artifacts_dir}'. "
            f"Has the Kedro explainability pipeline been run?"
        )
    with open(meta_path) as f:
        return json.load(f)


def _load_pipeline_shap_values(artifacts_dir: Path, metadata: dict) -> np.ndarray:
    """Load the SHAP values array from pipeline artifacts.

    Supports only the current single-output contract:
    one 2D `.npy` file with shape `(n_samples, n_features)`.

    Args:
        artifacts_dir: Directory containing pipeline output artifacts.
        metadata: Parsed shap_metadata.json.

    Returns:
        numpy array of shape (n_samples, n_features).

    Raises:
        FileNotFoundError: If the SHAP values file doesn't exist.
    """
    feature_names = _require_pipeline_feature_names(metadata)
    shap_paths = metadata.get("shap_saved_paths", ["shap_values.npy"])
    if not isinstance(shap_paths, list) or not shap_paths:
        raise ValueError(
            "Pipeline metadata 'shap_saved_paths' must be a non-empty list."
        )
    if len(shap_paths) != 1:
        raise ValueError(
            "Only single-output SHAP artifacts are supported. "
            f"Expected 1 path in 'shap_saved_paths', got {len(shap_paths)}."
        )
    shap_filename = shap_paths[0]  # primary file
    shap_path = artifacts_dir / shap_filename

    if not shap_path.exists():
        raise FileNotFoundError(
            f"SHAP values file not found: {shap_path}. "
            f"Expected '{shap_filename}' in '{artifacts_dir}'."
        )
    shap_values = np.load(shap_path)
    if shap_values.ndim != 2:
        raise ValueError(
            "Only 2D SHAP arrays with shape (n_samples, n_features) are supported."
        )
    if shap_values.shape[1] != len(feature_names):
        raise ValueError(
            "SHAP array width does not match pipeline metadata feature_names."
        )
    return shap_values


def load_shap_from_pipeline(
    artifacts_dir: str | Path,
    sample_index: int,
) -> ShapResult:
    """Read a single sample's SHAP explanation from pipeline artifacts.

    This is the PRODUCTION bridge between the Kedro explainability pipeline
    and our MCP toolkit. Instead of recomputing SHAP on the fly, we read
    the pre-computed values that the pipeline already persisted to disk.

    The Kedro pipeline (explainability_node) saves:
      - shap_values.npy: SHAP values for all explained samples
      - shap_expected_value.npy: baseline (expected) values
      - shap_metadata.json: feature names, explainer type, etc.

    This function extracts one sample's worth of data and returns it as
    a ShapResult, compatible with our narrators and plot functions.

    Note: prediction and probability are NOT available from SHAP artifacts
    alone (they require the model). These fields are set to placeholder
    values. If you need them, pass the model separately or use the
    on-the-fly compute_shap_values() function instead.

    Args:
        artifacts_dir: Path to directory containing pipeline outputs.
        sample_index: Which sample (row) to extract from the bulk results.

    Returns:
        ShapResult populated from the pipeline's pre-computed artifacts.

    Raises:
        FileNotFoundError: If required artifact files are missing.
        IndexError: If sample_index is beyond the number of explained samples.

    Example:
        >>> result = load_shap_from_pipeline("shap_results/", sample_index=42)
        >>> narrative = narrate_prediction(result, top_n=3)
    """
    artifacts_dir = Path(artifacts_dir)

    # Load metadata and SHAP values
    metadata = _load_pipeline_metadata(artifacts_dir)
    shap_values = _load_pipeline_shap_values(artifacts_dir, metadata)
    feature_names = _require_pipeline_feature_names(metadata)

    # Validate sample index
    n_samples = shap_values.shape[0]
    if sample_index < 0 or sample_index >= n_samples:
        raise IndexError(
            f"Sample index {sample_index} is out of range. "
            f"Pipeline computed SHAP for {n_samples} samples "
            f"(valid indices: 0\u2013{n_samples - 1})."
        )

    # Extract this sample's SHAP values
    sample_shap = shap_values[sample_index]  # shape: (n_features,)
    shap_dict = {
        name: round(float(val), 6)
        for name, val in zip(feature_names, sample_shap)
    }

    # Load base value (expected value)
    ev_filename = metadata.get("expected_value_saved", "shap_expected_value.npy")
    ev_path = artifacts_dir / ev_filename
    if ev_path.exists():
        expected_values = np.load(ev_path)
        if expected_values.ndim == 0:
            base_value = float(expected_values)
        elif expected_values.ndim == 1 and len(expected_values) == n_samples:
            base_value = float(expected_values[sample_index])
        elif expected_values.ndim == 1 and len(expected_values) == 1:
            base_value = float(expected_values[0])
        else:
            raise ValueError(
                "Only scalar or per-sample 1D expected value arrays are supported."
            )
    elif "expected_value_saved" in metadata:
        raise FileNotFoundError(
            f"Expected value file not found: {ev_path}."
        )
    else:
        base_value = 0.0

    # Feature values are not stored in standard pipeline artifacts.
    # Use zeros as placeholder — callers who need actual values should
    # join with the original test data.
    feature_value_dict = {name: 0.0 for name in feature_names}

    return ShapResult(
        prediction=0,  # unknown from SHAP artifacts alone
        prediction_label="unknown",
        probability=0.0,
        base_value=round(base_value, 6),
        shap_values=shap_dict,
        feature_values=feature_value_dict,
        feature_names=feature_names,
    )


def load_global_importance_from_pipeline(
    artifacts_dir: str | Path,
) -> list[FeatureImportance]:
    """Compute global feature importance from pipeline's pre-computed SHAP.

    Reads the bulk SHAP values that the Kedro pipeline already computed
    and derives mean absolute importance per feature — the same metric
    our on-the-fly compute_global_feature_importance() produces.

    This avoids recomputing SHAP across all samples, which is the most
    expensive operation in the explainability workflow.

    Args:
        artifacts_dir: Path to directory containing pipeline outputs.

    Returns:
        List of FeatureImportance, sorted by importance (highest first).
        Compatible with narrate_feature_comparison() and narrate_model_summary().

    Raises:
        FileNotFoundError: If required artifact files are missing.

    Example:
        >>> importances = load_global_importance_from_pipeline("shap_results/")
        >>> narrative = narrate_feature_comparison(importances, top_n=10)
    """
    artifacts_dir = Path(artifacts_dir)

    # Load metadata and SHAP values
    metadata = _load_pipeline_metadata(artifacts_dir)
    shap_values = _load_pipeline_shap_values(artifacts_dir, metadata)
    feature_names = _require_pipeline_feature_names(metadata)

    # Same aggregation as compute_global_feature_importance:
    # importance = mean(|SHAP|), direction = sign(mean(SHAP))
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    mean_signed = np.mean(shap_values, axis=0)

    importances = [
        FeatureImportance(
            name=name,
            importance=round(float(mean_abs[i]), 6),
            direction="positive" if mean_signed[i] > 0 else "negative",
            mean_shap=round(float(mean_signed[i]), 6),
        )
        for i, name in enumerate(feature_names)
    ]

    # Sort by importance (highest first)
    importances.sort(key=lambda f: f.importance, reverse=True)
    return importances
