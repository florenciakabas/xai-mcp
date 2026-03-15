"""Kedro adapter node functions — thin wrappers around pure computation (Layer 4).

Each node:
  - Accepts DataFrames + model + params from the Kedro catalog
  - Calls pure functions from explainers.py, drift.py, narrators.py
  - Returns pd.DataFrame matching result store schemas
  - Handles Spark DF → pandas coercion at the boundary with guardrails
"""

import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from xai_toolkit.drift import detect_drift as compute_dataset_drift
from xai_toolkit.explainers import (
    compute_data_hash,
    compute_global_feature_importance,
    compute_shap_values_batch,
    _extract_top_features,
)
from xai_toolkit.narrators import (
    narrate_dataset_drift,
    narrate_feature_drift,
    narrate_model_summary,
    narrate_prediction,
)
from xai_toolkit.registry import ModelRegistry
from xai_toolkit.result_store import (
    StoredDriftResult,
    StoredExplanation,
    StoredModelSummary,
)
from xai_toolkit.sampling import select_samples

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Spark → pandas guardrails
# ---------------------------------------------------------------------------

_WARN_THRESHOLD = 100_000
_REFUSE_THRESHOLD = 1_000_000


def _coerce_to_pandas(df: object, name: str = "DataFrame") -> pd.DataFrame:
    """Coerce a DataFrame to pandas, with size guardrails for Spark.

    Args:
        df: A pandas DataFrame or a Spark DataFrame.
        name: Human-readable name for log messages.

    Returns:
        A pandas DataFrame.

    Raises:
        ValueError: If a Spark DataFrame exceeds the refuse threshold.
        TypeError: If df is neither pandas nor Spark.
    """
    if isinstance(df, pd.DataFrame):
        return df

    # Check for Spark DataFrame (lazy import to avoid hard dependency)
    try:
        from pyspark.sql import DataFrame as SparkDataFrame
    except ImportError:
        SparkDataFrame = None

    if SparkDataFrame is not None and isinstance(df, SparkDataFrame):
        row_count = df.count()
        logger.info("%s has %d rows (Spark DataFrame).", name, row_count)

        if row_count > _REFUSE_THRESHOLD:
            raise ValueError(
                f"{name} has {row_count:,} rows, exceeding the limit of "
                f"{_REFUSE_THRESHOLD:,}. Sample or project before passing "
                f"to the XAI pipeline."
            )
        if row_count > _WARN_THRESHOLD:
            logger.warning(
                "%s has %d rows — this may be slow to convert to pandas.",
                name, row_count,
            )
        return df.toPandas()

    raise TypeError(
        f"{name} must be a pandas or Spark DataFrame, got {type(df).__name__}."
    )


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


def sample_indices_node(
    model: object,
    X: object,
    params: dict,
) -> list[int]:
    """Select which rows to explain.

    Params:
        n_samples: Number of samples (default: 100).
        strategy: "random" or "uncertainty" (default: "random").
        random_state: Seed (default: 42).
    """
    X_pd = _coerce_to_pandas(X, "X (for sampling)")

    # Validate model
    ModelRegistry._validate_supported_model(model, "pipeline_model")

    return select_samples(
        model=model,
        X=X_pd,
        n_samples=params.get("n_samples", 100),
        strategy=params.get("strategy", "random"),
        random_state=params.get("random_state", 42),
    )


def batch_explain_node(
    model: object,
    X_test: object,
    y_test: object,
    X_train: object | None,
    sample_indices: list[int],
    params: dict,
) -> pd.DataFrame:
    """Compute SHAP + narration for sampled rows → DataFrame.

    Returns a DataFrame matching StoredExplanation schema.
    """
    X_test_pd = _coerce_to_pandas(X_test, "X_test")
    y_test_pd = _coerce_to_pandas(y_test, "y_test") if not isinstance(y_test, pd.Series) else y_test
    X_train_pd = _coerce_to_pandas(X_train, "X_train") if X_train is not None else None

    if isinstance(y_test_pd, pd.DataFrame):
        y_test_pd = y_test_pd.squeeze()

    model_id = params.get("model_id", "pipeline_model")
    run_id = params.get("run_id", f"kedro-{datetime.now(timezone.utc).isoformat()}")
    target_names = params.get("target_names", ["class_0", "class_1"])
    computed_at = datetime.now(timezone.utc).isoformat()

    ModelRegistry._validate_supported_model(model, model_id)

    shap_results = compute_shap_values_batch(
        model=model,
        X=X_test_pd,
        sample_indices=sample_indices,
        target_names=target_names,
        background_data=X_train_pd,
    )
    if len(shap_results) != len(sample_indices):
        raise RuntimeError(
            "Batch SHAP returned fewer results than requested sample indices. "
            "Aborting to avoid misaligned sample-to-explanation mapping."
        )

    rows = []
    for shap_result, idx in zip(shap_results, sample_indices):
        top_features = _extract_top_features(shap_result, top_n=5)
        narrative = narrate_prediction(shap_result, top_n=3)
        data_hash = compute_data_hash(X_test_pd, sample_index=idx)

        rows.append(StoredExplanation(
            run_id=run_id,
            model_id=model_id,
            sample_index=idx,
            prediction=shap_result.prediction,
            prediction_label=shap_result.prediction_label,
            probability=shap_result.probability,
            narrative=narrative,
            top_features=json.dumps(top_features),
            shap_values=json.dumps(shap_result.shap_values),
            feature_values=json.dumps(shap_result.feature_values),
            data_hash=data_hash,
            computed_at=computed_at,
        ).model_dump())

    logger.info(
        "batch_explain: produced %d explanations for model '%s' (run=%s)",
        len(rows), model_id, run_id,
    )
    return pd.DataFrame(rows)


def detect_drift_node(
    X_train: object,
    X_scoring: object,
    params: dict,
) -> pd.DataFrame:
    """Drift detection + narration → DataFrame matching StoredDriftResult schema."""
    X_train_pd = _coerce_to_pandas(X_train, "X_train (reference)")
    X_scoring_pd = _coerce_to_pandas(X_scoring, "X_scoring (current)")

    model_id = params.get("model_id", "pipeline_model")
    run_id = params.get("run_id", f"kedro-{datetime.now(timezone.utc).isoformat()}")
    computed_at = datetime.now(timezone.utc).isoformat()

    drift_result = compute_dataset_drift(
        reference=X_train_pd,
        current=X_scoring_pd,
    )
    overall_narrative = narrate_dataset_drift(drift_result)

    rows = []
    for feature_result in drift_result.features:
        per_feature_narrative = narrate_feature_drift(feature_result)
        rows.append(StoredDriftResult(
            run_id=run_id,
            model_id=model_id,
            feature_name=feature_result.feature_name,
            test_name=feature_result.test_name,
            statistic=feature_result.statistic,
            p_value=feature_result.p_value,
            drift_detected=feature_result.drift_detected,
            severity=feature_result.severity,
            narrative=per_feature_narrative,
            overall_narrative=overall_narrative,
            overall_severity=drift_result.overall_severity,
            computed_at=computed_at,
        ).model_dump())

    logger.info(
        "detect_drift: %d features analyzed for model '%s', %d drifted (run=%s)",
        len(rows), model_id, drift_result.n_drifted, run_id,
    )
    return pd.DataFrame(rows)


def model_summary_node(
    model: object,
    X_test: object,
    metadata: dict,
    params: dict,
) -> pd.DataFrame:
    """Global feature importance → DataFrame matching StoredModelSummary schema."""
    X_test_pd = _coerce_to_pandas(X_test, "X_test (for summary)")

    model_id = params.get("model_id", "pipeline_model")
    run_id = params.get("run_id", f"kedro-{datetime.now(timezone.utc).isoformat()}")
    computed_at = datetime.now(timezone.utc).isoformat()

    ModelRegistry._validate_supported_model(model, model_id)

    target_names = metadata.get("target_names", ["class_0", "class_1"])
    importances = compute_global_feature_importance(
        model=model,
        X=X_test_pd,
        target_names=target_names,
    )

    # Build a ModelSummary for the narrative
    from xai_toolkit.schemas import ModelSummary
    summary = ModelSummary(
        model_type=metadata.get("model_type", "unknown"),
        accuracy=metadata.get("accuracy", 0.0),
        n_features=len(X_test_pd.columns),
        n_train_samples=metadata.get("n_train_samples", 0),
        n_test_samples=metadata.get("n_test_samples", len(X_test_pd)),
        target_names=target_names,
        top_features=importances[:5],
    )
    narrative = narrate_model_summary(summary)

    rows = []
    for rank, feat in enumerate(importances, start=1):
        rows.append(StoredModelSummary(
            run_id=run_id,
            model_id=model_id,
            feature_name=feat.name,
            importance=feat.importance,
            rank=rank,
            narrative=narrative,
            model_type=metadata.get("model_type", "unknown"),
            computed_at=computed_at,
        ).model_dump())

    logger.info(
        "model_summary: %d features ranked for model '%s' (run=%s)",
        len(rows), model_id, run_id,
    )
    return pd.DataFrame(rows)
