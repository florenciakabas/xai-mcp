"""MCP server — thin adapter layer (ADR-001).

This is the ONLY file that imports MCP/FastMCP. It wires together:
  - registry.py (model loading)
  - explainers.py (SHAP computation + data hashing)
  - narrators.py (English generation)
  - plots.py (visualization)
  - schemas.py (data contracts)

The server itself does NO computation or narrative generation.
It is an adapter: receive request → call pure functions → return response.

Day 3 additions:
  - data_hash in every ToolMetadata (audit trail, D3-S2)
  - Structured ErrorResponse for all failure modes (D3-S3, S4, S5)
"""

import re
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from xai_toolkit.explainers import (
    compute_data_hash,
    compute_dataset_description,
    compute_global_feature_importance,
    compute_model_summary,
    compute_partial_dependence,
    compute_shap_values,
)
from xai_toolkit.narrators import (
    narrate_dataset,
    narrate_feature_comparison,
    narrate_model_summary,
    narrate_partial_dependence,
    narrate_prediction,
)
from xai_toolkit.plots import plot_pdp_ice, plot_shap_bar, plot_shap_waterfall
from xai_toolkit.registry import ModelRegistry
from xai_toolkit.schemas import ErrorResponse, ToolMetadata, ToolResponse

# --- Server setup ---

mcp = FastMCP(
    "xai-toolkit",
    instructions=(
        "ML model explainability server. Provides plain-English explanations "
        "of model predictions backed by SHAP analysis. Call tools to get "
        "deterministic, reproducible explanations — do not interpret SHAP "
        "values yourself. Embed the 'narrative' field naturally in your response; "
        "it is the authoritative explanation and must not be supplemented or "
        "reinterpreted. When a response includes 'plot_base64', render it as "
        "an inline image. "
        "IMPORTANT — epistemic labeling: every tool response includes "
        "'grounded': true, meaning the answer was computed deterministically "
        "from a registered model and is audit-ready. If you answer a question "
        "WITHOUT calling any tool, you MUST begin your response with exactly: "
        "'\u26a0\ufe0f This answer is based on my general knowledge, not computed "
        "from your registered models. It has not been verified against your data "
        "and should not be used for audit or governance purposes.'"
    ),
)

# --- Model registry initialization ---

ROOT = Path(__file__).parent.parent.parent  # xai-mcp/
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

registry = ModelRegistry()

for _model_id in ("xgboost_breast_cancer", "rf_breast_cancer"):
    try:
        registry.load_from_disk(_model_id, MODELS_DIR, DATA_DIR)
    except FileNotFoundError as e:
        print(f"Warning: Could not load {_model_id} at startup: {e}")


# --- Response builders ---


def _build_response(
    narrative: str,
    evidence: dict,
    model_id: str,
    model_type: str = "unknown",
    plot_base64: str | None = None,
    **extra_metadata,
) -> dict:
    """Build a ToolResponse dict with consistent structure (ADR-005)."""
    return ToolResponse(
        narrative=narrative,
        evidence=evidence,
        metadata=ToolMetadata(
            model_id=model_id,
            model_type=model_type,
            **extra_metadata,
        ),
        plot_base64=plot_base64,
    ).model_dump()


def _build_error(
    error_code: str,
    message: str,
    available: list[str] | None = None,
    suggestion: str | None = None,
) -> dict:
    """Build a structured ErrorResponse dict (D3-S3, S4, S5).

    Returns a consistent shape every time so the LLM always knows what
    to expect, regardless of whether the tool call succeeded or failed.
    """
    return ErrorResponse(
        error_code=error_code,
        message=message,
        available=available or [],
        suggestion=suggestion,
    ).model_dump()


def _extract_suggestion(error_message: str) -> str | None:
    """Extract the closest-match suggestion from a ValueError message.

    compute_partial_dependence() raises ValueError with the format:
      "Feature 'X' not found. Did you mean: ['suggestion']? ..."

    We extract the first suggestion so ErrorResponse can surface it
    in a dedicated field rather than buried in the message string.

    Args:
        error_message: The full error message string.

    Returns:
        The first suggestion string, or None if none found.
    """
    match = re.search(r"Did you mean: \['([^']+)'", error_message)
    if match:
        return match.group(1)
    return None


# --- MCP Tools ---


@mcp.tool()
def explain_prediction(
    model_id: str,
    sample_index: int,
    include_plot: bool = True,
) -> dict:
    """Explain why a single sample received its classification.

    Returns a plain-English narrative explaining which features drove
    the model's prediction for the given sample, backed by SHAP values.
    Optionally includes a SHAP bar chart (tornado plot) visualization.

    Args:
        model_id: ID of a registered model (e.g., "xgboost_breast_cancer").
        sample_index: Row index in the test dataset to explain (0-based).
        include_plot: If True, include a SHAP bar chart as base64 PNG (default: True).
    """
    try:
        entry = registry.get(model_id)
    except KeyError:
        available = [m["model_id"] for m in registry.list_models()]
        return _build_error(
            error_code="MODEL_NOT_FOUND",
            message=(
                f"Model '{model_id}' is not registered. "
                f"Available models: {available}."
            ),
            available=available,
        )

    try:
        shap_result = compute_shap_values(
            model=entry.model,
            X=entry.X_test,
            sample_index=sample_index,
            target_names=entry.metadata.get("target_names"),
        )
    except IndexError as e:
        return _build_error(
            error_code="SAMPLE_OUT_OF_RANGE",
            message=str(e),
            available=[f"0\u2013{len(entry.X_test) - 1}"],
        )

    narrative = narrate_prediction(shap_result, top_n=3)
    data_hash = compute_data_hash(entry.X_test, sample_index=sample_index)

    plot_b64 = None
    if include_plot:
        plot_b64 = plot_shap_bar(shap_result, top_n=10)

    return _build_response(
        narrative=narrative,
        evidence=shap_result.model_dump(),
        model_id=model_id,
        model_type=entry.metadata.get("model_type", "unknown"),
        sample_index=sample_index,
        dataset_size=len(entry.X_test),
        data_hash=data_hash,
        plot_base64=plot_b64,
    )


@mcp.tool()
def explain_prediction_waterfall(
    model_id: str,
    sample_index: int,
) -> dict:
    """Show a SHAP waterfall plot for a single prediction.

    The waterfall shows how the base prediction builds up to the final
    prediction feature by feature. This is the most detailed SHAP visualization.

    Args:
        model_id: ID of a registered model (e.g., "xgboost_breast_cancer").
        sample_index: Row index in the test dataset to explain (0-based).
    """
    try:
        entry = registry.get(model_id)
    except KeyError:
        available = [m["model_id"] for m in registry.list_models()]
        return _build_error(
            error_code="MODEL_NOT_FOUND",
            message=(
                f"Model '{model_id}' is not registered. "
                f"Available models: {available}."
            ),
            available=available,
        )

    try:
        shap_result = compute_shap_values(
            model=entry.model,
            X=entry.X_test,
            sample_index=sample_index,
            target_names=entry.metadata.get("target_names"),
        )
    except IndexError as e:
        return _build_error(
            error_code="SAMPLE_OUT_OF_RANGE",
            message=str(e),
            available=[f"0\u2013{len(entry.X_test) - 1}"],
        )

    narrative = narrate_prediction(shap_result, top_n=3)
    data_hash = compute_data_hash(entry.X_test, sample_index=sample_index)
    plot_b64 = plot_shap_waterfall(shap_result, top_n=10)

    return _build_response(
        narrative=narrative,
        evidence=shap_result.model_dump(),
        model_id=model_id,
        model_type=entry.metadata.get("model_type", "unknown"),
        sample_index=sample_index,
        dataset_size=len(entry.X_test),
        data_hash=data_hash,
        plot_base64=plot_b64,
    )


@mcp.tool()
def summarize_model(model_id: str) -> dict:
    """Summarize what a model does and what drives its decisions.

    Returns model type, accuracy, number of features, and the top features
    ranked by importance — all in plain English.

    Args:
        model_id: ID of a registered model (e.g., "xgboost_breast_cancer").
    """
    try:
        entry = registry.get(model_id)
    except KeyError:
        available = [m["model_id"] for m in registry.list_models()]
        return _build_error(
            error_code="MODEL_NOT_FOUND",
            message=(
                f"Model '{model_id}' is not registered. "
                f"Available models: {available}."
            ),
            available=available,
        )

    summary = compute_model_summary(
        model=entry.model,
        X=entry.X_test,
        metadata=entry.metadata,
        top_n=5,
    )
    narrative = narrate_model_summary(summary)
    data_hash = compute_data_hash(entry.X_test)

    return _build_response(
        narrative=narrative,
        evidence=summary.model_dump(),
        model_id=model_id,
        model_type=entry.metadata.get("model_type", "unknown"),
        dataset_size=len(entry.X_test),
        data_hash=data_hash,
    )


@mcp.tool()
def compare_features(model_id: str, top_n: int = 10) -> dict:
    """Rank features by importance and describe which matter most.

    Returns a ranked list of features with their magnitude, direction,
    and comparative language — all in plain English.

    Args:
        model_id: ID of a registered model (e.g., "xgboost_breast_cancer").
        top_n: Number of top features to include (default: 10).
    """
    try:
        entry = registry.get(model_id)
    except KeyError:
        available = [m["model_id"] for m in registry.list_models()]
        return _build_error(
            error_code="MODEL_NOT_FOUND",
            message=(
                f"Model '{model_id}' is not registered. "
                f"Available models: {available}."
            ),
            available=available,
        )

    importances = compute_global_feature_importance(
        model=entry.model,
        X=entry.X_test,
        target_names=entry.metadata.get("target_names"),
    )
    narrative = narrate_feature_comparison(importances, top_n=top_n)
    data_hash = compute_data_hash(entry.X_test)

    evidence = {
        "features": [feat.model_dump() for feat in importances[:top_n]],
        "total_features": len(importances),
    }

    return _build_response(
        narrative=narrative,
        evidence=evidence,
        model_id=model_id,
        model_type=entry.metadata.get("model_type", "unknown"),
        dataset_size=len(entry.X_test),
        data_hash=data_hash,
    )


@mcp.tool()
def get_partial_dependence(
    model_id: str,
    feature_name: str,
    include_plot: bool = True,
) -> dict:
    """Show how a single feature affects predictions across its range.

    Returns a narrative describing the relationship between the feature
    and the model's predicted probability. Optionally includes a PDP + ICE
    plot (model-agnostic visualization, not SHAP-based).

    PDP (bold line) shows the average effect. ICE (gray lines) show individual
    sample effects, revealing heterogeneity the average hides.

    Args:
        model_id: ID of a registered model (e.g., "xgboost_breast_cancer").
        feature_name: Name of the feature to analyze (e.g., "mean radius").
        include_plot: If True, include a PDP+ICE plot as base64 PNG (default: True).
    """
    try:
        entry = registry.get(model_id)
    except KeyError:
        available = [m["model_id"] for m in registry.list_models()]
        return _build_error(
            error_code="MODEL_NOT_FOUND",
            message=(
                f"Model '{model_id}' is not registered. "
                f"Available models: {available}."
            ),
            available=available,
        )

    try:
        pdp_result = compute_partial_dependence(
            model=entry.model,
            X=entry.X_test,
            feature_name=feature_name,
        )
    except ValueError as e:
        error_str = str(e)
        suggestion = _extract_suggestion(error_str)
        return _build_error(
            error_code="FEATURE_NOT_FOUND",
            message=error_str,
            available=list(entry.X_test.columns),
            suggestion=suggestion,
        )

    narrative = narrate_partial_dependence(pdp_result)
    data_hash = compute_data_hash(entry.X_test)

    plot_b64 = None
    if include_plot:
        plot_b64 = plot_pdp_ice(pdp_result)

    evidence = pdp_result.model_dump()
    evidence.pop("ice_curves", None)

    return _build_response(
        narrative=narrative,
        evidence=evidence,
        model_id=model_id,
        model_type=entry.metadata.get("model_type", "unknown"),
        dataset_size=len(entry.X_test),
        data_hash=data_hash,
        plot_base64=plot_b64,
    )


@mcp.tool()
def list_models() -> dict:
    """List all registered models with their metadata.

    Returns model IDs, types, dataset names, feature counts, and accuracy.
    Use this to discover what models are available before asking questions.
    """
    models = registry.list_models()

    if not models:
        narrative = "No models are currently registered."
    elif len(models) == 1:
        m = models[0]
        narrative = (
            f"There is 1 model available: {m['model_id']} "
            f"({m['model_type']}, {m['feature_count']} features, "
            f"accuracy: {m['accuracy']:.1%})."
        )
    else:
        parts = [
            f"{m['model_id']} ({m['model_type']}, {m['feature_count']} features, "
            f"accuracy: {m['accuracy']:.1%})"
            for m in models
        ]
        narrative = (
            f"There are {len(models)} models available: "
            + "; ".join(parts) + "."
        )

    return ToolResponse(
        narrative=narrative,
        evidence={"models": models},
        metadata=ToolMetadata(
            model_id="registry",
            model_type="registry",
        ),
    ).model_dump()


@mcp.tool()
def describe_dataset(model_id: str) -> dict:
    """Describe the dataset associated with a model.

    Returns number of samples, features, class distribution, missing values,
    and basic statistics — all in plain English.

    Args:
        model_id: ID of a registered model (e.g., "xgboost_breast_cancer").
    """
    try:
        entry = registry.get(model_id)
    except KeyError:
        available = [m["model_id"] for m in registry.list_models()]
        return _build_error(
            error_code="MODEL_NOT_FOUND",
            message=(
                f"Model '{model_id}' is not registered. "
                f"Available models: {available}."
            ),
            available=available,
        )

    description = compute_dataset_description(
        X=entry.X_test,
        y=entry.y_test,
        target_names=entry.metadata.get("target_names"),
    )
    narrative = narrate_dataset(description)
    data_hash = compute_data_hash(entry.X_test)

    return _build_response(
        narrative=narrative,
        evidence=description.model_dump(),
        model_id=model_id,
        model_type=entry.metadata.get("model_type", "unknown"),
        dataset_size=len(entry.X_test),
        data_hash=data_hash,
    )


# --- Entrypoint ---

if __name__ == "__main__":
    mcp.run(transport="stdio")
