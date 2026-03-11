"""MCP server — thin adapter layer (ADR-001).

This is the ONLY file that imports MCP/FastMCP. It wires together:
  - registry.py (model loading)
  - explainers.py (SHAP computation + data hashing)
  - narrators.py (English generation)
  - plots.py (visualization)
  - schemas.py (data contracts)

The server itself does NO computation or narrative generation.
It is an adapter: receive request → call pure functions → return response.
"""

import logging
import re
from pathlib import Path

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

from xai_toolkit.instructions import get_glass_floor_principles, get_methodology_content
from xai_toolkit.explainers import (
    compute_data_hash,
    compute_dataset_description,
    compute_global_feature_importance,
    compute_model_summary,
    compute_partial_dependence,
    compute_prediction_comparison,
    compute_shap_values,
    extract_intrinsic_importances,
)
from xai_toolkit.drift import (
    detect_drift as compute_dataset_drift,
    detect_feature_drift as compute_feature_drift,
)
from xai_toolkit.narrators import (
    narrate_dataset,
    narrate_dataset_drift,
    narrate_feature_comparison,
    narrate_feature_drift,
    narrate_intrinsic_importance,
    narrate_model_summary,
    narrate_partial_dependence,
    narrate_prediction,
    narrate_prediction_comparison,
)
from xai_toolkit.knowledge import load_knowledge_base, search_chunks
from xai_toolkit.plots import plot_pdp_ice, plot_shap_bar, plot_shap_waterfall
from xai_toolkit.registry import ModelRegistry, RegisteredModel
from xai_toolkit.schemas import (
    ErrorCode,
    ErrorResponse,
    KnowledgeChunk,
    KnowledgeSearchResult,
    ToolMetadata,
    ToolResponse,
)

# --- Server setup ---

mcp = FastMCP(
    "xai-toolkit",
    instructions=(
        "ML model explainability server. Provides plain-English explanations "
        "of model predictions backed by SHAP analysis. Call tools to get "
        "deterministic, reproducible explanations — do not interpret SHAP "
        "values yourself. The LLM is the presenter, not the analyst. "
        "Embed the 'narrative' field naturally in your response; "
        "it is the authoritative explanation and must not be supplemented or "
        "reinterpreted. When a response includes 'plot_base64', render it as "
        "an inline image. "
        "IMPORTANT — epistemic labeling: every tool response includes "
        "'grounded': true, meaning the answer was computed deterministically "
        "from a registered model and is audit-ready. If you answer a question "
        "WITHOUT calling any tool, you MUST begin your response with exactly: "
        "'\u26a0\ufe0f This answer is based on my general knowledge, not computed "
        "from your registered models. It has not been verified against your data "
        "and should not be used for audit or governance purposes.' "
        "If no tool matches the question, it is better to say 'I don't know' "
        "than to guess. "
        "GLASS FLOOR PROTOCOL (ADR-009): When a user asks 'what should I do' "
        "or asks for recommendations after an explanation, call "
        "retrieve_business_context with a query derived from the explanation "
        "context (e.g., feature names, risk level, probability). Then present "
        "the response in TWO clearly separated layers: "
        "LAYER 1 (Deterministic): Present the tool narrative FIRST, exactly as "
        "returned. This is computed, reproducible, and audit-ready. "
        "LAYER 2 (Business Context): Present retrieved business context AFTER "
        "the deterministic layer. Always prefix with: "
        "'📋 Business Context (AI-interpreted from [source_document], "
        "section: [section_heading]):' and include the provenance_label value. "
        "NEVER mix Layer 1 and Layer 2 content. The deterministic explanation "
        "must stand alone and unmodified. Business context is ADDITIVE only."
    ),
)

# --- Model registry initialization ---

ROOT = Path(__file__).parent.parent.parent  # xai-mcp/
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
KNOWLEDGE_DIR = ROOT / "knowledge"

registry = ModelRegistry()
_knowledge_base: "KnowledgeBase | None" = None


def init_server(
    models_dir: Path = MODELS_DIR,
    data_dir: Path = DATA_DIR,
    knowledge_dir: Path = KNOWLEDGE_DIR,
) -> None:
    """Initialize models and knowledge base. Call before running server."""
    global _knowledge_base
    for model_id in ("xgboost_breast_cancer", "rf_breast_cancer"):
        try:
            registry.load_from_disk(model_id, models_dir, data_dir)
            logger.info("Loaded model: %s", model_id)
        except FileNotFoundError:
            logger.warning("Could not load model '%s' at startup.", model_id)
    _knowledge_base = load_knowledge_base(knowledge_dir)


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
    error_code: ErrorCode,
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


def _require_model(model_id: str) -> RegisteredModel | dict:
    """Look up a model by ID, returning a structured error dict on failure.

    Design pattern: Extract Method. Every tool function needs this same
    lookup-or-error logic. Centralizing it ensures consistent error shape
    and eliminates ~8 lines of duplication per tool.

    Returns:
        RegisteredModel on success, or a dict (ErrorResponse) on failure.
        Callers check ``isinstance(result, dict)`` to detect errors.
    """
    try:
        return registry.get(model_id)
    except KeyError:
        available = [m["model_id"] for m in registry.list_models()]
        return _build_error(
            error_code=ErrorCode.MODEL_NOT_FOUND,
            message=(
                f"Model '{model_id}' is not registered. "
                f"Available models: {available}."
            ),
            available=available,
        )


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
    entry = _require_model(model_id)
    if isinstance(entry, dict):
        return entry

    try:
        shap_result = compute_shap_values(
            model=entry.model,
            X=entry.X_test,
            sample_index=sample_index,
            target_names=entry.metadata.get("target_names"),
            background_data=entry.X_train,
        )
    except IndexError as e:
        return _build_error(
            error_code=ErrorCode.SAMPLE_OUT_OF_RANGE,
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
        detected_type=entry.metadata.get("detected_type"),
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
    entry = _require_model(model_id)
    if isinstance(entry, dict):
        return entry

    try:
        shap_result = compute_shap_values(
            model=entry.model,
            X=entry.X_test,
            sample_index=sample_index,
            target_names=entry.metadata.get("target_names"),
            background_data=entry.X_train,
        )
    except IndexError as e:
        return _build_error(
            error_code=ErrorCode.SAMPLE_OUT_OF_RANGE,
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
        detected_type=entry.metadata.get("detected_type"),
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
    entry = _require_model(model_id)
    if isinstance(entry, dict):
        return entry

    summary = compute_model_summary(
        model=entry.model,
        X=entry.X_test,
        metadata=entry.metadata,
        top_n=5,
        background_data=entry.X_train,
    )
    narrative = narrate_model_summary(summary)
    data_hash = compute_data_hash(entry.X_test)

    evidence = summary.model_dump()

    # Intrinsic importances (adapted from Tamas's _handle_intrinsically_explainable_model)
    intrinsic_result = extract_intrinsic_importances(
        model=entry.model,
        feature_names=list(entry.X_test.columns),
    )
    if intrinsic_result is not None:
        intrinsic, source_attr = intrinsic_result
        intrinsic_narrative = narrate_intrinsic_importance(
            importances=intrinsic,
            model_type=entry.metadata.get("model_type", "unknown"),
            n_features_total=len(entry.X_test.columns),
            source_attr=source_attr,
        )
        narrative += "\n\n[Intrinsic Interpretability] " + intrinsic_narrative
        evidence["intrinsic_importances"] = [f.model_dump() for f in intrinsic[:10]]

    return _build_response(
        narrative=narrative,
        evidence=evidence,
        model_id=model_id,
        model_type=entry.metadata.get("model_type", "unknown"),
        detected_type=entry.metadata.get("detected_type"),
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
    entry = _require_model(model_id)
    if isinstance(entry, dict):
        return entry

    importances = compute_global_feature_importance(
        model=entry.model,
        X=entry.X_test,
        target_names=entry.metadata.get("target_names"),
        background_data=entry.X_train,
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
        detected_type=entry.metadata.get("detected_type"),
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
    entry = _require_model(model_id)
    if isinstance(entry, dict):
        return entry

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
            error_code=ErrorCode.FEATURE_NOT_FOUND,
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
        detected_type=entry.metadata.get("detected_type"),
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
    entry = _require_model(model_id)
    if isinstance(entry, dict):
        return entry

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
        detected_type=entry.metadata.get("detected_type"),
        dataset_size=len(entry.X_test),
        data_hash=data_hash,
    )


@mcp.tool()
def compare_predictions(
    model_id_1: str,
    model_id_2: str,
    sample_index: int,
) -> dict:
    """Compare what two models predict for the same sample and explain why.

    Returns whether the models agree, their confidence levels, and which
    features they share or diverge on — all in plain English. Use this
    to build trust in predictions by checking cross-model consistency.

    Args:
        model_id_1: ID of the first model (e.g., "xgboost_breast_cancer").
        model_id_2: ID of the second model (e.g., "rf_breast_cancer").
        sample_index: Row index in the test dataset to compare (0-based).
    """
    # Validate both models exist
    models_data = []
    for mid in (model_id_1, model_id_2):
        result = _require_model(mid)
        if isinstance(result, dict):
            return result
        models_data.append((mid, result))

    # Use first model's test data (both share the same dataset)
    entry_1 = models_data[0][1]

    try:
        comparison = compute_prediction_comparison(
            models=[
                (mid, entry.model, entry.metadata)
                for mid, entry in models_data
            ],
            X=entry_1.X_test,
            sample_index=sample_index,
            background_data=entry_1.X_train,
        )
    except IndexError as e:
        return _build_error(
            error_code=ErrorCode.SAMPLE_OUT_OF_RANGE,
            message=str(e),
            available=[f"0\u2013{len(entry_1.X_test) - 1}"],
        )

    narrative = narrate_prediction_comparison(comparison)
    data_hash = compute_data_hash(entry_1.X_test, sample_index=sample_index)

    return _build_response(
        narrative=narrative,
        evidence=comparison.model_dump(),
        model_id=f"{model_id_1} vs {model_id_2}",
        model_type="comparison",
        sample_index=sample_index,
        dataset_size=len(entry_1.X_test),
        data_hash=data_hash,
    )


# --- Drift detection tools ---


@mcp.tool()
def detect_drift(model_id: str) -> dict:
    """Detect data drift between a model's training data and test data.

    Compares the reference (training) distribution against the current
    (test) distribution for every feature. Returns an aggregate summary
    of which features drifted, by how much, and overall severity.

    Numeric features are tested with PSI (primary) and KS (supporting).
    Categorical features are tested with chi-squared.

    Args:
        model_id: ID of a registered model (e.g., "xgboost_breast_cancer").
    """
    entry = _require_model(model_id)
    if isinstance(entry, dict):
        return entry

    if entry.X_train is None:
        return _build_error(
            error_code=ErrorCode.UNKNOWN_ERROR,
            message=(
                f"Model '{model_id}' has no training data (X_train). "
                "Drift detection requires both reference and current data."
            ),
        )

    drift_result = compute_dataset_drift(
        reference=entry.X_train,
        current=entry.X_test,
    )
    narrative = narrate_dataset_drift(drift_result)

    return _build_response(
        narrative=narrative,
        evidence=drift_result.model_dump(),
        model_id=model_id,
        model_type=entry.metadata.get("model_type", "unknown"),
        dataset_size=len(entry.X_test),
    )


@mcp.tool()
def detect_feature_drift(model_id: str, feature_name: str) -> dict:
    """Detect drift for a single feature between training and test data.

    Returns detailed drift analysis for one feature: statistical test
    results, severity, how the distribution shifted (direction, magnitude),
    and reference vs. current distribution summaries.

    Args:
        model_id: ID of a registered model (e.g., "xgboost_breast_cancer").
        feature_name: Name of the feature to analyze (e.g., "mean radius").
    """
    entry = _require_model(model_id)
    if isinstance(entry, dict):
        return entry

    if entry.X_train is None:
        return _build_error(
            error_code=ErrorCode.UNKNOWN_ERROR,
            message=(
                f"Model '{model_id}' has no training data (X_train). "
                "Drift detection requires both reference and current data."
            ),
        )

    if feature_name not in entry.X_train.columns:
        from difflib import get_close_matches

        available = list(entry.X_train.columns)
        close = get_close_matches(feature_name, available, n=3, cutoff=0.4)
        return _build_error(
            error_code=ErrorCode.FEATURE_NOT_FOUND,
            message=(
                f"Feature '{feature_name}' not found. "
                f"Did you mean: {close}? "
                f"Available features: {available}"
            ),
            available=available,
            suggestion=close[0] if close else None,
        )

    drift_result = compute_feature_drift(
        reference=entry.X_train[feature_name],
        current=entry.X_test[feature_name],
        feature_name=feature_name,
    )
    narrative = narrate_feature_drift(drift_result)

    return _build_response(
        narrative=narrative,
        evidence=drift_result.model_dump(),
        model_id=model_id,
        model_type=entry.metadata.get("model_type", "unknown"),
        dataset_size=len(entry.X_test),
    )


# --- Knowledge / RAG tool (ADR-009) ---


@mcp.tool()
def retrieve_business_context(
    query: str,
    top_k: int = 5,
) -> dict:
    """Retrieve relevant business context from the knowledge base.

    Searches loaded business documents (e.g., clinical protocols, operational
    rules) for sections relevant to the query. Returns ranked chunks with
    source provenance for the Glass Floor presentation pattern.

    Use this AFTER an explainability tool to find actionable business guidance.
    For example, after explain_prediction returns a high-probability malignant
    classification, call this with a query like 'high risk malignant biopsy'
    to retrieve the relevant clinical protocol sections.

    The provenance_label is always 'ai-interpreted' — any synthesis from
    these chunks by the LLM is NOT deterministic and must be clearly
    distinguished from grounded tool outputs.

    Args:
        query: Natural language search query (e.g., 'high risk biopsy referral').
        top_k: Maximum number of chunks to return (default: 5).
    """
    if _knowledge_base is None or not _knowledge_base.chunks:
        return KnowledgeSearchResult(
            chunks=[],
            query=query,
            documents_searched=[],
            retrieval_method="tfidf",
        ).model_dump()

    results = search_chunks(_knowledge_base, query, top_k=top_k)

    chunks = [
        KnowledgeChunk(
            text=chunk.text,
            source_document=chunk.source_document,
            document_id=chunk.document_id,
            section_heading=chunk.section_heading,
            chunk_index=chunk.chunk_index,
            relevance_score=round(score, 4),
        )
        for chunk, score in results
    ]

    return KnowledgeSearchResult(
        chunks=chunks,
        query=query,
        documents_searched=_knowledge_base.document_names,
        retrieval_method="tfidf",
    ).model_dump()


# --- Instruction-serving tools ---


@mcp.tool()
def get_xai_methodology() -> str:
    """Retrieve the XAI analysis methodology guide.

    Call this BEFORE starting any model explanation to understand the correct
    tool sequence (explanation funnel), Glass Floor protocol, and anti-patterns
    to avoid. Returns the full workflow guide.
    """
    try:
        return get_methodology_content(ROOT)
    except FileNotFoundError:
        return _build_error(
            error_code=ErrorCode.UNKNOWN_ERROR,
            message=(
                "SKILL.md not found at "
                f"{ROOT / 'skills' / 'xai-workflow' / 'SKILL.md'}. "
                "Ensure the skills/xai-workflow/ directory exists."
            ),
            suggestion="Check that the repository is complete and SKILL.md has not been removed.",
        )


@mcp.tool()
def get_glass_floor() -> str:
    """Retrieve the Glass Floor separation principles for presenting model explanations alongside business context.

    Call this when you need to present both deterministic model outputs and
    AI-interpreted business guidance. Returns the two-layer separation protocol.
    """
    return get_glass_floor_principles()


# --- MCP Prompts ---


@mcp.prompt()
def xai_methodology() -> str:
    """The XAI analysis methodology — explanation funnel, Glass Floor, anti-patterns."""
    return get_methodology_content(ROOT)


# --- Entrypoint ---


def main() -> None:
    """Entry point for console_scripts and Databricks App deployment."""
    import os

    init_server()

    transport = os.environ.get("XAI_TRANSPORT", "stdio")
    if transport == "http":
        mcp.settings.host = "0.0.0.0"
        mcp.settings.port = int(os.environ.get("XAI_PORT", "8000"))
        mcp.settings.stateless_http = True
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
