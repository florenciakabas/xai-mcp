"""CLI adapter — same pure functions, no MCP dependency (ADR-001).

This module provides a command-line interface to the XAI toolkit.
It follows the identical architectural pattern as server.py:

    cli.py (adapter) → explainers.py / narrators.py / knowledge.py (pure functions)

The CLI exists for two reasons:
    1. Environments where MCP servers are unavailable or restricted.
    2. Scripting, CI/CD pipelines, and programmatic access without an LLM.

All output is JSON to stdout, making it composable with jq, scripts, etc.

Usage:
    uv run python -m xai_toolkit.cli models
    uv run python -m xai_toolkit.cli explain --model xgboost_breast_cancer --sample 0
    uv run python -m xai_toolkit.cli context --query "high risk biopsy"

Design decisions:
    - argparse over click/typer: zero extra dependencies.
    - JSON output: machine-readable, composable, matches MCP tool output shape.
    - Same response schemas: a ToolResponse from CLI is identical to one from MCP.
    - No LLM involvement: the CLI outputs the deterministic narrative directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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
from xai_toolkit.knowledge import load_knowledge_base, search_chunks
from xai_toolkit.narrators import (
    narrate_dataset,
    narrate_feature_comparison,
    narrate_intrinsic_importance,
    narrate_model_summary,
    narrate_partial_dependence,
    narrate_prediction,
    narrate_prediction_comparison,
)
from xai_toolkit.plots import plot_pdp_ice, plot_shap_bar, plot_shap_waterfall
from xai_toolkit.registry import ModelRegistry
from xai_toolkit.schemas import (
    ErrorResponse,
    KnowledgeChunk,
    KnowledgeSearchResult,
    ToolMetadata,
    ToolResponse,
)


# ---------------------------------------------------------------------------
# Shared setup (mirrors server.py startup)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent.parent  # xai-mcp/
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
KNOWLEDGE_DIR = ROOT / "knowledge"


def _output(data: dict) -> None:
    """Write a dict as formatted JSON to stdout.

    Centralised output so every command produces consistent, parseable JSON.
    """
    json.dump(data, sys.stdout, indent=2, default=str)
    print()  # trailing newline


def _build_response(
    narrative: str,
    evidence: dict,
    model_id: str,
    model_type: str = "unknown",
    plot_base64: str | None = None,
    **extra_metadata: object,
) -> dict:
    """Build a ToolResponse dict — identical shape to server.py (ADR-005)."""
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
    """Build an ErrorResponse dict — identical shape to server.py."""
    return ErrorResponse(
        error_code=error_code,
        message=message,
        available=available or [],
        suggestion=suggestion,
    ).model_dump()


# ---------------------------------------------------------------------------
# Command handlers — one per subcommand
# ---------------------------------------------------------------------------


def cmd_models(args: argparse.Namespace, registry: ModelRegistry) -> None:
    """List all registered models."""
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
        narrative = f"There are {len(models)} models available: " + "; ".join(parts) + "."

    _output(ToolResponse(
        narrative=narrative,
        evidence={"models": models},
        metadata=ToolMetadata(model_id="registry", model_type="registry"),
    ).model_dump())


def cmd_explain(args: argparse.Namespace, registry: ModelRegistry) -> None:
    """Explain a single prediction."""
    try:
        entry = registry.get(args.model)
    except KeyError:
        available = [m["model_id"] for m in registry.list_models()]
        _output(_build_error("MODEL_NOT_FOUND", f"Model '{args.model}' not registered.", available))
        return

    try:
        shap_result = compute_shap_values(
            model=entry.model, X=entry.X_test, sample_index=args.sample,
            target_names=entry.metadata.get("target_names"),
            background_data=entry.X_train,
        )
    except IndexError as e:
        _output(_build_error("SAMPLE_OUT_OF_RANGE", str(e), [f"0–{len(entry.X_test) - 1}"]))
        return

    narrative = narrate_prediction(shap_result, top_n=3)
    data_hash = compute_data_hash(entry.X_test, sample_index=args.sample)
    plot_b64 = None if args.no_plot else plot_shap_bar(shap_result, top_n=10)

    _output(_build_response(
        narrative=narrative, evidence=shap_result.model_dump(),
        model_id=args.model, model_type=entry.metadata.get("model_type", "unknown"),
        detected_type=entry.metadata.get("detected_type"),
        sample_index=args.sample, dataset_size=len(entry.X_test),
        data_hash=data_hash, plot_base64=plot_b64,
    ))


def cmd_waterfall(args: argparse.Namespace, registry: ModelRegistry) -> None:
    """Show SHAP waterfall for a prediction."""
    try:
        entry = registry.get(args.model)
    except KeyError:
        available = [m["model_id"] for m in registry.list_models()]
        _output(_build_error("MODEL_NOT_FOUND", f"Model '{args.model}' not registered.", available))
        return

    try:
        shap_result = compute_shap_values(
            model=entry.model, X=entry.X_test, sample_index=args.sample,
            target_names=entry.metadata.get("target_names"),
            background_data=entry.X_train,
        )
    except IndexError as e:
        _output(_build_error("SAMPLE_OUT_OF_RANGE", str(e), [f"0–{len(entry.X_test) - 1}"]))
        return

    narrative = narrate_prediction(shap_result, top_n=3)
    data_hash = compute_data_hash(entry.X_test, sample_index=args.sample)
    plot_b64 = plot_shap_waterfall(shap_result, top_n=10)

    _output(_build_response(
        narrative=narrative, evidence=shap_result.model_dump(),
        model_id=args.model, model_type=entry.metadata.get("model_type", "unknown"),
        detected_type=entry.metadata.get("detected_type"),
        sample_index=args.sample, dataset_size=len(entry.X_test),
        data_hash=data_hash, plot_base64=plot_b64,
    ))


def cmd_summarize(args: argparse.Namespace, registry: ModelRegistry) -> None:
    """Summarize a model."""
    try:
        entry = registry.get(args.model)
    except KeyError:
        available = [m["model_id"] for m in registry.list_models()]
        _output(_build_error("MODEL_NOT_FOUND", f"Model '{args.model}' not registered.", available))
        return

    summary = compute_model_summary(
        model=entry.model, X=entry.X_test, metadata=entry.metadata, top_n=5,
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

    _output(_build_response(
        narrative=narrative, evidence=evidence,
        model_id=args.model, model_type=entry.metadata.get("model_type", "unknown"),
        detected_type=entry.metadata.get("detected_type"),
        dataset_size=len(entry.X_test), data_hash=data_hash,
    ))


def cmd_features(args: argparse.Namespace, registry: ModelRegistry) -> None:
    """Rank features by importance."""
    try:
        entry = registry.get(args.model)
    except KeyError:
        available = [m["model_id"] for m in registry.list_models()]
        _output(_build_error("MODEL_NOT_FOUND", f"Model '{args.model}' not registered.", available))
        return

    importances = compute_global_feature_importance(
        model=entry.model, X=entry.X_test,
        target_names=entry.metadata.get("target_names"),
        background_data=entry.X_train,
    )
    narrative = narrate_feature_comparison(importances, top_n=args.top_n)
    data_hash = compute_data_hash(entry.X_test)

    evidence = {
        "features": [feat.model_dump() for feat in importances[:args.top_n]],
        "total_features": len(importances),
    }
    _output(_build_response(
        narrative=narrative, evidence=evidence,
        model_id=args.model, model_type=entry.metadata.get("model_type", "unknown"),
        detected_type=entry.metadata.get("detected_type"),
        dataset_size=len(entry.X_test), data_hash=data_hash,
    ))


def cmd_pdp(args: argparse.Namespace, registry: ModelRegistry) -> None:
    """Partial dependence for a feature."""
    try:
        entry = registry.get(args.model)
    except KeyError:
        available = [m["model_id"] for m in registry.list_models()]
        _output(_build_error("MODEL_NOT_FOUND", f"Model '{args.model}' not registered.", available))
        return

    try:
        pdp_result = compute_partial_dependence(
            model=entry.model, X=entry.X_test, feature_name=args.feature,
        )
    except ValueError as e:
        _output(_build_error("FEATURE_NOT_FOUND", str(e), list(entry.X_test.columns)))
        return

    narrative = narrate_partial_dependence(pdp_result)
    data_hash = compute_data_hash(entry.X_test)
    plot_b64 = None if args.no_plot else plot_pdp_ice(pdp_result)

    evidence = pdp_result.model_dump()
    evidence.pop("ice_curves", None)

    _output(_build_response(
        narrative=narrative, evidence=evidence,
        model_id=args.model, model_type=entry.metadata.get("model_type", "unknown"),
        detected_type=entry.metadata.get("detected_type"),
        dataset_size=len(entry.X_test), data_hash=data_hash,
        plot_base64=plot_b64,
    ))


def cmd_dataset(args: argparse.Namespace, registry: ModelRegistry) -> None:
    """Describe a model's dataset."""
    try:
        entry = registry.get(args.model)
    except KeyError:
        available = [m["model_id"] for m in registry.list_models()]
        _output(_build_error("MODEL_NOT_FOUND", f"Model '{args.model}' not registered.", available))
        return

    description = compute_dataset_description(
        X=entry.X_test, y=entry.y_test,
        target_names=entry.metadata.get("target_names"),
    )
    narrative = narrate_dataset(description)
    data_hash = compute_data_hash(entry.X_test)

    _output(_build_response(
        narrative=narrative, evidence=description.model_dump(),
        model_id=args.model, model_type=entry.metadata.get("model_type", "unknown"),
        detected_type=entry.metadata.get("detected_type"),
        dataset_size=len(entry.X_test), data_hash=data_hash,
    ))


def cmd_compare(args: argparse.Namespace, registry: ModelRegistry) -> None:
    """Compare predictions from two models on the same sample."""
    for mid in (args.model1, args.model2):
        try:
            registry.get(mid)
        except KeyError:
            available = [m["model_id"] for m in registry.list_models()]
            _output(_build_error("MODEL_NOT_FOUND", f"Model '{mid}' not registered.", available))
            return

    entry_1 = registry.get(args.model1)

    try:
        comparison = compute_prediction_comparison(
            models=[
                (args.model1, registry.get(args.model1).model, registry.get(args.model1).metadata),
                (args.model2, registry.get(args.model2).model, registry.get(args.model2).metadata),
            ],
            X=entry_1.X_test,
            sample_index=args.sample,
            background_data=entry_1.X_train,
        )
    except IndexError as e:
        _output(_build_error("SAMPLE_OUT_OF_RANGE", str(e), [f"0\u2013{len(entry_1.X_test) - 1}"]))
        return

    narrative = narrate_prediction_comparison(comparison)
    data_hash = compute_data_hash(entry_1.X_test, sample_index=args.sample)

    _output(_build_response(
        narrative=narrative, evidence=comparison.model_dump(),
        model_id=f"{args.model1} vs {args.model2}", model_type="comparison",
        sample_index=args.sample, dataset_size=len(entry_1.X_test),
        data_hash=data_hash,
    ))


def cmd_context(args: argparse.Namespace, kb: object) -> None:
    """Retrieve business context (RAG search).

    Args:
        args: Parsed CLI arguments with query and top_k.
        kb: Loaded KnowledgeBase instance.
    """
    if not kb.chunks:
        _output(KnowledgeSearchResult(
            chunks=[], query=args.query,
            documents_searched=[], retrieval_method="tfidf",
        ).model_dump())
        return

    results = search_chunks(kb, args.query, top_k=args.top_k)

    chunks = [
        KnowledgeChunk(
            text=chunk.text, source_document=chunk.source_document,
            document_id=chunk.document_id, section_heading=chunk.section_heading,
            chunk_index=chunk.chunk_index, relevance_score=round(score, 4),
        )
        for chunk, score in results
    ]

    _output(KnowledgeSearchResult(
        chunks=chunks, query=args.query,
        documents_searched=kb.document_names, retrieval_method="tfidf",
    ).model_dump())


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with subcommands.

    Each subcommand maps to one MCP tool, keeping the mental model
    consistent across both interfaces.

    Returns:
        Configured ArgumentParser ready for parse_args().
    """
    parser = argparse.ArgumentParser(
        prog="xai-toolkit",
        description="ML Explainability Toolkit — CLI interface",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # models (no args needed)
    sub.add_parser("models", help="List registered models")

    # explain
    p = sub.add_parser("explain", help="Explain a single prediction")
    p.add_argument("--model", required=True, help="Model ID")
    p.add_argument("--sample", required=True, type=int, help="Sample index")
    p.add_argument("--no-plot", action="store_true", help="Skip plot generation")

    # waterfall
    p = sub.add_parser("waterfall", help="SHAP waterfall for a prediction")
    p.add_argument("--model", required=True, help="Model ID")
    p.add_argument("--sample", required=True, type=int, help="Sample index")

    # summarize
    p = sub.add_parser("summarize", help="Summarize a model")
    p.add_argument("--model", required=True, help="Model ID")

    # features
    p = sub.add_parser("features", help="Rank features by importance")
    p.add_argument("--model", required=True, help="Model ID")
    p.add_argument("--top-n", type=int, default=10, help="Number of features")

    # pdp
    p = sub.add_parser("pdp", help="Partial dependence for a feature")
    p.add_argument("--model", required=True, help="Model ID")
    p.add_argument("--feature", required=True, help="Feature name")
    p.add_argument("--no-plot", action="store_true", help="Skip plot generation")

    # dataset
    p = sub.add_parser("dataset", help="Describe a model's dataset")
    p.add_argument("--model", required=True, help="Model ID")

    # compare
    p = sub.add_parser("compare", help="Compare predictions from two models")
    p.add_argument("--model1", required=True, help="First model ID")
    p.add_argument("--model2", required=True, help="Second model ID")
    p.add_argument("--sample", required=True, type=int, help="Sample index")

    # context (RAG)
    p = sub.add_parser("context", help="Retrieve business context")
    p.add_argument("--query", required=True, help="Search query")
    p.add_argument("--top-k", type=int, default=5, help="Max chunks to return")

    return parser


# ---------------------------------------------------------------------------
# Dispatch table and entrypoint
# ---------------------------------------------------------------------------

# Commands that need the model registry
_REGISTRY_COMMANDS = {
    "models": cmd_models,
    "explain": cmd_explain,
    "waterfall": cmd_waterfall,
    "summarize": cmd_summarize,
    "features": cmd_features,
    "pdp": cmd_pdp,
    "dataset": cmd_dataset,
    "compare": cmd_compare,
}


def main() -> None:
    """CLI entrypoint. Parse args, load resources, dispatch command.

    Resource loading is lazy: registry is only populated if a model command
    is called, knowledge base only loaded for 'context' command.
    """
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "context":
        kb = load_knowledge_base(KNOWLEDGE_DIR)
        cmd_context(args, kb)
    elif args.command in _REGISTRY_COMMANDS:
        registry = ModelRegistry()
        for model_id in ("xgboost_breast_cancer", "rf_breast_cancer"):
            try:
                registry.load_from_disk(model_id, MODELS_DIR, DATA_DIR)
            except FileNotFoundError:
                pass  # graceful — model simply won't be listed
        _REGISTRY_COMMANDS[args.command](args, registry)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
