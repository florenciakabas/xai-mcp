"""Notebook-first wrappers for structured XAI interactions.

These helpers are intentionally thin. They route to existing server tools and
return consistent envelopes that include trust metadata for auditability.
"""

from __future__ import annotations

from typing import Literal

from xai_toolkit import server as server_module

NotebookIntent = Literal[
    "auto",
    "explain_sample",
    "drift_summary",
    "standard_briefing",
    "summarize_model",
    "list_models",
]


def _ensure_initialized() -> None:
    """Initialize server once when wrappers are used in notebooks."""
    if not server_module.registry.list_models():
        server_module.init_server()


def _extract_trust(tool_response: dict) -> dict:
    metadata = tool_response.get("metadata", {})
    return {
        "provenance": metadata.get("provenance", "unknown"),
        "data_source": metadata.get("data_source", "unknown"),
        "run_id": metadata.get("run_id"),
        "model_id": metadata.get("model_id"),
        "resolved_run_strategy": metadata.get(
            "resolved_run_strategy", "not_applicable"
        ),
        "applied_skills": metadata.get("applied_skills", []),
    }


def explain_sample(
    *,
    model_id: str,
    sample_index: int,
    include_plot: bool = False,
) -> dict:
    """Explain one sample using retrieval-first semantics."""
    _ensure_initialized()
    response = server_module.explain_prediction(
        model_id=model_id, sample_index=sample_index, include_plot=include_plot
    )
    return {
        "intent": "explain_sample",
        "tool": "explain_prediction",
        "response": response,
        "trust": _extract_trust(response),
    }


def drift_summary(
    *,
    model_id: str,
    run_id: str | None = None,
    severity: str | None = None,
) -> dict:
    """Return drift summary with retrieval-first preference."""
    _ensure_initialized()
    response = server_module.list_drift_alerts(
        model_id=model_id,
        severity=severity,
        run_id=run_id,
    )
    alerts = response.get("evidence", {}).get("alerts", [])
    if not alerts and run_id is None and severity is None:
        response = server_module.detect_drift(model_id=model_id)
        tool_name = "detect_drift"
    else:
        tool_name = "list_drift_alerts"
    return {
        "intent": "drift_summary",
        "tool": tool_name,
        "response": response,
        "trust": _extract_trust(response),
    }


def ask_xai(
    *,
    question: str,
    intent: NotebookIntent = "auto",
    model_id: str | None = None,
    sample_index: int | None = None,
    run_id: str | None = None,
    top_cases: int = 5,
    include_plot: bool = False,
) -> dict:
    """Single notebook entrypoint with simple intent routing."""
    _ensure_initialized()

    resolved_intent = intent
    if intent == "auto":
        q = question.lower()
        if sample_index is not None:
            resolved_intent = "explain_sample"
        elif "drift" in q:
            resolved_intent = "drift_summary"
        elif "brief" in q or "latest run" in q:
            resolved_intent = "standard_briefing"
        elif "summarize" in q or "summary" in q:
            resolved_intent = "summarize_model"
        else:
            resolved_intent = "list_models"

    if resolved_intent == "explain_sample":
        if model_id is None or sample_index is None:
            raise ValueError(
                "explain_sample intent requires model_id and sample_index."
            )
        wrapped = explain_sample(
            model_id=model_id,
            sample_index=sample_index,
            include_plot=include_plot,
        )
    elif resolved_intent == "drift_summary":
        if model_id is None:
            raise ValueError("drift_summary intent requires model_id.")
        wrapped = drift_summary(model_id=model_id, run_id=run_id, severity=None)
    elif resolved_intent == "standard_briefing":
        response = server_module.standard_briefing(
            model_id=model_id, run_id=run_id, top_cases=top_cases
        )
        wrapped = {
            "intent": "standard_briefing",
            "tool": "standard_briefing",
            "response": response,
            "trust": _extract_trust(response),
        }
    elif resolved_intent == "summarize_model":
        if model_id is None:
            raise ValueError("summarize_model intent requires model_id.")
        response = server_module.summarize_model(model_id=model_id)
        wrapped = {
            "intent": "summarize_model",
            "tool": "summarize_model",
            "response": response,
            "trust": _extract_trust(response),
        }
    elif resolved_intent == "list_models":
        response = server_module.list_models()
        wrapped = {
            "intent": "list_models",
            "tool": "list_models",
            "response": response,
            "trust": _extract_trust(response),
        }
    else:
        raise ValueError(f"Unsupported intent: {resolved_intent}")

    wrapped["question"] = question
    wrapped["resolved_intent"] = resolved_intent
    return wrapped
