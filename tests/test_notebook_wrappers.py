"""Tests for notebook-first wrapper helpers."""

from __future__ import annotations

import pytest

from xai_toolkit import notebook_wrappers as nw


def _fake_tool_response(**metadata_overrides):
    metadata = {
        "model_id": "xgboost_breast_cancer",
        "provenance": "precomputed",
        "data_source": "result_store",
        "run_id": "run-001",
        "resolved_run_strategy": "latest_successful_by_computed_at",
        "applied_skills": [],
    }
    metadata.update(metadata_overrides)
    return {
        "narrative": "ok",
        "evidence": {},
        "metadata": metadata,
        "grounded": True,
    }


def test_explain_sample_returns_trust_envelope(monkeypatch):
    monkeypatch.setattr(nw.server_module.registry, "list_models", lambda: [1])
    monkeypatch.setattr(
        nw.server_module,
        "explain_prediction",
        lambda **_: _fake_tool_response(),
    )

    result = nw.explain_sample(model_id="xgboost_breast_cancer", sample_index=42)
    assert result["tool"] == "explain_prediction"
    assert result["trust"]["provenance"] == "precomputed"
    assert result["trust"]["run_id"] == "run-001"


def test_drift_summary_falls_back_to_detect_drift(monkeypatch):
    monkeypatch.setattr(nw.server_module.registry, "list_models", lambda: [1])
    monkeypatch.setattr(
        nw.server_module,
        "list_drift_alerts",
        lambda **_: {"narrative": "none", "evidence": {"alerts": []}, "metadata": {}},
    )
    monkeypatch.setattr(
        nw.server_module,
        "detect_drift",
        lambda **_: _fake_tool_response(provenance="on_the_fly", data_source="model_registry"),
    )

    result = nw.drift_summary(model_id="xgboost_breast_cancer")
    assert result["tool"] == "detect_drift"
    assert result["trust"]["provenance"] == "on_the_fly"


def test_ask_xai_auto_routes_to_explain_sample(monkeypatch):
    monkeypatch.setattr(nw.server_module.registry, "list_models", lambda: [1])
    monkeypatch.setattr(
        nw.server_module,
        "explain_prediction",
        lambda **_: _fake_tool_response(),
    )

    result = nw.ask_xai(
        question="Why sample 42?",
        intent="auto",
        model_id="xgboost_breast_cancer",
        sample_index=42,
    )
    assert result["resolved_intent"] == "explain_sample"
    assert result["tool"] == "explain_prediction"


def test_ask_xai_requires_model_for_drift():
    with pytest.raises(ValueError, match="drift_summary intent requires model_id"):
        nw.ask_xai(question="show drift", intent="drift_summary")
