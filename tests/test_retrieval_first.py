"""Tests for retrieval-first MCP tool behavior (Scenario Group 5)."""

import json

import pytest

from xai_toolkit.result_store import (
    StoredDriftResult,
    StoredExplanation,
    save_drift_results,
    save_explanations,
)
from xai_toolkit.server import (
    _store_path,
    _try_load_stored_drift,
    _try_load_stored_explanation,
    detect_drift,
    explain_prediction,
    init_server,
    list_drift_alerts,
    list_explained_samples,
    registry,
    standard_briefing,
)
import xai_toolkit.server as server_module


def _make_stored_explanation(
    model_id="xgboost_breast_cancer",
    sample_index=42,
    run_id="batch-2026-03-13",
) -> StoredExplanation:
    return StoredExplanation(
        run_id=run_id,
        model_id=model_id,
        sample_index=sample_index,
        prediction=1,
        prediction_label="malignant",
        probability=0.92,
        narrative="Precomputed: Sample 42 classified as malignant.",
        top_features=json.dumps([{"name": "worst_radius", "shap_value": 0.4}]),
        shap_values=json.dumps({"worst_radius": 0.4, "mean_texture": -0.1}),
        feature_values=json.dumps({"worst_radius": 25.0, "mean_texture": 15.0}),
        data_hash="b" * 64,
        computed_at="2026-03-13T14:00:00Z",
    )


def _make_stored_drift(
    model_id="xgboost_breast_cancer",
    feature_name="worst_radius",
    severity="severe",
    run_id="batch-2026-03-13",
    computed_at="2026-03-13T14:00:00Z",
) -> StoredDriftResult:
    return StoredDriftResult(
        run_id=run_id,
        model_id=model_id,
        feature_name=feature_name,
        test_name="psi",
        statistic=0.35,
        p_value=0.001,
        drift_detected=True,
        severity=severity,
        narrative=f"{feature_name} shows {severity} drift.",
        overall_narrative="Severe drift detected in dataset.",
        overall_severity="severe",
        computed_at=computed_at,
    )


class TestRetrievalFirstExplain:
    """S5.1–S5.2 — explain_prediction checks store first."""

    def test_returns_precomputed_when_available(self, tmp_path):
        """S5.1 — Precomputed result returned with source='precomputed'."""
        # Set up store with precomputed result
        stored = _make_stored_explanation(sample_index=42)
        save_explanations([stored], tmp_path)

        # Point server at this store
        old_store = server_module._store_path
        server_module._store_path = tmp_path
        try:
            result = explain_prediction("xgboost_breast_cancer", 42, include_plot=False)
            assert result["metadata"]["source"] == "precomputed"
            assert result["metadata"]["batch_run_id"] == "batch-2026-03-13"
            assert result["metadata"]["run_id"] == "batch-2026-03-13"
            assert result["metadata"]["resolved_run_strategy"] == "latest_successful_by_computed_at"
            assert result["metadata"]["provenance"] == "precomputed"
            assert result["metadata"]["data_source"] == "result_store"
            assert "Precomputed" in result["narrative"]
        finally:
            server_module._store_path = old_store

    def test_falls_back_to_on_the_fly(self, tmp_path):
        """S5.2 — Falls back to on-the-fly when not in store."""
        # Empty store
        old_store = server_module._store_path
        server_module._store_path = tmp_path
        try:
            result = explain_prediction("xgboost_breast_cancer", 0, include_plot=False)
            assert result["metadata"]["source"] == "on_the_fly"
            assert result["metadata"].get("batch_run_id") is None
            assert result["metadata"]["provenance"] == "on_the_fly"
            assert result["metadata"]["resolved_run_strategy"] == "not_applicable"
        finally:
            server_module._store_path = old_store

    def test_malformed_precomputed_payload_falls_back_to_on_the_fly(self, tmp_path):
        """Malformed stored JSON should not crash explain_prediction."""
        bad = _make_stored_explanation(sample_index=7)
        bad = bad.model_copy(update={"shap_values": "{not-json"})
        save_explanations([bad], tmp_path)

        old_store = server_module._store_path
        server_module._store_path = tmp_path
        try:
            result = explain_prediction("xgboost_breast_cancer", 7, include_plot=False)
            assert result["metadata"]["source"] == "on_the_fly"
            assert result["metadata"].get("batch_run_id") is None
        finally:
            server_module._store_path = old_store


class TestRetrievalFirstDrift:
    """S5.3–S5.4 — detect_drift checks store first."""

    def test_returns_precomputed_drift(self, tmp_path):
        """S5.3 — Precomputed drift returned with source='precomputed'."""
        stored = [
            _make_stored_drift(feature_name="f1", severity="severe"),
            _make_stored_drift(feature_name="f2", severity="moderate"),
        ]
        save_drift_results(stored, tmp_path)

        old_store = server_module._store_path
        server_module._store_path = tmp_path
        try:
            result = detect_drift("xgboost_breast_cancer")
            assert result["metadata"]["source"] == "precomputed"
            assert result["metadata"]["batch_run_id"] == "batch-2026-03-13"
            assert result["metadata"]["run_id"] == "batch-2026-03-13"
            assert result["metadata"]["resolved_run_strategy"] == "latest_successful_by_computed_at"
        finally:
            server_module._store_path = old_store

    def test_falls_back_to_on_the_fly_drift(self, tmp_path):
        """S5.4 — Falls back to on-the-fly when not in store."""
        old_store = server_module._store_path
        server_module._store_path = tmp_path
        try:
            result = detect_drift("xgboost_breast_cancer")
            assert result["metadata"]["source"] == "on_the_fly"
        finally:
            server_module._store_path = old_store


class TestDiscoveryTools:
    """S5.5–S5.9 — Discovery tools."""

    def test_list_drift_alerts_filter_severity(self, tmp_path):
        """S5.5 — Filter by severity."""
        stored = [
            _make_stored_drift(feature_name="f1", severity="severe"),
            _make_stored_drift(feature_name="f2", severity="severe"),
            _make_stored_drift(feature_name="f3", severity="moderate"),
        ]
        save_drift_results(stored, tmp_path)

        old_store = server_module._store_path
        server_module._store_path = tmp_path
        try:
            result = list_drift_alerts(
                model_id="xgboost_breast_cancer",
                severity="severe",
            )
            alerts = result["evidence"]["alerts"]
            assert len(alerts) == 2
            assert all(a["severity"] == "severe" for a in alerts)
        finally:
            server_module._store_path = old_store

    def test_list_drift_alerts_respects_run_id_filter(self, tmp_path):
        """Run filter should fetch requested run, not only latest."""
        old_run = _make_stored_drift(
            feature_name="f_old",
            run_id="run-old",
            severity="severe",
        )
        new_run = _make_stored_drift(
            feature_name="f_new",
            run_id="run-new",
            severity="moderate",
            # Ensure this sorts as latest.
            computed_at="2026-03-14T14:00:00Z",
        )
        save_drift_results([old_run], tmp_path)
        save_drift_results([new_run], tmp_path)

        old_store = server_module._store_path
        server_module._store_path = tmp_path
        try:
            result = list_drift_alerts(
                model_id="xgboost_breast_cancer",
                run_id="run-old",
            )
            alerts = result["evidence"]["alerts"]
            assert len(alerts) == 1
            assert alerts[0]["run_id"] == "run-old"
            assert alerts[0]["feature_name"] == "f_old"
            assert result["metadata"]["resolved_run_strategy"] == "explicit_run_id"
            assert result["metadata"]["run_id"] == "run-old"
        finally:
            server_module._store_path = old_store

    def test_list_explained_samples(self, tmp_path):
        """S5.6 — Shows available precomputed explanations."""
        stored = [
            _make_stored_explanation(sample_index=i)
            for i in [0, 5, 12, 42, 99]
        ]
        save_explanations(stored, tmp_path)

        old_store = server_module._store_path
        server_module._store_path = tmp_path
        try:
            result = list_explained_samples("xgboost_breast_cancer")
            samples = result["evidence"]["samples"]
            assert len(samples) == 5
            indices = [s["sample_index"] for s in samples]
            assert set(indices) == {0, 5, 12, 42, 99}
        finally:
            server_module._store_path = old_store

    def test_list_drift_alerts_empty(self, tmp_path):
        """S5.9 — No batch results returns empty list."""
        old_store = server_module._store_path
        server_module._store_path = tmp_path
        try:
            result = list_drift_alerts(model_id="xgboost_breast_cancer")
            assert result["evidence"]["alerts"] == []
            assert "No batch drift results" in result["narrative"]
        finally:
            server_module._store_path = old_store

    def test_list_explained_samples_empty(self, tmp_path):
        """No precomputed explanations returns empty list."""
        old_store = server_module._store_path
        server_module._store_path = tmp_path
        try:
            result = list_explained_samples("xgboost_breast_cancer")
            assert result["evidence"]["samples"] == []
            assert "No precomputed" in result["narrative"]
        finally:
            server_module._store_path = old_store


class TestStoreUnavailable:
    """S5.8 — Store unavailable falls back gracefully."""

    def test_explain_with_no_store(self):
        """When store path is None, falls back to on-the-fly."""
        old_store = server_module._store_path
        server_module._store_path = None
        try:
            result = explain_prediction("xgboost_breast_cancer", 0, include_plot=False)
            # Should still work via on-the-fly
            assert result["metadata"]["source"] == "on_the_fly"
        finally:
            server_module._store_path = old_store


class TestStandardBriefing:
    """Standard briefing should summarize persisted batch results."""

    def test_standard_briefing_from_store(self, tmp_path):
        save_explanations(
            [
                _make_stored_explanation(sample_index=10, run_id="run-brief"),
                _make_stored_explanation(sample_index=11, run_id="run-brief"),
            ],
            tmp_path,
        )
        save_drift_results(
            [
                _make_stored_drift(feature_name="f1", run_id="run-brief", severity="severe"),
                _make_stored_drift(feature_name="f2", run_id="run-brief", severity="moderate"),
            ],
            tmp_path,
        )

        old_store = server_module._store_path
        server_module._store_path = tmp_path
        try:
            result = standard_briefing(model_id="xgboost_breast_cancer", run_id="run-brief")
            assert result["metadata"]["source"] == "precomputed"
            assert result["evidence"]["count_models"] == 1
            model = result["evidence"]["models"][0]
            assert model["model_id"] == "xgboost_breast_cancer"
            assert model["run_id"] == "run-brief"
            assert model["n_precomputed_explanations"] == 2
            assert model["n_drifted_features"] == 2
            assert "top_explained_cases" in model
        finally:
            server_module._store_path = old_store

    def test_standard_briefing_empty_store(self, tmp_path):
        old_store = server_module._store_path
        server_module._store_path = tmp_path
        try:
            result = standard_briefing(model_id="xgboost_breast_cancer")
            assert result["metadata"]["source"] == "precomputed"
            assert result["evidence"]["count_models"] == 0
            assert "No precomputed batch results" in result["narrative"]
        finally:
            server_module._store_path = old_store
