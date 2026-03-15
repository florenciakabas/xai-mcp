"""Tests for result store read/write (Scenario Group 4)."""

import json

import pytest

from xai_toolkit.result_store import (
    LATEST_RUN_STRATEGY,
    StoredDriftResult,
    StoredExplanation,
    StoredModelSummary,
    load_drift_results,
    load_explanation,
    load_explanations,
    resolve_latest_successful_run_id,
    save_drift_results,
    save_explanations,
    save_model_summaries,
)
from xai_toolkit.schemas import (
    StoredDriftResult as StoredDriftResultSchema,
    StoredExplanation as StoredExplanationSchema,
    StoredModelSummary as StoredModelSummarySchema,
)


def _make_explanation(
    model_id="test_model",
    sample_index=0,
    run_id="run-001",
    computed_at="2026-03-13T14:00:00Z",
) -> StoredExplanation:
    return StoredExplanation(
        run_id=run_id,
        model_id=model_id,
        sample_index=sample_index,
        prediction=1,
        prediction_label="malignant",
        probability=0.87,
        narrative="Test narrative for sample.",
        top_features=json.dumps([{"name": "f1", "shap_value": 0.3}]),
        shap_values=json.dumps({"f1": 0.3, "f2": -0.1}),
        feature_values=json.dumps({"f1": 1.5, "f2": 2.0}),
        data_hash="a" * 64,
        computed_at=computed_at,
    )


def _make_drift_result(
    model_id="test_model",
    feature_name="f1",
    run_id="run-001",
    severity="moderate",
    drift_detected=True,
    computed_at="2026-03-13T14:00:00Z",
) -> StoredDriftResult:
    return StoredDriftResult(
        run_id=run_id,
        model_id=model_id,
        feature_name=feature_name,
        test_name="psi",
        statistic=0.15,
        p_value=0.03,
        drift_detected=drift_detected,
        severity=severity,
        narrative=f"Feature {feature_name} shows {severity} drift.",
        overall_narrative="Overall moderate drift detected.",
        overall_severity="moderate",
        computed_at=computed_at,
    )


class TestSaveLoadExplanations:
    """S4.1–S4.2 — Happy: round-trip and lookup."""

    def test_round_trip(self, tmp_path):
        """S4.1 — Save and load explanations preserve all fields."""
        explanations = [
            _make_explanation(sample_index=i) for i in range(5)
        ]
        save_explanations(explanations, tmp_path)

        loaded = load_explanations("test_model", tmp_path)
        assert len(loaded) == 5
        for orig, loaded_exp in zip(explanations, loaded):
            assert loaded_exp.narrative == orig.narrative
            assert loaded_exp.shap_values == orig.shap_values
            assert loaded_exp.data_hash == orig.data_hash

    def test_load_by_sample_index(self, tmp_path):
        """S4.2 — Load by model_id and sample_index."""
        explanations = [
            _make_explanation(sample_index=i) for i in range(5)
        ]
        save_explanations(explanations, tmp_path)

        result = load_explanation("test_model", 3, tmp_path)
        assert result is not None
        assert result.sample_index == 3

    def test_multiple_models(self, tmp_path):
        """S4.2 — Multiple models coexist."""
        for mid in ("model_a", "model_b", "model_c"):
            exps = [_make_explanation(model_id=mid, sample_index=i) for i in range(5)]
            save_explanations(exps, tmp_path)

        result = load_explanation("model_b", 2, tmp_path)
        assert result is not None
        assert result.model_id == "model_b"
        assert result.sample_index == 2

    def test_mixed_model_ids_raise_value_error(self, tmp_path):
        """Mixed model batches should be rejected explicitly."""
        mixed = [
            _make_explanation(model_id="model_a", sample_index=0),
            _make_explanation(model_id="model_b", sample_index=1),
        ]
        with pytest.raises(
            ValueError,
            match="must share the same model_id",
        ):
            save_explanations(mixed, tmp_path)


class TestMultipleRuns:
    """S4.3 — Multiple runs coexist."""

    def test_latest_run_returned_by_default(self, tmp_path):
        old = [_make_explanation(run_id="run-old", computed_at="2026-03-12T10:00:00Z")]
        new = [_make_explanation(run_id="run-new", computed_at="2026-03-13T14:00:00Z")]
        save_explanations(old, tmp_path)
        save_explanations(new, tmp_path)

        result = load_explanation("test_model", 0, tmp_path)
        assert result is not None
        assert result.run_id == "run-new"

    def test_specific_run_id_filter(self, tmp_path):
        old = [_make_explanation(run_id="run-old", computed_at="2026-03-12T10:00:00Z")]
        new = [_make_explanation(run_id="run-new", computed_at="2026-03-13T14:00:00Z")]
        save_explanations(old, tmp_path)
        save_explanations(new, tmp_path)

        result = load_explanation("test_model", 0, tmp_path, run_id="run-old")
        assert result is not None
        assert result.run_id == "run-old"

    def test_latest_run_resolver_is_deterministic(self):
        import pandas as pd

        df = pd.DataFrame(
            {
                "run_id": ["run-a", "run-b", "run-c"],
                "computed_at": [
                    "2026-03-12T10:00:00Z",
                    "2026-03-14T10:00:00Z",
                    "2026-03-13T10:00:00Z",
                ],
            }
        )
        assert resolve_latest_successful_run_id(df) == "run-b"

    def test_load_explanation_defaults_to_latest_run_only(self, tmp_path):
        old = [_make_explanation(sample_index=42, run_id="run-old", computed_at="2026-03-12T10:00:00Z")]
        new = [_make_explanation(sample_index=7, run_id="run-new", computed_at="2026-03-13T14:00:00Z")]
        save_explanations(old, tmp_path)
        save_explanations(new, tmp_path)

        # sample 42 exists only in older run and should not leak into "latest run" reads
        result = load_explanation("test_model", 42, tmp_path)
        assert result is None


class TestEmptyStore:
    """S4.4–S4.6 — Unhappy paths."""

    def test_load_from_empty_store(self, tmp_path):
        """S4.4 — Returns None, not exception."""
        result = load_explanation("test_model", 0, tmp_path)
        assert result is None

    def test_load_nonexistent_model(self, tmp_path):
        """S4.5 — Returns None for unknown model."""
        exps = [_make_explanation(model_id="model_a")]
        save_explanations(exps, tmp_path)

        result = load_explanation("model_b", 0, tmp_path)
        assert result is None

    def test_corrupted_parquet(self, tmp_path):
        """S4.6 — Corrupted file returns None."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()
        (model_dir / "explanations.parquet").write_text("not a parquet file")

        result = load_explanation("test_model", 0, tmp_path)
        assert result is None


class TestDriftStore:
    """Drift result store round-trip."""

    def test_drift_round_trip(self, tmp_path):
        results = [
            _make_drift_result(feature_name="f1", severity="severe"),
            _make_drift_result(feature_name="f2", severity="none", drift_detected=False),
        ]
        save_drift_results(results, tmp_path)

        loaded = load_drift_results("test_model", tmp_path)
        assert len(loaded) == 2
        assert loaded[0].feature_name == "f1"
        assert loaded[0].severity == "severe"

    def test_drift_empty_store(self, tmp_path):
        result = load_drift_results("nonexistent", tmp_path)
        assert result == []

    def test_drift_mixed_model_ids_raise_value_error(self, tmp_path):
        mixed = [
            _make_drift_result(model_id="model_a", feature_name="f1"),
            _make_drift_result(model_id="model_b", feature_name="f2"),
        ]
        with pytest.raises(
            ValueError,
            match="must share the same model_id",
        ):
            save_drift_results(mixed, tmp_path)


class TestModelSummaryStore:
    def test_model_summary_mixed_model_ids_raise_value_error(self, tmp_path):
        mixed = [
            StoredModelSummary(
                run_id="run-001",
                model_id="model_a",
                feature_name="f1",
                importance=0.5,
                rank=1,
                narrative="Feature f1 is important.",
                model_type="xgboost",
                computed_at="2026-03-13T14:00:00Z",
            ),
            StoredModelSummary(
                run_id="run-001",
                model_id="model_b",
                feature_name="f1",
                importance=0.5,
                rank=1,
                narrative="Feature f1 is important.",
                model_type="xgboost",
                computed_at="2026-03-13T14:00:00Z",
            ),
        ]
        with pytest.raises(
            ValueError,
            match="must share the same model_id",
        ):
            save_model_summaries(mixed, tmp_path)


class TestStoredContractsSource:
    """Stored contracts should be defined in schemas.py (single source of truth)."""

    def test_stored_contracts_come_from_schemas_module(self):
        assert StoredExplanation is StoredExplanationSchema
        assert StoredDriftResult is StoredDriftResultSchema
        assert StoredModelSummary is StoredModelSummarySchema

    def test_latest_run_strategy_constant(self):
        assert LATEST_RUN_STRATEGY == "latest_successful_by_computed_at"
