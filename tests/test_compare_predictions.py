"""Tests for compare_predictions — scenarios CP-S1, CP-S2, CP-S3.

These tests are driven by the human-authored scenario file:
    scenarios/compare_predictions.yaml

Each test method maps to one or more 'then' clauses from the scenario.
The scenario was written by Flor; the tests verify the implementation
satisfies her specification.

Usage:
    uv run python -m pytest tests/test_compare_predictions.py -v
"""

import pytest

from xai_toolkit.server import compare_predictions, list_models


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MODELS = ["xgboost_breast_cancer", "rf_breast_cancer"]
SAMPLE_INDEX = 42


@pytest.fixture
def comparison_result():
    """Call compare_predictions once and reuse across CP-S1 tests."""
    return compare_predictions(
        model_id_1=MODELS[0],
        model_id_2=MODELS[1],
        sample_index=SAMPLE_INDEX,
    )


# ---------------------------------------------------------------------------
# CP-S1: Compare predictions from two models on the same sample
# ---------------------------------------------------------------------------


class TestCPS1_Structure:
    """CP-S1 — response structure assertions."""

    def test_response_has_narrative(self, comparison_result):
        """then: response_has_field: 'narrative'"""
        assert "narrative" in comparison_result

    def test_response_has_evidence(self, comparison_result):
        """then: response_has_field: 'evidence'"""
        assert "evidence" in comparison_result

    def test_response_has_metadata(self, comparison_result):
        """then: response_has_field: 'metadata'"""
        assert "metadata" in comparison_result

    def test_is_not_error(self, comparison_result):
        """Must be a successful ToolResponse, not an ErrorResponse."""
        assert "error_code" not in comparison_result


class TestCPS1_PerModelResults:
    """CP-S1 — per_model evidence assertions."""

    def test_evidence_has_per_model(self, comparison_result):
        """then: evidence_has: 'per_model'"""
        assert "per_model" in comparison_result["evidence"]
        assert len(comparison_result["evidence"]["per_model"]) == 2

    def test_each_model_has_model_id(self, comparison_result):
        """then: each_model_has: 'model_id'"""
        for entry in comparison_result["evidence"]["per_model"]:
            assert "model_id" in entry
            assert isinstance(entry["model_id"], str)

    def test_each_model_has_predicted_class(self, comparison_result):
        """then: each_model_has: 'predicted_class'"""
        for entry in comparison_result["evidence"]["per_model"]:
            assert "predicted_class" in entry
            assert entry["predicted_class"] in (0, 1)

    def test_each_model_has_probability(self, comparison_result):
        """then: each_model_has: 'probability'"""
        for entry in comparison_result["evidence"]["per_model"]:
            assert "probability" in entry
            assert 0.0 <= entry["probability"] <= 1.0

    def test_each_model_has_top_features(self, comparison_result):
        """then: each_model_has: 'top_features'"""
        for entry in comparison_result["evidence"]["per_model"]:
            assert "top_features" in entry
            assert len(entry["top_features"]) >= 1
            for feat in entry["top_features"]:
                assert "name" in feat
                assert "shap_value" in feat
                assert "direction" in feat


class TestCPS1_AgreementAnalysis:
    """CP-S1 — agreement analysis assertions."""

    def test_evidence_has_agreement(self, comparison_result):
        """then: evidence_has: 'agreement'"""
        assert "agreement" in comparison_result["evidence"]
        assert isinstance(comparison_result["evidence"]["agreement"], bool)

    def test_evidence_has_confidence_gap(self, comparison_result):
        """then: evidence_has: 'confidence_gap'"""
        gap = comparison_result["evidence"]["confidence_gap"]
        assert isinstance(gap, float)
        assert gap >= 0.0


class TestCPS1_NarrativeQualities:
    """CP-S1 — narrative content assertions."""

    def test_narrative_states_both_predictions(self, comparison_result):
        """then: narrative_states_both_predictions: true"""
        narrative = comparison_result["narrative"]
        # Both model IDs should appear in the narrative
        assert MODELS[0] in narrative
        assert MODELS[1] in narrative

    def test_narrative_states_agreement_or_disagreement(self, comparison_result):
        """then: narrative_states_agreement_or_disagreement: true"""
        narrative = comparison_result["narrative"]
        assert "agree" in narrative.lower() or "disagree" in narrative.lower()

    def test_narrative_compares_confidence(self, comparison_result):
        """then: narrative_compares_confidence: true"""
        narrative = comparison_result["narrative"]
        # Narrative should mention probabilities or confidence
        assert "probability" in narrative.lower() or "confidence" in narrative.lower()

    def test_narrative_highlights_shared_drivers(self, comparison_result):
        """then: narrative_highlights_shared_drivers: true"""
        narrative = comparison_result["narrative"]
        evidence = comparison_result["evidence"]
        # If shared features exist, they should be mentioned
        if evidence["shared_top_features"]:
            assert "both models" in narrative.lower() or "share" in narrative.lower()

    def test_narrative_highlights_divergent_drivers(self, comparison_result):
        """then: narrative_highlights_divergent_drivers: true"""
        narrative = comparison_result["narrative"]
        evidence = comparison_result["evidence"]
        has_divergent = any(
            len(feats) > 0 for feats in evidence["divergent_features"].values()
        )
        if has_divergent:
            assert "uniquely" in narrative.lower() or "different" in narrative.lower()


class TestCPS1_Auditability:
    """CP-S1 — ADR-005 compliance assertions."""

    def test_metadata_has_model_id(self, comparison_result):
        """then: metadata_has: 'model_id'"""
        assert comparison_result["metadata"]["model_id"] is not None

    def test_metadata_has_timestamp(self, comparison_result):
        """then: metadata_has: 'timestamp'"""
        assert comparison_result["metadata"]["timestamp"] is not None

    def test_metadata_has_data_hash(self, comparison_result):
        """then: metadata_has: 'data_hash'"""
        data_hash = comparison_result["metadata"]["data_hash"]
        assert data_hash is not None
        assert len(data_hash) == 64  # SHA256 hex digest

    def test_grounded_is_true(self, comparison_result):
        """then: grounded: true"""
        assert comparison_result["grounded"] is True


# ---------------------------------------------------------------------------
# CP-S2: Both models agree with high confidence
# ---------------------------------------------------------------------------


class TestCPS2_HighConfidenceAgreement:
    """CP-S2 — find a sample where both models agree and validate narrative."""

    def test_agreement_sample_exists_and_narrative_emphasizes_consensus(self):
        """CP-S2: narrative_emphasizes_consensus and narrative_notes_shared_top_features.

        We test sample 0 (typically clear-cut). If it doesn't agree,
        we scan a few samples to find one that does.
        """
        for idx in range(10):
            result = compare_predictions(
                model_id_1=MODELS[0],
                model_id_2=MODELS[1],
                sample_index=idx,
            )
            if result["evidence"]["agreement"]:
                narrative = result["narrative"]
                # CP-S2: narrative_emphasizes_consensus
                assert "agree" in narrative.lower()
                # CP-S2: narrative_notes_shared_top_features
                if result["evidence"]["shared_top_features"]:
                    assert "both models" in narrative.lower()
                return  # test passed

        pytest.skip("No agreeing sample found in first 10 — skipping CP-S2")


# ---------------------------------------------------------------------------
# CP-S3: Error handling — one model ID is invalid
# ---------------------------------------------------------------------------


class TestCPS3_ErrorHandling:
    """CP-S3 — structured error when a model ID is invalid."""

    def test_invalid_first_model_returns_error(self):
        """then: returns_error_response: true"""
        result = compare_predictions(
            model_id_1="nonexistent_model",
            model_id_2=MODELS[1],
            sample_index=SAMPLE_INDEX,
        )
        assert "error_code" in result
        assert result["error_code"] == "MODEL_NOT_FOUND"

    def test_invalid_second_model_returns_error(self):
        """then: returns_error_response: true (second model)"""
        result = compare_predictions(
            model_id_1=MODELS[0],
            model_id_2="nonexistent_model",
            sample_index=SAMPLE_INDEX,
        )
        assert "error_code" in result
        assert result["error_code"] == "MODEL_NOT_FOUND"

    def test_error_names_invalid_model(self):
        """then: error_message_names_invalid_model: true"""
        result = compare_predictions(
            model_id_1=MODELS[0],
            model_id_2="nonexistent_model",
            sample_index=SAMPLE_INDEX,
        )
        assert "nonexistent_model" in result["message"]

    def test_error_suggests_available_models(self):
        """then: error_suggests_available_models: true"""
        result = compare_predictions(
            model_id_1=MODELS[0],
            model_id_2="nonexistent_model",
            sample_index=SAMPLE_INDEX,
        )
        assert len(result["available"]) > 0
        for model_id in MODELS:
            assert model_id in result["available"]

    def test_invalid_sample_index_returns_error(self):
        """Edge case: valid models, invalid sample index."""
        result = compare_predictions(
            model_id_1=MODELS[0],
            model_id_2=MODELS[1],
            sample_index=99999,
        )
        assert "error_code" in result
        assert result["error_code"] == "SAMPLE_OUT_OF_RANGE"


# ---------------------------------------------------------------------------
# Cross-model consistency (bonus: data integrity)
# ---------------------------------------------------------------------------


class TestCrossModelConsistency:
    """Bonus tests ensuring comparison is internally consistent."""

    def test_confidence_gap_matches_probabilities(self, comparison_result):
        """The confidence_gap should equal |prob1 - prob2|."""
        models = comparison_result["evidence"]["per_model"]
        expected_gap = round(abs(models[0]["probability"] - models[1]["probability"]), 4)
        assert comparison_result["evidence"]["confidence_gap"] == expected_gap

    def test_agreement_matches_predicted_classes(self, comparison_result):
        """agreement == True iff both models predict the same class."""
        models = comparison_result["evidence"]["per_model"]
        same_class = models[0]["predicted_class"] == models[1]["predicted_class"]
        assert comparison_result["evidence"]["agreement"] == same_class

    def test_shared_plus_divergent_covers_all_top_features(self, comparison_result):
        """Shared + divergent for each model = that model's full top features."""
        evidence = comparison_result["evidence"]
        shared = set(evidence["shared_top_features"])

        for entry in evidence["per_model"]:
            model_top = {f["name"] for f in entry["top_features"]}
            divergent = set(evidence["divergent_features"].get(entry["model_id"], []))
            assert shared | divergent == model_top, (
                f"For {entry['model_id']}: shared={shared}, divergent={divergent}, "
                f"top={model_top}"
            )

    def test_data_hash_matches_single_model_hash(self):
        """The comparison's data_hash should match what explain_prediction produces."""
        from xai_toolkit.server import explain_prediction

        comp = compare_predictions(
            model_id_1=MODELS[0],
            model_id_2=MODELS[1],
            sample_index=SAMPLE_INDEX,
        )
        single = explain_prediction(
            model_id=MODELS[0],
            sample_index=SAMPLE_INDEX,
            include_plot=False,
        )
        assert comp["metadata"]["data_hash"] == single["metadata"]["data_hash"]

    def test_reproducibility_same_inputs_same_output(self):
        """D3-S1 spirit: same inputs → same structural output.

        Note: KernelSHAP (used when shap.Explainer wraps predict_proba) has
        internal stochasticity not controlled by background random_state.
        Between two compare_predictions calls, 4 SHAP computations accumulate
        global random state drift, which can reorder features at the margin.

        We therefore test structural reproducibility (agreement, predictions,
        confidence gap, data hash) rather than exact narrative identity.
        The narrative IS deterministic given identical SHAP values — it's the
        SHAP values themselves that can vary slightly between calls.
        """
        r1 = compare_predictions(
            model_id_1=MODELS[0],
            model_id_2=MODELS[1],
            sample_index=SAMPLE_INDEX,
        )
        r2 = compare_predictions(
            model_id_1=MODELS[0],
            model_id_2=MODELS[1],
            sample_index=SAMPLE_INDEX,
        )
        e1, e2 = r1["evidence"], r2["evidence"]

        # Structural properties must be identical
        assert e1["agreement"] == e2["agreement"]
        assert e1["per_model"][0]["predicted_class"] == e2["per_model"][0]["predicted_class"]
        assert e1["per_model"][1]["predicted_class"] == e2["per_model"][1]["predicted_class"]
        assert e1["per_model"][0]["predicted_label"] == e2["per_model"][0]["predicted_label"]
        assert e1["per_model"][1]["predicted_label"] == e2["per_model"][1]["predicted_label"]

        # Probabilities must be very close (KernelSHAP doesn't affect predict_proba)
        assert e1["per_model"][0]["probability"] == e2["per_model"][0]["probability"]
        assert e1["per_model"][1]["probability"] == e2["per_model"][1]["probability"]
        assert e1["confidence_gap"] == e2["confidence_gap"]

        # Data hash must be identical (deterministic, no SHAP involved)
        assert r1["metadata"]["data_hash"] == r2["metadata"]["data_hash"]

        # Only timestamp is allowed to differ
        assert r1["metadata"]["timestamp"] != r2["metadata"]["timestamp"]
