"""Tests for framework-agnostic design — D4-S1, D4-S2.

Proves that all 7 MCP tools work identically with both XGBoost and
RandomForest models without any code changes. If this test file passes,
the abstraction is sound.

These tests load real models, so they are slower than unit tests.
They are the integration-level acceptance gate for Day 4.

Usage:
    uv run python scripts/train_rf_model.py   # run once first
    uv run python -m pytest tests/test_second_model.py -v
"""

import pytest

from xai_toolkit.server import (
    compare_features,
    describe_dataset,
    explain_prediction,
    explain_prediction_waterfall,
    get_partial_dependence,
    list_models,
    summarize_model,
)

# Both models share the same breast cancer dataset and test split
MODELS = ["xgboost_breast_cancer", "rf_breast_cancer"]
SAMPLE_INDEX = 42
FEATURE_NAME = "mean radius"  # valid in both models (same feature set)


# ---------------------------------------------------------------------------
# D4-S1: Every tool succeeds for both model types, zero code changes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_id", MODELS)
class TestAllToolsWorkForBothModels:
    """D4-S1: parametrized over both models — same code, same tools."""

    def test_list_models_includes_model(self, model_id: str):
        result = list_models()
        model_ids = [m["model_id"] for m in result["evidence"]["models"]]
        assert model_id in model_ids

    def test_summarize_model_returns_narrative(self, model_id: str):
        result = summarize_model(model_id=model_id)
        assert "narrative" in result
        assert len(result["narrative"]) > 50
        assert "error_code" not in result

    def test_summarize_model_has_data_hash(self, model_id: str):
        result = summarize_model(model_id=model_id)
        assert result["metadata"]["data_hash"] is not None
        assert len(result["metadata"]["data_hash"]) == 64

    def test_compare_features_returns_narrative(self, model_id: str):
        result = compare_features(model_id=model_id)
        assert "narrative" in result
        assert "error_code" not in result

    def test_compare_features_has_features(self, model_id: str):
        result = compare_features(model_id=model_id)
        assert len(result["evidence"]["features"]) > 0

    def test_describe_dataset_returns_narrative(self, model_id: str):
        result = describe_dataset(model_id=model_id)
        assert "narrative" in result
        assert "error_code" not in result

    def test_explain_prediction_returns_narrative(self, model_id: str):
        result = explain_prediction(
            model_id=model_id,
            sample_index=SAMPLE_INDEX,
            include_plot=False,  # skip plot for speed in CI
        )
        assert "narrative" in result
        assert "error_code" not in result
        assert "probability" in result["narrative"]

    def test_explain_prediction_has_shap_values(self, model_id: str):
        result = explain_prediction(
            model_id=model_id,
            sample_index=SAMPLE_INDEX,
            include_plot=False,
        )
        assert "shap_values" in result["evidence"]
        assert len(result["evidence"]["shap_values"]) == 30  # 30 features

    def test_explain_prediction_waterfall_returns_narrative(self, model_id: str):
        result = explain_prediction_waterfall(
            model_id=model_id,
            sample_index=SAMPLE_INDEX,
        )
        assert "narrative" in result
        assert "error_code" not in result

    def test_get_partial_dependence_returns_narrative(self, model_id: str):
        result = get_partial_dependence(
            model_id=model_id,
            feature_name=FEATURE_NAME,
            include_plot=False,
        )
        assert "narrative" in result
        assert "error_code" not in result
        assert FEATURE_NAME in result["narrative"]

    def test_model_type_is_correct(self, model_id: str):
        result = summarize_model(model_id=model_id)
        expected_types = {
            "xgboost_breast_cancer": "XGBClassifier",
            "rf_breast_cancer": "RandomForestClassifier",
        }
        assert result["metadata"]["model_type"] == expected_types[model_id]


# ---------------------------------------------------------------------------
# D4-S2: Models produce different but valid feature importances
# ---------------------------------------------------------------------------


class TestModelsProduceDifferentImportances:
    """D4-S2: XGBoost and RF agree on data but differ on feature rankings.

    This is expected behaviour — different algorithms weight features
    differently. The toolkit handles both transparently.
    """

    def test_top_feature_may_differ_between_models(self):
        """The two models are allowed to disagree on the #1 feature."""
        xgb = compare_features(model_id="xgboost_breast_cancer")
        rf = compare_features(model_id="rf_breast_cancer")
        xgb_top = xgb["evidence"]["features"][0]["name"]
        rf_top = rf["evidence"]["features"][0]["name"]
        # We don't assert they differ — just that both are valid feature names
        assert isinstance(xgb_top, str) and len(xgb_top) > 0
        assert isinstance(rf_top, str) and len(rf_top) > 0

    def test_both_models_explain_same_sample_differently(self):
        """Same sample index, different probabilities — models can disagree."""
        xgb = explain_prediction(
            model_id="xgboost_breast_cancer",
            sample_index=SAMPLE_INDEX,
            include_plot=False,
        )
        rf = explain_prediction(
            model_id="rf_breast_cancer",
            sample_index=SAMPLE_INDEX,
            include_plot=False,
        )
        xgb_prob = xgb["evidence"]["probability"]
        rf_prob = rf["evidence"]["probability"]
        # Both must be valid probabilities
        assert 0.0 <= xgb_prob <= 1.0
        assert 0.0 <= rf_prob <= 1.0

    def test_both_models_share_same_feature_names(self):
        """Both models trained on the same dataset — same 30 features.

        We request top_n=30 (all features) before comparing names so that
        differing importance rankings between XGBoost and RF don't cause the
        default top-10 truncation to produce different feature sets.
        """
        xgb = compare_features(model_id="xgboost_breast_cancer", top_n=30)
        rf = compare_features(model_id="rf_breast_cancer", top_n=30)
        xgb_names = {f["name"] for f in xgb["evidence"]["features"]}
        rf_names = {f["name"] for f in rf["evidence"]["features"]}
        assert xgb_names == rf_names, (
            f"Models should share all 30 feature names but got:\n"
            f"  XGB only: {xgb_names - rf_names}\n"
            f"  RF only:  {rf_names - xgb_names}"
        )

    def test_data_hashes_match_across_models(self):
        """Both models use the same test data — hashes must be identical."""
        xgb = describe_dataset(model_id="xgboost_breast_cancer")
        rf = describe_dataset(model_id="rf_breast_cancer")
        assert xgb["metadata"]["data_hash"] == rf["metadata"]["data_hash"]
