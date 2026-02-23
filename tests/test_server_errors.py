"""Tests for structured error responses from server.py (D3-S3, S4, S5).

Calls tool functions directly — no MCP server process needed.
The real registry is used so errors interact with actual loaded models,
giving us confidence the error paths work end-to-end.

Design: Tests verify the *shape* and *content* of error responses.
They do not test internal implementation details, keeping them resilient
to future refactoring.
"""

import pytest

from xai_toolkit.server import (
    _build_error,
    _extract_suggestion,
    compare_features,
    describe_dataset,
    explain_prediction,
    explain_prediction_waterfall,
    get_partial_dependence,
    summarize_model,
)

REAL_MODEL = "xgboost_breast_cancer"
GHOST_MODEL = "this_model_does_not_exist"
BREAST_CANCER_N_FEATURES = 30


# ---------------------------------------------------------------------------
# _build_error — unit tests for the helper itself
# ---------------------------------------------------------------------------


class TestBuildError:
    """_build_error always returns a fully-formed ErrorResponse dict."""

    def test_has_all_required_fields(self):
        err = _build_error("MODEL_NOT_FOUND", "Test message")
        for field in ("error_code", "message", "available", "suggestion"):
            assert field in err, f"Missing field: {field}"

    def test_error_code_is_preserved(self):
        err = _build_error("SAMPLE_OUT_OF_RANGE", "bad index")
        assert err["error_code"] == "SAMPLE_OUT_OF_RANGE"

    def test_message_is_preserved(self):
        err = _build_error("UNKNOWN_ERROR", "something broke")
        assert err["message"] == "something broke"

    def test_available_defaults_to_empty_list(self):
        err = _build_error("UNKNOWN_ERROR", "msg")
        assert err["available"] == []

    def test_available_is_populated_when_provided(self):
        err = _build_error("MODEL_NOT_FOUND", "msg", available=["model_a", "model_b"])
        assert err["available"] == ["model_a", "model_b"]

    def test_suggestion_defaults_to_none(self):
        err = _build_error("FEATURE_NOT_FOUND", "msg")
        assert err["suggestion"] is None

    def test_suggestion_is_populated_when_provided(self):
        err = _build_error("FEATURE_NOT_FOUND", "msg", suggestion="mean radius")
        assert err["suggestion"] == "mean radius"


# ---------------------------------------------------------------------------
# _extract_suggestion — unit tests for the regex helper
# ---------------------------------------------------------------------------


class TestExtractSuggestion:

    def test_extracts_single_suggestion(self):
        msg = "Feature 'mean_radus' not found. Did you mean: ['mean radius']? Available: ..."
        assert _extract_suggestion(msg) == "mean radius"

    def test_extracts_first_from_multiple(self):
        msg = "Did you mean: ['mean radius', 'worst radius']?"
        assert _extract_suggestion(msg) == "mean radius"

    def test_returns_none_when_no_suggestion(self):
        msg = "Feature not found. No close matches."
        assert _extract_suggestion(msg) is None

    def test_returns_none_for_empty_string(self):
        assert _extract_suggestion("") is None


# ---------------------------------------------------------------------------
# explain_prediction — D3-S3 (model not found) and D3-S4 (out of range)
# ---------------------------------------------------------------------------


class TestExplainPredictionErrors:

    # D3-S3: model not found
    def test_unknown_model_returns_model_not_found(self):
        result = explain_prediction(model_id=GHOST_MODEL, sample_index=0)
        assert result["error_code"] == "MODEL_NOT_FOUND"

    def test_unknown_model_message_names_the_bad_id(self):
        result = explain_prediction(model_id=GHOST_MODEL, sample_index=0)
        assert GHOST_MODEL in result["message"]

    def test_unknown_model_message_says_not_registered(self):
        result = explain_prediction(model_id=GHOST_MODEL, sample_index=0)
        assert "not registered" in result["message"].lower()

    def test_unknown_model_available_lists_real_models(self):
        """User is told what to use instead — available is non-empty."""
        result = explain_prediction(model_id=GHOST_MODEL, sample_index=0)
        assert isinstance(result["available"], list)
        assert len(result["available"]) >= 1
        assert REAL_MODEL in result["available"]

    # D3-S4: sample out of range
    def test_large_index_returns_sample_out_of_range(self):
        result = explain_prediction(model_id=REAL_MODEL, sample_index=99999)
        assert result["error_code"] == "SAMPLE_OUT_OF_RANGE"

    def test_large_index_message_contains_out_of_range(self):
        result = explain_prediction(model_id=REAL_MODEL, sample_index=99999)
        assert "out of range" in result["message"].lower()

    def test_large_index_message_contains_dataset_size(self):
        """Error message tells the user how many samples actually exist."""
        result = explain_prediction(model_id=REAL_MODEL, sample_index=99999)
        # compute_shap_values embeds the real dataset size in its message
        assert "114" in result["message"] or "samples" in result["message"].lower()

    def test_large_index_available_shows_valid_range(self):
        """available shows '0–N' so user knows what to ask for."""
        result = explain_prediction(model_id=REAL_MODEL, sample_index=99999)
        assert len(result["available"]) == 1
        assert "\u2013" in result["available"][0]  # en-dash in "0–113"

    def test_negative_index_returns_sample_out_of_range(self):
        result = explain_prediction(model_id=REAL_MODEL, sample_index=-1)
        assert result["error_code"] == "SAMPLE_OUT_OF_RANGE"

    # D3-S3 for waterfall tool too
    def test_waterfall_unknown_model_returns_model_not_found(self):
        result = explain_prediction_waterfall(model_id=GHOST_MODEL, sample_index=0)
        assert result["error_code"] == "MODEL_NOT_FOUND"

    def test_waterfall_large_index_returns_sample_out_of_range(self):
        result = explain_prediction_waterfall(model_id=REAL_MODEL, sample_index=99999)
        assert result["error_code"] == "SAMPLE_OUT_OF_RANGE"


# ---------------------------------------------------------------------------
# get_partial_dependence — D3-S5 (invalid feature name)
# ---------------------------------------------------------------------------


class TestGetPartialDependenceErrors:

    def test_typo_feature_returns_feature_not_found(self):
        """D3-S5: deliberate typo → FEATURE_NOT_FOUND."""
        result = get_partial_dependence(
            model_id=REAL_MODEL,
            feature_name="mean_radus",  # missing 'i'
        )
        assert result["error_code"] == "FEATURE_NOT_FOUND"

    def test_typo_feature_message_contains_bad_name(self):
        result = get_partial_dependence(
            model_id=REAL_MODEL,
            feature_name="mean_radus",
        )
        assert "mean_radus" in result["message"]

    def test_typo_feature_suggestion_is_populated(self):
        """D3-S5: suggestion field provides the closest valid name."""
        result = get_partial_dependence(
            model_id=REAL_MODEL,
            feature_name="mean_radus",
        )
        assert result["suggestion"] is not None
        assert len(result["suggestion"]) > 0

    def test_typo_feature_available_lists_all_features(self):
        """D3-S5: available contains all feature names so user can browse."""
        result = get_partial_dependence(
            model_id=REAL_MODEL,
            feature_name="completely_made_up_feature",
        )
        assert isinstance(result["available"], list)
        assert len(result["available"]) == BREAST_CANCER_N_FEATURES

    def test_completely_unknown_feature_returns_feature_not_found(self):
        result = get_partial_dependence(
            model_id=REAL_MODEL,
            feature_name="absolutely_not_a_feature_xyz",
        )
        assert result["error_code"] == "FEATURE_NOT_FOUND"

    def test_model_not_found_takes_priority_over_feature_error(self):
        """If model is missing, MODEL_NOT_FOUND is returned first."""
        result = get_partial_dependence(
            model_id=GHOST_MODEL,
            feature_name="mean_radus",
        )
        assert result["error_code"] == "MODEL_NOT_FOUND"


# ---------------------------------------------------------------------------
# All tools: MODEL_NOT_FOUND is consistent
# ---------------------------------------------------------------------------


class TestModelNotFoundConsistency:
    """Every tool that accepts model_id handles MODEL_NOT_FOUND the same way."""

    @pytest.mark.parametrize("tool_fn,kwargs", [
        (summarize_model,    {"model_id": GHOST_MODEL}),
        (compare_features,   {"model_id": GHOST_MODEL}),
        (describe_dataset,   {"model_id": GHOST_MODEL}),
    ])
    def test_returns_model_not_found_error_code(self, tool_fn, kwargs):
        result = tool_fn(**kwargs)
        assert result["error_code"] == "MODEL_NOT_FOUND"

    @pytest.mark.parametrize("tool_fn,kwargs", [
        (summarize_model,    {"model_id": GHOST_MODEL}),
        (compare_features,   {"model_id": GHOST_MODEL}),
        (describe_dataset,   {"model_id": GHOST_MODEL}),
    ])
    def test_available_is_non_empty_list(self, tool_fn, kwargs):
        result = tool_fn(**kwargs)
        assert isinstance(result["available"], list)
        assert len(result["available"]) >= 1
