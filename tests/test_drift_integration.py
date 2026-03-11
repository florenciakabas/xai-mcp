"""Integration tests for drift detection — end-to-end through MCP tools.

Covers scenarios F1–F2, G1–G2, H1–H3 from the BDD scenarios.
"""

import pytest
from pydantic import ValidationError

from xai_toolkit.schemas import (
    DatasetDriftResult,
    DistributionSummary,
    FeatureDriftResult,
)
from xai_toolkit.server import detect_drift, detect_feature_drift


# ---------------------------------------------------------------------------
# Group F: detect_drift MCP tool
# ---------------------------------------------------------------------------


class TestDetectDriftTool:
    """F1–F2: Dataset-level drift MCP tool."""

    def test_tool_response_shape(self):
        """F1: Returns ToolResponse with narrative, evidence, metadata."""
        result = detect_drift("xgboost_breast_cancer")

        assert isinstance(result, dict)
        assert "narrative" in result
        assert "evidence" in result
        assert "metadata" in result
        assert result.get("grounded") is True
        assert result["metadata"]["model_id"] == "xgboost_breast_cancer"

        # Evidence contains drift fields
        evidence = result["evidence"]
        assert "features" in evidence
        assert "n_features" in evidence
        assert "n_drifted" in evidence
        assert "share_drifted" in evidence
        assert "overall_severity" in evidence

    def test_tool_unknown_model(self):
        """F2: Unknown model → ErrorResponse."""
        result = detect_drift("nonexistent_model")

        assert isinstance(result, dict)
        assert result.get("error_code") == "MODEL_NOT_FOUND"
        assert "available" in result


# ---------------------------------------------------------------------------
# Group G: detect_feature_drift MCP tool
# ---------------------------------------------------------------------------


class TestDetectFeatureDriftTool:
    """G1–G2: Feature-level drift MCP tool."""

    def test_feature_tool_response_shape(self):
        """G1: Returns ToolResponse with narrative, evidence, metadata."""
        result = detect_feature_drift("xgboost_breast_cancer", "mean radius")

        assert isinstance(result, dict)
        assert "narrative" in result
        assert "evidence" in result
        assert "metadata" in result

        evidence = result["evidence"]
        assert "feature_name" in evidence
        assert evidence["feature_name"] == "mean radius"
        assert "test_name" in evidence
        assert "statistic" in evidence
        assert "severity" in evidence
        assert "reference_summary" in evidence
        assert "current_summary" in evidence

    def test_feature_tool_unknown_feature(self):
        """G2: Unknown feature → ErrorResponse."""
        result = detect_feature_drift("xgboost_breast_cancer", "nonexistent_feature")

        assert isinstance(result, dict)
        assert result.get("error_code") == "FEATURE_NOT_FOUND"
        assert "available" in result

    def test_feature_tool_unknown_model(self):
        """Unknown model → ErrorResponse."""
        result = detect_feature_drift("nonexistent_model", "mean radius")

        assert isinstance(result, dict)
        assert result.get("error_code") == "MODEL_NOT_FOUND"


# ---------------------------------------------------------------------------
# Group H: Schema validation
# ---------------------------------------------------------------------------


class TestSchemaValidation:
    """H1–H3: Pydantic rejects invalid data."""

    def test_schema_rejects_invalid_pvalue(self):
        """H1: p_value > 1.0 → ValidationError."""
        with pytest.raises(ValidationError):
            FeatureDriftResult(
                feature_name="test",
                test_name="ks",
                statistic=0.5,
                p_value=1.5,
                drift_detected=True,
                severity="moderate",
                reference_summary=DistributionSummary(
                    mean=0, std=1, median=0, min=-3, max=3,
                    n_samples=100, quantile_25=-0.67, quantile_75=0.67,
                ),
                current_summary=DistributionSummary(
                    mean=0, std=1, median=0, min=-3, max=3,
                    n_samples=100, quantile_25=-0.67, quantile_75=0.67,
                ),
            )

    def test_schema_rejects_negative_ndrifted(self):
        """H2: n_drifted < 0 → ValidationError."""
        with pytest.raises(ValidationError):
            DatasetDriftResult(
                features=[],
                n_features=1,
                n_drifted=-1,
                share_drifted=0.0,
                overall_severity="none",
            )

    def test_schema_rejects_zero_nsamples(self):
        """H3: n_samples = 0 → ValidationError."""
        with pytest.raises(ValidationError):
            DistributionSummary(
                mean=0, std=0, median=0, min=0, max=0,
                n_samples=0, quantile_25=0, quantile_75=0,
            )
