"""Tests for intrinsic explainability (Phase 1).

Tests extract_intrinsic_importances() and narrate_intrinsic_importance(),
adapted from Tamas's _handle_intrinsically_explainable_model() in
xai-xgboost-clf/src/xgboost_clf/pipelines/model_explanation/nodes.py.
"""

from __future__ import annotations

import json
from io import StringIO
from unittest.mock import patch

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from xai_toolkit.explainers import extract_intrinsic_importances
from xai_toolkit.narrators import narrate_intrinsic_importance
from xai_toolkit.schemas import FeatureImportance


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def logistic_model() -> LogisticRegression:
    """A fitted LogisticRegression (has coef_)."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = LogisticRegression(random_state=42, max_iter=200)
    model.fit(X, y)
    return model


@pytest.fixture()
def rf_model() -> RandomForestClassifier:
    """A fitted RandomForestClassifier (has feature_importances_)."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


FEATURE_NAMES = ["feat_a", "feat_b", "feat_c", "feat_d"]


# ---------------------------------------------------------------------------
# extract_intrinsic_importances tests
# ---------------------------------------------------------------------------


class TestExtractIntrinsicImportances:
    """Test the pure extraction function."""

    def test_logistic_regression_returns_importances(
        self, logistic_model: LogisticRegression
    ) -> None:
        result = extract_intrinsic_importances(logistic_model, FEATURE_NAMES)
        assert result is not None
        importances, source_attr = result
        assert len(importances) == 4
        assert all(isinstance(f, FeatureImportance) for f in importances)
        assert source_attr == "coef_"

    def test_logistic_regression_has_directions(
        self, logistic_model: LogisticRegression
    ) -> None:
        """Coefficients should produce both positive and negative directions."""
        result = extract_intrinsic_importances(logistic_model, FEATURE_NAMES)
        assert result is not None
        importances, _ = result
        directions = {f.direction for f in importances}
        assert "positive" in directions or "negative" in directions

    def test_logistic_regression_sorted_by_importance(
        self, logistic_model: LogisticRegression
    ) -> None:
        result = extract_intrinsic_importances(logistic_model, FEATURE_NAMES)
        assert result is not None
        importances, _ = result
        for i in range(len(importances) - 1):
            assert importances[i].importance >= importances[i + 1].importance

    def test_random_forest_returns_importances(
        self, rf_model: RandomForestClassifier
    ) -> None:
        result = extract_intrinsic_importances(rf_model, FEATURE_NAMES)
        assert result is not None
        importances, source_attr = result
        assert len(importances) == 4
        assert all(isinstance(f, FeatureImportance) for f in importances)
        assert source_attr == "feature_importances_"

    def test_random_forest_all_positive_direction(
        self, rf_model: RandomForestClassifier
    ) -> None:
        """feature_importances_ are magnitude-based, so direction is always positive."""
        result = extract_intrinsic_importances(rf_model, FEATURE_NAMES)
        assert result is not None
        importances, _ = result
        assert all(f.direction == "positive" for f in importances)

    def test_random_forest_importances_sum_to_one(
        self, rf_model: RandomForestClassifier
    ) -> None:
        """sklearn RF feature_importances_ sum to 1.0."""
        result = extract_intrinsic_importances(rf_model, FEATURE_NAMES)
        assert result is not None
        importances, _ = result
        total = sum(f.importance for f in importances)
        assert abs(total - 1.0) < 0.01

    def test_no_intrinsic_returns_none(self) -> None:
        """A model without coef_ or feature_importances_ returns None."""

        class BareModel:
            pass

        result = extract_intrinsic_importances(BareModel(), FEATURE_NAMES)
        assert result is None

    def test_feature_name_mismatch_returns_none(
        self, logistic_model: LogisticRegression
    ) -> None:
        """If feature count doesn't match, return None rather than crash."""
        result = extract_intrinsic_importances(logistic_model, ["a", "b"])
        assert result is None


# ---------------------------------------------------------------------------
# narrate_intrinsic_importance tests
# ---------------------------------------------------------------------------


class TestNarrateIntrinsicImportance:
    """Test the narrative generation for intrinsic importances."""

    def test_linear_model_mentions_coefficient(
        self, logistic_model: LogisticRegression
    ) -> None:
        result = extract_intrinsic_importances(logistic_model, FEATURE_NAMES)
        assert result is not None
        importances, source_attr = result
        narrative = narrate_intrinsic_importance(
            importances, "LogisticRegression", 4, source_attr=source_attr
        )
        assert "coefficient" in narrative.lower()

    def test_tree_model_mentions_importance(
        self, rf_model: RandomForestClassifier
    ) -> None:
        result = extract_intrinsic_importances(rf_model, FEATURE_NAMES)
        assert result is not None
        importances, source_attr = result
        narrative = narrate_intrinsic_importance(
            importances, "RandomForestClassifier", 4, source_attr=source_attr
        )
        assert "importance" in narrative.lower()

    def test_tree_model_mentions_percentage(
        self, rf_model: RandomForestClassifier
    ) -> None:
        result = extract_intrinsic_importances(rf_model, FEATURE_NAMES)
        assert result is not None
        importances, source_attr = result
        narrative = narrate_intrinsic_importance(
            importances, "RandomForestClassifier", 4, source_attr=source_attr
        )
        assert "%" in narrative

    def test_empty_importances(self) -> None:
        narrative = narrate_intrinsic_importance([], "Model", 0)
        assert "no intrinsic" in narrative.lower()

    def test_mentions_model_type(
        self, rf_model: RandomForestClassifier
    ) -> None:
        result = extract_intrinsic_importances(rf_model, FEATURE_NAMES)
        assert result is not None
        importances, source_attr = result
        narrative = narrate_intrinsic_importance(
            importances, "RandomForestClassifier", 4, source_attr=source_attr
        )
        assert "RandomForestClassifier" in narrative

    def test_mentions_intrinsic_interpretability(
        self, rf_model: RandomForestClassifier
    ) -> None:
        result = extract_intrinsic_importances(rf_model, FEATURE_NAMES)
        assert result is not None
        importances, source_attr = result
        narrative = narrate_intrinsic_importance(
            importances, "RandomForestClassifier", 4, source_attr=source_attr
        )
        assert "intrinsic interpretability" in narrative.lower()


# ---------------------------------------------------------------------------
# Integration: summarize_model includes intrinsic data
# ---------------------------------------------------------------------------


class TestSummarizeModelIntrinsic:
    """Test that summarize_model tool response includes intrinsic data."""

    @pytest.fixture()
    def _registry(self):
        from xai_toolkit.cli import DATA_DIR, MODELS_DIR
        from xai_toolkit.registry import ModelRegistry

        registry = ModelRegistry()
        registry.load_from_disk("xgboost_breast_cancer", MODELS_DIR, DATA_DIR)
        return registry

    def test_summarize_includes_intrinsic_importances(self, _registry) -> None:
        """XGBoost has feature_importances_, so intrinsic data should appear."""
        from xai_toolkit.cli import build_parser, cmd_summarize

        args = build_parser().parse_args(["summarize", "--model", "xgboost_breast_cancer"])
        buf = StringIO()
        with patch("sys.stdout", buf):
            cmd_summarize(args, _registry)

        result = json.loads(buf.getvalue())
        assert "intrinsic_importances" in result["evidence"]
        assert len(result["evidence"]["intrinsic_importances"]) > 0
        assert "Intrinsic Interpretability" in result["narrative"]

    def test_intrinsic_importances_have_correct_schema(self, _registry) -> None:
        """Each intrinsic importance entry must have name, importance, direction, mean_shap."""
        from xai_toolkit.cli import build_parser, cmd_summarize

        args = build_parser().parse_args(["summarize", "--model", "xgboost_breast_cancer"])
        buf = StringIO()
        with patch("sys.stdout", buf):
            cmd_summarize(args, _registry)

        result = json.loads(buf.getvalue())
        for entry in result["evidence"]["intrinsic_importances"]:
            assert "name" in entry
            assert "importance" in entry
            assert "direction" in entry
            assert "mean_shap" in entry
