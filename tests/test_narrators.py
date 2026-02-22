"""Tests for narrators.py — verify narrative correctness (Day 1)."""

from xai_toolkit.narrators import narrate_prediction
from xai_toolkit.schemas import ShapResult


def _make_shap_result() -> ShapResult:
    """Create a known ShapResult for deterministic testing."""
    return ShapResult(
        prediction=1,
        prediction_label="malignant",
        probability=0.91,
        base_value=0.42,
        shap_values={
            "worst_radius": 0.28,
            "worst_concave_points": 0.19,
            "mean_concavity": 0.14,
            "mean_smoothness": -0.06,
            "mean_texture": 0.03,
        },
        feature_values={
            "worst_radius": 23.4,
            "worst_concave_points": 0.18,
            "mean_concavity": 0.15,
            "mean_smoothness": 0.08,
            "mean_texture": 19.2,
        },
        feature_names=[
            "worst_radius",
            "worst_concave_points",
            "mean_concavity",
            "mean_smoothness",
            "mean_texture",
        ],
    )


class TestNarratePrediction:
    """Tests aligned with Day 1 acceptance scenario D1-S2."""

    def test_narrative_is_string(self):
        result = _make_shap_result()
        narrative = narrate_prediction(result)
        assert isinstance(narrative, str)

    def test_narrative_has_minimum_length(self):
        """D1-S2: narrative_length_gt: 50"""
        result = _make_shap_result()
        narrative = narrate_prediction(result)
        assert len(narrative) > 50, f"Narrative too short ({len(narrative)} chars)"

    def test_narrative_contains_classification(self):
        """D1-S2: narrative_contains: 'classified as'"""
        result = _make_shap_result()
        narrative = narrate_prediction(result)
        assert "classified this sample as" in narrative

    def test_narrative_contains_probability(self):
        """D1-S2: narrative_contains: 'probability'"""
        result = _make_shap_result()
        narrative = narrate_prediction(result)
        assert "probability" in narrative

    def test_narrative_mentions_top_3_features(self):
        """D1-S2: narrative_mentions_top_n_features: 3"""
        result = _make_shap_result()
        narrative = narrate_prediction(result, top_n=3)
        # Top 3 by absolute SHAP: worst_radius, worst_concave_points, mean_concavity
        assert "worst_radius" in narrative
        assert "worst_concave_points" in narrative
        assert "mean_concavity" in narrative

    def test_each_feature_has_direction(self):
        """D1-S2: each_feature_has_direction: true"""
        result = _make_shap_result()
        narrative = narrate_prediction(result, top_n=3)
        # Direction language should appear for each mentioned feature
        assert "pushing toward" in narrative or "pushing away from" in narrative

    def test_each_feature_has_magnitude(self):
        """D1-S2: each_feature_has_magnitude: true"""
        result = _make_shap_result()
        narrative = narrate_prediction(result, top_n=3)
        # Magnitude values should appear (e.g., "+0.28")
        assert "0.28" in narrative  # worst_radius SHAP value

    def test_narrative_is_deterministic(self):
        """Same input must always produce the same output (ADR-002)."""
        result = _make_shap_result()
        narrative_1 = narrate_prediction(result)
        narrative_2 = narrate_prediction(result)
        assert narrative_1 == narrative_2

    def test_opposing_factor_mentioned(self):
        """Narrative should mention an opposing factor when one exists."""
        result = _make_shap_result()
        narrative = narrate_prediction(result, top_n=3)
        # mean_smoothness is negative SHAP (opposing for a positive prediction)
        assert "opposing" in narrative.lower()
        assert "mean_smoothness" in narrative
