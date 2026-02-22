"""Tests for narrators.py — verify narrative correctness.

All tests use hand-crafted data to keep them fast and deterministic.
No SHAP computation, no model loading — pure narrator logic.
"""

from xai_toolkit.narrators import (
    narrate_dataset,
    narrate_feature_comparison,
    narrate_model_summary,
    narrate_partial_dependence,
    narrate_prediction,
)
from xai_toolkit.schemas import (
    DatasetDescription,
    FeatureImportance,
    ModelSummary,
    PartialDependenceResult,
    ShapResult,
)


# --- Fixtures (hand-crafted data) ---


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


def _make_model_summary() -> ModelSummary:
    return ModelSummary(
        model_type="XGBClassifier",
        accuracy=0.956,
        n_features=30,
        n_train_samples=455,
        n_test_samples=114,
        target_names=["malignant", "benign"],
        top_features=[
            FeatureImportance(name="worst_radius", importance=0.15, direction="positive", mean_shap=0.12),
            FeatureImportance(name="worst_concave_points", importance=0.10, direction="positive", mean_shap=0.08),
            FeatureImportance(name="mean_concavity", importance=0.07, direction="positive", mean_shap=0.05),
        ],
    )


def _make_feature_importances() -> list[FeatureImportance]:
    return [
        FeatureImportance(name="worst_radius", importance=0.15, direction="positive", mean_shap=0.12),
        FeatureImportance(name="worst_concave_points", importance=0.10, direction="positive", mean_shap=0.08),
        FeatureImportance(name="mean_concavity", importance=0.07, direction="positive", mean_shap=0.05),
        FeatureImportance(name="mean_smoothness", importance=0.04, direction="negative", mean_shap=-0.03),
        FeatureImportance(name="mean_texture", importance=0.02, direction="positive", mean_shap=0.01),
    ]


def _make_pdp_result() -> PartialDependenceResult:
    return PartialDependenceResult(
        feature_name="mean radius",
        feature_values=[6.98, 10.0, 15.0, 20.0, 28.11],
        predictions=[0.12, 0.25, 0.55, 0.78, 0.89],
        ice_curves=[],  # ICE curves not needed for narrator tests
        feature_min=6.98,
        feature_max=28.11,
        prediction_min=0.12,
        prediction_max=0.89,
    )


def _make_dataset_description() -> DatasetDescription:
    return DatasetDescription(
        n_samples=114,
        n_features=30,
        feature_names=["worst_radius", "mean_texture"],
        class_distribution={"malignant": 43, "benign": 71},
        missing_values=0,
        feature_stats={
            "worst_radius": {"mean": 16.27, "std": 4.83, "min": 7.93, "max": 36.04},
            "mean_texture": {"mean": 19.29, "std": 4.30, "min": 9.71, "max": 39.28},
        },
    )


# --- narrate_prediction tests (Day 1, kept for regression) ---


class TestNarratePrediction:
    """Tests aligned with Day 1 acceptance scenario D1-S2."""

    def test_narrative_is_string(self):
        result = _make_shap_result()
        narrative = narrate_prediction(result)
        assert isinstance(narrative, str)

    def test_narrative_has_minimum_length(self):
        result = _make_shap_result()
        narrative = narrate_prediction(result)
        assert len(narrative) > 50, f"Narrative too short ({len(narrative)} chars)"

    def test_narrative_contains_classification(self):
        result = _make_shap_result()
        narrative = narrate_prediction(result)
        assert "classified this sample as" in narrative

    def test_narrative_contains_probability(self):
        result = _make_shap_result()
        narrative = narrate_prediction(result)
        assert "probability" in narrative

    def test_narrative_mentions_top_3_features(self):
        result = _make_shap_result()
        narrative = narrate_prediction(result, top_n=3)
        assert "worst_radius" in narrative
        assert "worst_concave_points" in narrative
        assert "mean_concavity" in narrative

    def test_each_feature_has_direction(self):
        result = _make_shap_result()
        narrative = narrate_prediction(result, top_n=3)
        assert "pushing toward" in narrative or "pushing away from" in narrative

    def test_each_feature_has_magnitude(self):
        result = _make_shap_result()
        narrative = narrate_prediction(result, top_n=3)
        assert "0.28" in narrative

    def test_narrative_is_deterministic(self):
        result = _make_shap_result()
        narrative_1 = narrate_prediction(result)
        narrative_2 = narrate_prediction(result)
        assert narrative_1 == narrative_2

    def test_opposing_factor_mentioned(self):
        result = _make_shap_result()
        narrative = narrate_prediction(result, top_n=3)
        assert "opposing" in narrative.lower()
        assert "mean_smoothness" in narrative


# --- narrate_model_summary tests (Day 2, D2-S1) ---


class TestNarrateModelSummary:
    """Tests aligned with scenario D2-S1."""

    def test_contains_model_type(self):
        narrative = narrate_model_summary(_make_model_summary())
        assert "XGBClassifier" in narrative

    def test_contains_accuracy(self):
        narrative = narrate_model_summary(_make_model_summary())
        assert "95.6%" in narrative

    def test_contains_feature_count(self):
        narrative = narrate_model_summary(_make_model_summary())
        assert "30 features" in narrative

    def test_contains_top_features(self):
        narrative = narrate_model_summary(_make_model_summary())
        assert "worst_radius" in narrative
        assert "worst_concave_points" in narrative
        assert "mean_concavity" in narrative

    def test_contains_target_names(self):
        narrative = narrate_model_summary(_make_model_summary())
        assert "malignant" in narrative
        assert "benign" in narrative

    def test_is_deterministic(self):
        n1 = narrate_model_summary(_make_model_summary())
        n2 = narrate_model_summary(_make_model_summary())
        assert n1 == n2


# --- narrate_feature_comparison tests (Day 2, D2-S2) ---


class TestNarrateFeatureComparison:
    """Tests aligned with scenario D2-S2."""

    def test_contains_ranked_features(self):
        narrative = narrate_feature_comparison(_make_feature_importances())
        assert "worst_radius" in narrative
        assert "#1" in narrative

    def test_features_appear_in_order(self):
        narrative = narrate_feature_comparison(_make_feature_importances())
        pos_radius = narrative.index("worst_radius")
        pos_concave = narrative.index("worst_concave_points")
        assert pos_radius < pos_concave, "Features should appear in importance order"

    def test_contains_magnitude(self):
        narrative = narrate_feature_comparison(_make_feature_importances())
        assert "0.15" in narrative  # worst_radius importance

    def test_contains_direction(self):
        narrative = narrate_feature_comparison(_make_feature_importances())
        assert "increase risk" in narrative or "decrease risk" in narrative

    def test_uses_comparative_language(self):
        narrative = narrate_feature_comparison(_make_feature_importances())
        # The first feature is 1.5x more important → should mention this
        assert "more influential" in narrative or "Ranked" in narrative

    def test_is_deterministic(self):
        n1 = narrate_feature_comparison(_make_feature_importances())
        n2 = narrate_feature_comparison(_make_feature_importances())
        assert n1 == n2


# --- narrate_partial_dependence tests (Day 2, D2-S3) ---


class TestNarratePartialDependence:
    """Tests aligned with scenario D2-S3."""

    def test_describes_relationship(self):
        narrative = narrate_partial_dependence(_make_pdp_result())
        assert "mean radius" in narrative
        assert "increases" in narrative or "decreases" in narrative

    def test_contains_feature_range(self):
        narrative = narrate_partial_dependence(_make_pdp_result())
        assert "6.98" in narrative
        assert "28.11" in narrative

    def test_contains_prediction_range(self):
        narrative = narrate_partial_dependence(_make_pdp_result())
        assert "12.0%" in narrative or "12%" in narrative
        assert "89.0%" in narrative or "89%" in narrative

    def test_mentions_steepest_change(self):
        narrative = narrate_partial_dependence(_make_pdp_result())
        assert "steepest" in narrative

    def test_is_deterministic(self):
        n1 = narrate_partial_dependence(_make_pdp_result())
        n2 = narrate_partial_dependence(_make_pdp_result())
        assert n1 == n2


# --- narrate_dataset tests (Day 2, D2-S4) ---


class TestNarrateDataset:
    """Tests aligned with scenario D2-S4."""

    def test_contains_sample_count(self):
        narrative = narrate_dataset(_make_dataset_description())
        assert "114 samples" in narrative

    def test_contains_feature_count(self):
        narrative = narrate_dataset(_make_dataset_description())
        assert "30 features" in narrative

    def test_contains_class_distribution(self):
        narrative = narrate_dataset(_make_dataset_description())
        assert "malignant" in narrative
        assert "benign" in narrative
        assert "43" in narrative
        assert "71" in narrative

    def test_mentions_missing_values(self):
        narrative = narrate_dataset(_make_dataset_description())
        assert "missing" in narrative.lower()

    def test_zero_missing_says_no_missing(self):
        narrative = narrate_dataset(_make_dataset_description())
        assert "no missing" in narrative.lower()

    def test_is_deterministic(self):
        n1 = narrate_dataset(_make_dataset_description())
        n2 = narrate_dataset(_make_dataset_description())
        assert n1 == n2
