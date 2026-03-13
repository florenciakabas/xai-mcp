"""Tests for pipeline bridge functions — reading pre-computed SHAP artifacts.

These functions bridge between the Kedro explainability pipeline's output
(numpy arrays + JSON metadata on disk) and our MCP toolkit's Pydantic models.

The Kedro pipeline computes SHAP values in batch (expensive, scheduled).
Our MCP toolkit serves explanations interactively (cheap, on-demand).
The bridge lets us read what the pipeline already computed, rather than
recomputing SHAP on the fly.

Test strategy:
  - Create realistic pipeline artifacts in a temporary directory
  - Verify our bridge functions can read them into our Pydantic schemas
  - Confirm error handling when artifacts are missing or malformed
"""

import json

import numpy as np
import pytest

from xai_toolkit.explainers import (
    load_shap_from_pipeline,
    load_global_importance_from_pipeline,
)
from xai_toolkit.schemas import FeatureImportance, ShapResult


# ---------------------------------------------------------------------------
# Fixtures — simulate the Kedro pipeline's output on disk
# ---------------------------------------------------------------------------

@pytest.fixture
def pipeline_artifacts(tmp_path):
    """Create a realistic set of pipeline output artifacts.

    Mimics what `explainability_node` in the colleague's Kedro pipeline
    writes to `save_dir`:
      - shap_values.npy          (n_samples, n_features)
      - shap_expected_value.npy  (n_samples,) or scalar
      - shap_metadata.json       with feature_names, detected_type, etc.

    Uses 5 samples × 3 features for fast tests.
    """
    n_samples = 5
    feature_names = ["pressure", "temperature", "flow_rate"]
    n_features = len(feature_names)

    # SHAP values: (n_samples, n_features)
    shap_values = np.array([
        [0.10, -0.05, 0.03],   # sample 0
        [0.20, -0.15, 0.08],   # sample 1
        [-0.12, 0.25, -0.01],  # sample 2
        [0.05, 0.02, 0.30],    # sample 3
        [-0.08, 0.10, -0.20],  # sample 4
    ])
    np.save(tmp_path / "shap_values.npy", shap_values)

    # Expected value (base value): scalar broadcast to per-sample array
    expected_value = np.full(n_samples, 0.45)
    np.save(tmp_path / "shap_expected_value.npy", expected_value)

    # Metadata (matches the colleague's _save_metadata schema)
    metadata = {
        "feature_names": feature_names,
        "n_rows_explained": n_samples,
        "n_features": n_features,
        "expected_value_saved": "shap_expected_value.npy",
        "shap_saved_paths": ["shap_values.npy"],
        "detected_type": "xgboost",
        "explainer_type": "tree",
    }
    with open(tmp_path / "shap_metadata.json", "w") as f:
        json.dump(metadata, f)

    return tmp_path, shap_values, expected_value, feature_names


# ---------------------------------------------------------------------------
# Tests: load_shap_from_pipeline (single-sample extraction)
# ---------------------------------------------------------------------------

class TestLoadShapFromPipeline:
    """Tests for reading a single sample's SHAP values from pipeline artifacts."""

    def test_returns_shap_result(self, pipeline_artifacts):
        """Bridge function returns our ShapResult schema."""
        artifacts_dir, _, _, _ = pipeline_artifacts
        result = load_shap_from_pipeline(artifacts_dir, sample_index=0)
        assert isinstance(result, ShapResult)

    def test_correct_feature_names(self, pipeline_artifacts):
        """Feature names come from the pipeline metadata, not invented."""
        artifacts_dir, _, _, feature_names = pipeline_artifacts
        result = load_shap_from_pipeline(artifacts_dir, sample_index=0)
        assert result.feature_names == feature_names

    def test_correct_shap_values_for_sample(self, pipeline_artifacts):
        """SHAP values for sample 1 match what was saved to disk."""
        artifacts_dir, shap_values, _, feature_names = pipeline_artifacts
        result = load_shap_from_pipeline(artifacts_dir, sample_index=1)
        for i, name in enumerate(feature_names):
            assert result.shap_values[name] == pytest.approx(
                shap_values[1, i], abs=1e-6
            ), f"Mismatch for feature '{name}'"

    def test_correct_base_value(self, pipeline_artifacts):
        """Base value is read from the expected_value artifact."""
        artifacts_dir, _, expected_value, _ = pipeline_artifacts
        result = load_shap_from_pipeline(artifacts_dir, sample_index=0)
        assert result.base_value == pytest.approx(expected_value[0], abs=1e-6)

    def test_sample_index_out_of_range_raises(self, pipeline_artifacts):
        """Requesting a sample beyond what the pipeline computed → IndexError."""
        artifacts_dir, _, _, _ = pipeline_artifacts
        with pytest.raises(IndexError, match="out of range"):
            load_shap_from_pipeline(artifacts_dir, sample_index=99)

    def test_negative_sample_index_raises(self, pipeline_artifacts):
        """Negative indices are not allowed."""
        artifacts_dir, _, _, _ = pipeline_artifacts
        with pytest.raises(IndexError, match="out of range"):
            load_shap_from_pipeline(artifacts_dir, sample_index=-1)

    def test_missing_metadata_raises(self, tmp_path):
        """Missing shap_metadata.json → clear FileNotFoundError."""
        np.save(tmp_path / "shap_values.npy", np.array([[0.1]]))
        with pytest.raises(FileNotFoundError, match="shap_metadata.json"):
            load_shap_from_pipeline(tmp_path, sample_index=0)

    def test_missing_shap_values_raises(self, tmp_path):
        """Missing shap_values.npy → clear FileNotFoundError."""
        metadata = {"feature_names": ["x"], "shap_saved_paths": ["shap_values.npy"]}
        with open(tmp_path / "shap_metadata.json", "w") as f:
            json.dump(metadata, f)
        with pytest.raises(FileNotFoundError, match="shap_values.npy"):
            load_shap_from_pipeline(tmp_path, sample_index=0)

    def test_multiple_shap_paths_are_rejected(self, tmp_path):
        """The narrowed contract supports only one SHAP value file."""
        metadata = {
            "feature_names": ["x"],
            "n_features": 1,
            "shap_saved_paths": ["class_0.npy", "class_1.npy"],
        }
        with open(tmp_path / "shap_metadata.json", "w") as f:
            json.dump(metadata, f)
        np.save(tmp_path / "class_0.npy", np.array([[0.1]]))
        np.save(tmp_path / "class_1.npy", np.array([[0.2]]))

        with pytest.raises(ValueError, match="Only single-output SHAP artifacts"):
            load_shap_from_pipeline(tmp_path, sample_index=0)

    def test_feature_name_width_mismatch_is_rejected(self, tmp_path):
        """Feature count must agree between metadata and SHAP array width."""
        metadata = {
            "feature_names": ["x", "y"],
            "n_features": 2,
            "shap_saved_paths": ["shap_values.npy"],
        }
        with open(tmp_path / "shap_metadata.json", "w") as f:
            json.dump(metadata, f)
        np.save(tmp_path / "shap_values.npy", np.array([[0.1]]))

        with pytest.raises(ValueError, match="SHAP array width does not match"):
            load_shap_from_pipeline(tmp_path, sample_index=0)

    def test_expected_value_shape_mismatch_is_rejected(self, pipeline_artifacts):
        """Only scalar or per-sample expected value arrays are supported."""
        artifacts_dir, _, _, _ = pipeline_artifacts
        np.save(artifacts_dir / "shap_expected_value.npy", np.array([0.1, 0.2]))

        with pytest.raises(ValueError, match="Only scalar or per-sample 1D expected value arrays are supported"):
            load_shap_from_pipeline(artifacts_dir, sample_index=0)

    def test_prediction_fields_are_placeholder(self, pipeline_artifacts):
        """Pipeline artifacts don't include predictions; fields should be
        sensible defaults since we don't have the model at read time."""
        artifacts_dir, _, _, _ = pipeline_artifacts
        result = load_shap_from_pipeline(artifacts_dir, sample_index=0)
        # prediction and probability are unknowable from SHAP alone
        assert result.prediction_label == "unknown"

    def test_deterministic_across_calls(self, pipeline_artifacts):
        """Same artifacts + same index → identical result, always."""
        artifacts_dir, _, _, _ = pipeline_artifacts
        r1 = load_shap_from_pipeline(artifacts_dir, sample_index=2)
        r2 = load_shap_from_pipeline(artifacts_dir, sample_index=2)
        assert r1 == r2


# ---------------------------------------------------------------------------
# Tests: load_global_importance_from_pipeline
# ---------------------------------------------------------------------------

class TestLoadGlobalImportanceFromPipeline:
    """Tests for computing global feature importance from pipeline artifacts."""

    def test_returns_list_of_feature_importance(self, pipeline_artifacts):
        """Returns a list of our FeatureImportance schema objects."""
        artifacts_dir, _, _, _ = pipeline_artifacts
        result = load_global_importance_from_pipeline(artifacts_dir)
        assert isinstance(result, list)
        assert all(isinstance(f, FeatureImportance) for f in result)

    def test_sorted_by_importance_descending(self, pipeline_artifacts):
        """Features are ranked highest importance first."""
        artifacts_dir, _, _, _ = pipeline_artifacts
        result = load_global_importance_from_pipeline(artifacts_dir)
        importances = [f.importance for f in result]
        assert importances == sorted(importances, reverse=True)

    def test_correct_number_of_features(self, pipeline_artifacts):
        """One entry per feature in the metadata."""
        artifacts_dir, _, _, feature_names = pipeline_artifacts
        result = load_global_importance_from_pipeline(artifacts_dir)
        assert len(result) == len(feature_names)

    def test_importance_values_are_mean_abs_shap(self, pipeline_artifacts):
        """Importance = mean(|SHAP|) across all samples, matching our convention."""
        artifacts_dir, shap_values, _, feature_names = pipeline_artifacts
        result = load_global_importance_from_pipeline(artifacts_dir)

        expected_importance = np.mean(np.abs(shap_values), axis=0)

        # Build a lookup since result is sorted by importance, not feature order
        result_dict = {f.name: f.importance for f in result}
        for i, name in enumerate(feature_names):
            assert result_dict[name] == pytest.approx(
                expected_importance[i], abs=1e-6
            ), f"Importance mismatch for '{name}'"

    def test_direction_matches_mean_signed_shap(self, pipeline_artifacts):
        """Direction reflects whether the feature's average SHAP is positive or negative."""
        artifacts_dir, shap_values, _, feature_names = pipeline_artifacts
        result = load_global_importance_from_pipeline(artifacts_dir)

        mean_signed = np.mean(shap_values, axis=0)
        result_dict = {f.name: f.direction for f in result}

        for i, name in enumerate(feature_names):
            expected_dir = "positive" if mean_signed[i] > 0 else "negative"
            assert result_dict[name] == expected_dir, (
                f"Direction mismatch for '{name}': "
                f"expected '{expected_dir}', got '{result_dict[name]}'"
            )

    def test_missing_metadata_raises(self, tmp_path):
        """Missing metadata file → FileNotFoundError."""
        np.save(tmp_path / "shap_values.npy", np.array([[0.1]]))
        with pytest.raises(FileNotFoundError, match="shap_metadata.json"):
            load_global_importance_from_pipeline(tmp_path)

    def test_multiple_shap_paths_are_rejected(self, tmp_path):
        metadata = {
            "feature_names": ["x"],
            "n_features": 1,
            "shap_saved_paths": ["class_0.npy", "class_1.npy"],
        }
        with open(tmp_path / "shap_metadata.json", "w") as f:
            json.dump(metadata, f)
        np.save(tmp_path / "class_0.npy", np.array([[0.1]]))
        np.save(tmp_path / "class_1.npy", np.array([[0.2]]))

        with pytest.raises(ValueError, match="Only single-output SHAP artifacts"):
            load_global_importance_from_pipeline(tmp_path)
