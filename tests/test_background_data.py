"""Tests for Phase 2: Training data as SHAP background (TD-14 fix).

Verifies that:
- compute_shap_values() uses background_data when provided
- Fallback to test data when background_data is None
- Registry loads X_train from disk when available
- Registry handles missing X_train gracefully

Adapted from Tamas's explainability_node() which correctly uses X_train
for background distribution and X_test for samples to explain.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from xai_toolkit.explainers import (
    compute_global_feature_importance,
    compute_shap_values,
)
from xai_toolkit.registry import ModelRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def toy_data():
    """Create a toy dataset with distinct train/test splits."""
    rng = np.random.RandomState(42)
    X = pd.DataFrame(rng.randn(200, 4), columns=["a", "b", "c", "d"])
    y = pd.Series((X["a"] + X["b"] > 0).astype(int), name="target")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42,
    )
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


@pytest.fixture()
def multiclass_data():
    """Create a simple multiclass dataset to verify the narrowed contract."""
    from sklearn.datasets import load_iris

    data = load_iris(as_frame=True)
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y,
    )
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# compute_shap_values background_data tests
# ---------------------------------------------------------------------------


class TestShapBackgroundData:
    """Test that background_data parameter is used correctly."""

    def test_with_background_data_succeeds(self, toy_data) -> None:
        """SHAP computation works when background_data is provided."""
        model, X_train, X_test, _, _ = toy_data
        result = compute_shap_values(
            model=model, X=X_test, sample_index=0,
            background_data=X_train,
        )
        assert result is not None
        assert len(result.shap_values) == 4

    def test_without_background_data_succeeds(self, toy_data) -> None:
        """SHAP computation works with fallback (background_data=None)."""
        model, _, X_test, _, _ = toy_data
        result = compute_shap_values(
            model=model, X=X_test, sample_index=0,
            background_data=None,
        )
        assert result is not None
        assert len(result.shap_values) == 4

    def test_background_data_is_actually_used(self, toy_data) -> None:
        """When background_data is provided, it should be sampled, not X."""
        model, X_train, X_test, _, _ = toy_data
        with patch("xai_toolkit.explainers.shap.Explainer") as mock_explainer:
            # Configure mock to look real enough
            mock_exp_instance = mock_explainer.return_value
            mock_result = type("Obj", (), {
                "values": np.zeros((1, 4, 2)),
                "base_values": np.array([[0.5, 0.5]]),
            })()
            mock_exp_instance.return_value = mock_result

            compute_shap_values(
                model=model, X=X_test, sample_index=0,
                background_data=X_train,
            )

            # The Explainer should have been called with background from X_train
            call_args = mock_explainer.call_args
            background_used = call_args[0][1]  # second positional arg
            assert len(background_used) <= min(50, len(X_train))
            # Background indices should be from X_train, not X_test
            assert all(idx in X_train.index for idx in background_used.index)

    def test_fallback_uses_test_data(self, toy_data) -> None:
        """When background_data is None, background should come from X (test)."""
        model, _, X_test, _, _ = toy_data
        with patch("xai_toolkit.explainers.shap.Explainer") as mock_explainer:
            mock_exp_instance = mock_explainer.return_value
            mock_result = type("Obj", (), {
                "values": np.zeros((1, 4, 2)),
                "base_values": np.array([[0.5, 0.5]]),
            })()
            mock_exp_instance.return_value = mock_result

            compute_shap_values(
                model=model, X=X_test, sample_index=0,
                background_data=None,
            )

            call_args = mock_explainer.call_args
            background_used = call_args[0][1]
            assert all(idx in X_test.index for idx in background_used.index)

    def test_multiclass_model_is_rejected(self, multiclass_data) -> None:
        """The current explainer contract is intentionally binary-only."""
        model, X_train, X_test, _, _ = multiclass_data
        with pytest.raises(ValueError, match="Only binary classifiers"):
            compute_shap_values(
                model=model,
                X=X_test,
                sample_index=0,
                background_data=X_train,
            )

    def test_background_column_mismatch_is_rejected(self, toy_data) -> None:
        """Background data must match the explained feature schema exactly."""
        model, X_train, X_test, _, _ = toy_data
        mismatched_background = X_train[["b", "a", "c", "d"]]
        with pytest.raises(ValueError, match="background_data columns must exactly match"):
            compute_shap_values(
                model=model,
                X=X_test,
                sample_index=0,
                background_data=mismatched_background,
            )


# ---------------------------------------------------------------------------
# compute_global_feature_importance background_data tests
# ---------------------------------------------------------------------------


class TestGlobalImportanceBackgroundData:
    """Test background_data in global feature importance."""

    def test_with_background_data(self, toy_data) -> None:
        model, X_train, X_test, _, _ = toy_data
        result = compute_global_feature_importance(
            model=model, X=X_test,
            background_data=X_train,
        )
        assert len(result) == 4
        assert all(f.importance >= 0 for f in result)

    def test_without_background_data(self, toy_data) -> None:
        model, _, X_test, _, _ = toy_data
        result = compute_global_feature_importance(
            model=model, X=X_test,
            background_data=None,
        )
        assert len(result) == 4

    def test_multiclass_model_is_rejected(self, multiclass_data) -> None:
        model, X_train, X_test, _, _ = multiclass_data
        with pytest.raises(ValueError, match="Only binary classifiers"):
            compute_global_feature_importance(
                model=model,
                X=X_test,
                background_data=X_train,
            )


# ---------------------------------------------------------------------------
# Registry X_train loading tests
# ---------------------------------------------------------------------------


class TestRegistryXTrain:
    """Test that ModelRegistry loads X_train when available."""

    def test_registry_loads_x_train(self) -> None:
        """Registry should load X_train from disk when the file exists."""
        from xai_toolkit.cli import DATA_DIR, MODELS_DIR

        registry = ModelRegistry()
        registry.load_from_disk("xgboost_breast_cancer", MODELS_DIR, DATA_DIR)
        entry = registry.get("xgboost_breast_cancer")

        # X_train should be loaded since we saved it in train_toy_model.py
        assert entry.X_train is not None
        assert isinstance(entry.X_train, pd.DataFrame)
        assert len(entry.X_train) > 0
        # X_train should have same columns as X_test
        assert list(entry.X_train.columns) == list(entry.X_test.columns)

    def test_registry_handles_missing_x_train(self, tmp_path) -> None:
        """Registry should set X_train=None when train file doesn't exist."""
        import json
        import joblib
        from sklearn.ensemble import RandomForestClassifier

        # Create minimal model artifacts without X_train
        models_dir = tmp_path / "models"
        data_dir = tmp_path / "data"
        models_dir.mkdir()
        data_dir.mkdir()

        rng = np.random.RandomState(42)
        X = pd.DataFrame(rng.randn(20, 2), columns=["f1", "f2"])
        y = pd.Series((X["f1"] > 0).astype(int))

        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)

        joblib.dump(model, models_dir / "test_model.joblib")
        X.to_csv(data_dir / "test_test_X.csv", index=False)
        y.to_csv(data_dir / "test_test_y.csv", index=False)

        meta = {
            "model_id": "test_model",
            "dataset_name": "test",
            "feature_names": ["f1", "f2"],
            "target_names": ["class_0", "class_1"],
        }
        with open(models_dir / "test_model_meta.json", "w") as f:
            json.dump(meta, f)

        registry = ModelRegistry()
        registry.load_from_disk("test_model", models_dir, data_dir)
        entry = registry.get("test_model")
        assert entry.X_train is None

    def test_registry_rejects_feature_name_mismatch(self, tmp_path) -> None:
        """Loaded CSV columns must agree with metadata feature_names exactly."""
        import json
        import joblib

        models_dir = tmp_path / "models"
        data_dir = tmp_path / "data"
        models_dir.mkdir()
        data_dir.mkdir()

        rng = np.random.RandomState(42)
        X = pd.DataFrame(rng.randn(20, 2), columns=["f1", "f2"])
        y = pd.Series((X["f1"] > 0).astype(int))

        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)

        joblib.dump(model, models_dir / "test_model.joblib")
        X.to_csv(data_dir / "test_test_X.csv", index=False)
        y.to_csv(data_dir / "test_test_y.csv", index=False)

        meta = {
            "model_id": "test_model",
            "dataset_name": "test",
            "feature_names": ["f2", "f1"],
            "target_names": ["class_0", "class_1"],
        }
        with open(models_dir / "test_model_meta.json", "w") as f:
            json.dump(meta, f)

        registry = ModelRegistry()
        with pytest.raises(ValueError, match="must exactly match metadata feature_names"):
            registry.load_from_disk("test_model", models_dir, data_dir)
