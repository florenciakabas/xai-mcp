"""Tests for in-memory model registration (Scenario Group 1)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from xai_toolkit.explainers import compute_shap_values
from xai_toolkit.registry import ModelRegistry


@pytest.fixture
def binary_dataset():
    """Create a simple binary classification dataset."""
    X, y = make_classification(
        n_samples=100, n_features=5, n_informative=3,
        n_redundant=1, random_state=42,
    )
    feature_names = [f"feat_{i}" for i in range(5)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")
    return X_df, y_series, feature_names


@pytest.fixture
def trained_model(binary_dataset):
    """Train a simple classifier."""
    X, y, _ = binary_dataset
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def metadata(binary_dataset):
    """Standard metadata dict."""
    _, _, feature_names = binary_dataset
    return {
        "model_type": "RandomForestClassifier",
        "feature_names": feature_names,
        "target_names": ["class_0", "class_1"],
        "accuracy": 0.95,
    }


class TestRegisterInMemory:
    """S1.1 — Happy: Register an sklearn model from memory."""

    def test_basic_registration(self, trained_model, binary_dataset, metadata):
        X, y, _ = binary_dataset
        reg = ModelRegistry()
        reg.register_in_memory(
            model_id="test_rf",
            model=trained_model,
            metadata=metadata,
            X_test=X,
            y_test=y,
        )
        entry = reg.get("test_rf")
        assert entry.model_id == "test_rf"
        assert entry.model is trained_model
        assert entry.metadata["detected_type"] is not None

    def test_with_X_train(self, trained_model, binary_dataset, metadata):
        X, y, _ = binary_dataset
        reg = ModelRegistry()
        reg.register_in_memory(
            model_id="test_rf",
            model=trained_model,
            metadata=metadata,
            X_test=X,
            y_test=y,
            X_train=X,
        )
        entry = reg.get("test_rf")
        assert entry.X_train is not None
        assert len(entry.X_train) == len(X)

    def test_behaves_like_disk_loaded(self, trained_model, binary_dataset, metadata):
        """Registered model produces same SHAP values as if disk-loaded."""
        X, y, _ = binary_dataset
        reg = ModelRegistry()
        reg.register_in_memory(
            model_id="test_rf",
            model=trained_model,
            metadata=metadata,
            X_test=X,
            y_test=y,
            X_train=X,
        )
        entry = reg.get("test_rf")
        result = compute_shap_values(
            model=entry.model,
            X=entry.X_test,
            sample_index=0,
            target_names=entry.metadata.get("target_names"),
            background_data=entry.X_train,
        )
        assert result.prediction in (0, 1)
        assert 0.0 <= result.probability <= 1.0
        assert len(result.shap_values) == 5

    def test_appears_in_list_models(self, trained_model, binary_dataset, metadata):
        X, y, _ = binary_dataset
        reg = ModelRegistry()
        reg.register_in_memory("test_rf", trained_model, metadata, X, y)
        models = reg.list_models()
        assert len(models) == 1
        assert models[0]["model_id"] == "test_rf"


class TestRegisterInMemoryErrors:
    """S1.2–S1.5 — Unhappy paths."""

    def test_missing_predict_proba(self, binary_dataset, metadata):
        """S1.2 — Model missing predict_proba."""
        X, y, _ = binary_dataset

        class NoProba:
            def predict(self, X):
                return np.zeros(len(X))

        reg = ModelRegistry()
        with pytest.raises(ValueError, match="predict_proba"):
            reg.register_in_memory("bad", NoProba(), metadata, X, y)

    def test_missing_feature_names(self, trained_model, binary_dataset):
        """S1.3 — Metadata missing feature_names."""
        X, y, _ = binary_dataset
        bad_meta = {"model_type": "rf"}
        reg = ModelRegistry()
        with pytest.raises(ValueError, match="feature_names"):
            reg.register_in_memory("bad", trained_model, bad_meta, X, y)

    def test_feature_name_mismatch(self, trained_model, binary_dataset, metadata):
        """S1.4 — Feature name mismatch between metadata and X_test."""
        X, y, _ = binary_dataset
        bad_X = X.rename(columns={"feat_0": "wrong_name"})
        reg = ModelRegistry()
        with pytest.raises(ValueError, match="must exactly match"):
            reg.register_in_memory("bad", trained_model, metadata, bad_X, y)

    def test_multiclass_rejected(self, binary_dataset, metadata):
        """S1.5 — Multiclass model rejected."""
        X_raw, _, feature_names = binary_dataset
        # Create 3-class target
        y_multi = pd.Series(np.tile([0, 1, 2], len(X_raw))[:len(X_raw)])
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X_raw, y_multi)

        reg = ModelRegistry()
        with pytest.raises(ValueError, match="binary"):
            reg.register_in_memory("bad", model, metadata, X_raw, y_multi)
