"""Tests for Kedro adapter node functions in isolation (Scenario Group 7).

No Kedro runner needed — we call node functions directly with pandas DataFrames.
"""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from xai_toolkit.kedro_adapter.nodes import (
    _coerce_to_pandas,
    batch_explain_node,
    detect_drift_node,
    model_summary_node,
    sample_indices_node,
)
from xai_toolkit.result_store import StoredDriftResult, StoredExplanation, StoredModelSummary
from xai_toolkit.schemas import ShapResult


@pytest.fixture
def binary_data():
    """Simple binary classification dataset."""
    X, y = make_classification(
        n_samples=100, n_features=5, n_informative=3,
        random_state=42,
    )
    feature_names = [f"feat_{i}" for i in range(5)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")
    return X_df, y_series, feature_names


@pytest.fixture
def trained_rf(binary_data):
    X, y, _ = binary_data
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def metadata(binary_data):
    _, _, feature_names = binary_data
    return {
        "model_type": "RandomForestClassifier",
        "feature_names": feature_names,
        "target_names": ["benign", "malignant"],
        "accuracy": 0.95,
        "n_train_samples": 100,
        "n_test_samples": 100,
    }


@pytest.fixture
def params():
    return {
        "model_id": "test_model",
        "run_id": "test-run-001",
        "n_samples": 5,
        "strategy": "random",
        "random_state": 42,
        "target_names": ["benign", "malignant"],
    }


class TestSampleIndicesNode:
    def test_returns_correct_count(self, trained_rf, binary_data, params):
        X, _, _ = binary_data
        indices = sample_indices_node(trained_rf, X, params)
        assert len(indices) == 5
        assert all(isinstance(i, int) for i in indices)
        assert all(0 <= i < len(X) for i in indices)

    def test_reproducible(self, trained_rf, binary_data, params):
        X, _, _ = binary_data
        r1 = sample_indices_node(trained_rf, X, params)
        r2 = sample_indices_node(trained_rf, X, params)
        assert r1 == r2


class TestBatchExplainNode:
    """S7.1 — batch_explain node produces correct DataFrame."""

    def test_produces_correct_schema(self, trained_rf, binary_data, params):
        X, y, _ = binary_data
        indices = [0, 1, 2, 3, 4]
        df = batch_explain_node(trained_rf, X, y, X, indices, params)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

        # Validate schema matches StoredExplanation
        required_cols = set(StoredExplanation.model_fields.keys())
        assert required_cols.issubset(set(df.columns))

    def test_narratives_non_empty(self, trained_rf, binary_data, params):
        X, y, _ = binary_data
        df = batch_explain_node(trained_rf, X, y, X, [0, 1], params)
        for narrative in df["narrative"]:
            assert isinstance(narrative, str)
            assert len(narrative) > 10

    def test_model_id_and_run_id(self, trained_rf, binary_data, params):
        X, y, _ = binary_data
        df = batch_explain_node(trained_rf, X, y, X, [0], params)
        assert df.iloc[0]["model_id"] == "test_model"
        assert df.iloc[0]["run_id"] == "test-run-001"

    def test_shap_values_are_valid_json(self, trained_rf, binary_data, params):
        X, y, _ = binary_data
        df = batch_explain_node(trained_rf, X, y, X, [0], params)
        shap_dict = json.loads(df.iloc[0]["shap_values"])
        assert isinstance(shap_dict, dict)
        assert len(shap_dict) == 5  # 5 features

    def test_without_X_train(self, trained_rf, binary_data, params):
        X, y, _ = binary_data
        df = batch_explain_node(trained_rf, X, y, None, [0], params)
        assert len(df) == 1

    def test_data_hash_present(self, trained_rf, binary_data, params):
        X, y, _ = binary_data
        df = batch_explain_node(trained_rf, X, y, X, [0], params)
        assert len(df.iloc[0]["data_hash"]) == 64

    def test_raises_on_partial_batch_shap_to_prevent_misalignment(
        self,
        monkeypatch: pytest.MonkeyPatch,
        trained_rf,
        binary_data,
        params,
    ):
        X, y, _ = binary_data

        def _partial(*args, **kwargs):
            return [
                ShapResult(
                    prediction=1,
                    prediction_label="malignant",
                    probability=0.9,
                    base_value=0.0,
                    shap_values={
                        "feat_0": 0.1,
                        "feat_1": 0.0,
                        "feat_2": 0.0,
                        "feat_3": 0.0,
                        "feat_4": 0.0,
                    },
                    feature_values={
                        "feat_0": 0.1,
                        "feat_1": 0.1,
                        "feat_2": 0.1,
                        "feat_3": 0.1,
                        "feat_4": 0.1,
                    },
                    feature_names=["feat_0", "feat_1", "feat_2", "feat_3", "feat_4"],
                )
            ]

        monkeypatch.setattr("xai_toolkit.kedro_adapter.nodes.compute_shap_values_batch", _partial)
        with pytest.raises(RuntimeError, match="fewer results"):
            batch_explain_node(trained_rf, X, y, X, [0, 1], params)


class TestDetectDriftNode:
    """S7.2 — detect_drift node produces correct DataFrame."""

    def test_produces_correct_schema(self, binary_data, params):
        X, _, _ = binary_data
        # Use same data for reference and current (no real drift)
        df = detect_drift_node(X, X, params)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5  # one row per feature

        required_cols = set(StoredDriftResult.model_fields.keys())
        assert required_cols.issubset(set(df.columns))

    def test_detects_drift_when_present(self, binary_data, params):
        X, _, _ = binary_data
        # Shift one feature significantly
        X_shifted = X.copy()
        X_shifted["feat_0"] = X_shifted["feat_0"] + 10
        df = detect_drift_node(X, X_shifted, params)
        drifted = df[df["drift_detected"] == True]
        assert len(drifted) >= 1

    def test_overall_narrative_present(self, binary_data, params):
        X, _, _ = binary_data
        df = detect_drift_node(X, X, params)
        for narrative in df["overall_narrative"]:
            assert isinstance(narrative, str)
            assert len(narrative) > 10

    def test_per_feature_narrative(self, binary_data, params):
        X, _, _ = binary_data
        df = detect_drift_node(X, X, params)
        for narrative in df["narrative"]:
            assert isinstance(narrative, str)
            assert len(narrative) > 10


class TestModelSummaryNode:
    def test_produces_correct_schema(self, trained_rf, binary_data, metadata, params):
        X, _, _ = binary_data
        df = model_summary_node(trained_rf, X, metadata, params)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5  # one row per feature

        required_cols = set(StoredModelSummary.model_fields.keys())
        assert required_cols.issubset(set(df.columns))

    def test_ranks_are_sequential(self, trained_rf, binary_data, metadata, params):
        X, _, _ = binary_data
        df = model_summary_node(trained_rf, X, metadata, params)
        ranks = df["rank"].tolist()
        assert ranks == list(range(1, len(ranks) + 1))

    def test_importances_sorted_descending(self, trained_rf, binary_data, metadata, params):
        X, _, _ = binary_data
        df = model_summary_node(trained_rf, X, metadata, params)
        importances = df["importance"].tolist()
        assert importances == sorted(importances, reverse=True)


class TestCoerceToPandas:
    def test_pandas_passthrough(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = _coerce_to_pandas(df, "test")
        assert result is df

    def test_rejects_unknown_type(self):
        with pytest.raises(TypeError, match="must be a pandas or Spark"):
            _coerce_to_pandas([1, 2, 3], "test")


class TestModelValidationInNodes:
    """S7.4 — Model doesn't satisfy ClassifierProtocol."""

    def test_non_classifier_fails(self, binary_data, params):
        X, y, _ = binary_data

        class BadModel:
            def predict(self, X):
                return np.zeros(len(X))

        with pytest.raises(ValueError, match="predict_proba"):
            sample_indices_node(BadModel(), X, params)

    def test_non_classifier_fails_in_batch_explain(self, binary_data, params):
        X, y, _ = binary_data

        class BadModel:
            def predict(self, X):
                return np.zeros(len(X))

        with pytest.raises(ValueError, match="predict_proba"):
            batch_explain_node(BadModel(), X, y, X, [0], params)
