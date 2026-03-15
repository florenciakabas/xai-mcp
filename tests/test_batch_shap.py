"""Tests for batch SHAP computation (Scenario Group 2)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from xai_toolkit.explainers import compute_shap_values, compute_shap_values_batch


@pytest.fixture
def model_and_data():
    """Train a model on simple data."""
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=50, n_features=5, n_informative=3,
        random_state=42,
    )
    feature_names = [f"f_{i}" for i in range(5)]
    X_df = pd.DataFrame(X, columns=feature_names)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_df, y)
    return model, X_df


class TestBatchShap:
    """S2.1–S2.2 — Happy paths."""

    def test_batch_matches_individual(self, model_and_data):
        """S2.1 — Batch produces same results as individual calls."""
        model, X = model_and_data
        indices = [0, 1, 2]

        batch_results = compute_shap_values_batch(model, X, indices)
        assert len(batch_results) == 3

        for i, idx in enumerate(indices):
            individual = compute_shap_values(model, X, idx)
            assert batch_results[i].prediction == individual.prediction
            assert batch_results[i].probability == individual.probability
            assert batch_results[i].prediction_label == individual.prediction_label

    def test_deterministic(self, model_and_data):
        """S2.1 — Running twice gives same output."""
        model, X = model_and_data
        r1 = compute_shap_values_batch(model, X, [0, 1])
        r2 = compute_shap_values_batch(model, X, [0, 1])
        for a, b in zip(r1, r2):
            assert a.prediction == b.prediction
            assert a.probability == b.probability

    def test_batch_returns_correct_count(self, model_and_data):
        """Batch of 5 returns 5 results."""
        model, X = model_and_data
        results = compute_shap_values_batch(model, X, [0, 1, 2, 3, 4])
        assert len(results) == 5


class TestBatchShapErrors:
    """S2.3–S2.5 — Unhappy paths."""

    def test_out_of_range_fails_upfront(self, model_and_data):
        """S2.5 — sample_indices out of range raises before computation."""
        model, X = model_and_data
        with pytest.raises(IndexError, match="out of range"):
            compute_shap_values_batch(model, X, [0, 999999])

    def test_negative_index_fails_upfront(self, model_and_data):
        """Negative index raises upfront."""
        model, X = model_and_data
        with pytest.raises(IndexError, match="out of range"):
            compute_shap_values_batch(model, X, [-1, 0])

    def test_empty_indices(self, model_and_data):
        """Empty indices returns empty results."""
        model, X = model_and_data
        results = compute_shap_values_batch(model, X, [])
        assert results == []
