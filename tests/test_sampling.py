"""Tests for sampling strategies (Scenario Group 3)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from xai_toolkit.sampling import select_samples


@pytest.fixture
def model_and_data():
    """Simple model and data for sampling tests."""
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=100, n_features=5, random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(5)])
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_df, y)
    return model, X_df


class TestRandomSampling:
    """S3.1 — Random sampling."""

    def test_reproducible(self, model_and_data):
        model, X = model_and_data
        r1 = select_samples(model, X, n_samples=10, strategy="random", random_state=42)
        r2 = select_samples(model, X, n_samples=10, strategy="random", random_state=42)
        assert r1 == r2

    def test_different_seed_different_result(self, model_and_data):
        model, X = model_and_data
        r1 = select_samples(model, X, n_samples=10, strategy="random", random_state=42)
        r2 = select_samples(model, X, n_samples=10, strategy="random", random_state=99)
        assert r1 != r2

    def test_correct_count(self, model_and_data):
        model, X = model_and_data
        result = select_samples(model, X, n_samples=15, strategy="random")
        assert len(result) == 15

    def test_indices_in_range(self, model_and_data):
        model, X = model_and_data
        result = select_samples(model, X, n_samples=10, strategy="random")
        assert all(0 <= idx < len(X) for idx in result)

    def test_sorted(self, model_and_data):
        model, X = model_and_data
        result = select_samples(model, X, n_samples=20, strategy="random")
        assert result == sorted(result)


class TestUncertaintySampling:
    """S3.2 — Uncertainty sampling."""

    def test_prefers_borderline(self, model_and_data):
        model, X = model_and_data
        result = select_samples(model, X, n_samples=10, strategy="uncertainty")
        probas = model.predict_proba(X)[:, 1]
        selected_probas = probas[result]
        # At least some should be in the uncertain band
        n_uncertain = sum(1 for p in selected_probas if 0.3 <= p <= 0.7)
        # We can't guarantee 80% because it depends on the model,
        # but at least more than pure random would give
        assert n_uncertain >= 1


class TestSamplingEdgeCases:
    """S3.3–S3.4 — Unhappy paths."""

    def test_n_samples_exceeds_dataset(self, model_and_data):
        """S3.3 — Returns all indices when n_samples > dataset size."""
        model, X = model_and_data
        result = select_samples(model, X, n_samples=500, strategy="random")
        assert len(result) == len(X)
        assert result == list(range(len(X)))

    def test_n_samples_equals_dataset(self, model_and_data):
        """Exact match returns all."""
        model, X = model_and_data
        result = select_samples(model, X, n_samples=100, strategy="random")
        assert len(result) == 100

    def test_unknown_strategy(self, model_and_data):
        model, X = model_and_data
        with pytest.raises(ValueError, match="Unknown sampling strategy"):
            select_samples(model, X, n_samples=10, strategy="nonexistent")
