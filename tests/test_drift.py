"""Tests for drift detection computation (drift.py).

Covers scenarios A1–A6, B1–B5, C1–C5 from the BDD scenarios.
"""

import logging

import numpy as np
import pandas as pd
import pytest

from xai_toolkit.drift import (
    compute_distribution_summary,
    compute_psi,
    detect_drift,
    detect_feature_drift,
)
from xai_toolkit.schemas import (
    DatasetDriftResult,
    DistributionSummary,
    FeatureDriftResult,
)


# ---------------------------------------------------------------------------
# Helpers: deterministic test data
# ---------------------------------------------------------------------------


def _make_numeric_reference(n: int = 500, seed: int = 42) -> pd.Series:
    """Reference distribution: N(0, 1)."""
    rng = np.random.RandomState(seed)
    return pd.Series(rng.randn(n), name="feature")


def _make_numeric_shifted(n: int = 500, seed: int = 99) -> pd.Series:
    """Shifted distribution: N(2, 1) — significant drift from N(0,1)."""
    rng = np.random.RandomState(seed)
    return pd.Series(rng.randn(n) + 2, name="feature")


def _make_categorical_reference(n: int = 500, seed: int = 42) -> pd.Series:
    """Reference categorical: A=50%, B=30%, C=20%."""
    rng = np.random.RandomState(seed)
    choices = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
    return pd.Series(choices, name="category")


def _make_categorical_shifted(n: int = 500, seed: int = 99) -> pd.Series:
    """Shifted categorical: A=20%, B=50%, C=30%."""
    rng = np.random.RandomState(seed)
    choices = rng.choice(["A", "B", "C"], size=n, p=[0.2, 0.5, 0.3])
    return pd.Series(choices, name="category")


# ---------------------------------------------------------------------------
# Group B5: DistributionSummary
# ---------------------------------------------------------------------------


class TestDistributionSummary:
    """B5: DistributionSummary correctness."""

    def test_known_values(self):
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        summary = compute_distribution_summary(series)

        assert summary.mean == 3.0
        assert summary.median == 3.0
        assert summary.min == 1.0
        assert summary.max == 5.0
        assert summary.n_samples == 5
        assert summary.quantile_25 == 2.0
        assert summary.quantile_75 == 4.0
        assert round(summary.std, 2) == round(float(series.std()), 2)


# ---------------------------------------------------------------------------
# Group A: Dataset-level drift
# ---------------------------------------------------------------------------


class TestDetectDrift:
    """A1–A6: Dataset-level drift detection."""

    def test_mixed_numeric_categorical(self):
        """A1, A4: Mixed features, partial drift."""
        ref = pd.DataFrame({
            "num_stable_1": _make_numeric_reference(),
            "num_stable_2": _make_numeric_reference(seed=10),
            "num_shifted": _make_numeric_reference(),
            "cat_shifted": _make_categorical_reference(),
        })
        cur = pd.DataFrame({
            "num_stable_1": _make_numeric_reference(seed=11),
            "num_stable_2": _make_numeric_reference(seed=12),
            "num_shifted": _make_numeric_shifted(),
            "cat_shifted": _make_categorical_shifted(),
        })

        result = detect_drift(ref, cur)

        assert isinstance(result, DatasetDriftResult)
        assert result.n_features == 4
        assert result.n_drifted >= 2  # at least the shifted features
        assert result.share_drifted > 0.0

        # Check test type selection (A4)
        by_name = {f.feature_name: f for f in result.features}
        assert by_name["num_shifted"].test_name == "psi"
        assert by_name["num_shifted"].drift_detected is True
        assert by_name["cat_shifted"].test_name == "chi_squared"
        assert by_name["cat_shifted"].drift_detected is True

        # Numeric features should have KS p_value as supporting evidence
        assert by_name["num_shifted"].p_value is not None

    def test_no_drift(self):
        """A2: Same distribution → no drift."""
        ref = pd.DataFrame({
            "f1": _make_numeric_reference(seed=42),
            "f2": _make_numeric_reference(seed=43),
            "f3": _make_numeric_reference(seed=44),
        })
        # Current from same distribution (different seed but same params)
        cur = pd.DataFrame({
            "f1": _make_numeric_reference(seed=52),
            "f2": _make_numeric_reference(seed=53),
            "f3": _make_numeric_reference(seed=54),
        })

        result = detect_drift(ref, cur)

        assert result.n_drifted == 0
        assert result.share_drifted == 0.0
        assert result.overall_severity == "none"
        assert all(not f.drift_detected for f in result.features)

    def test_all_features_drifted(self):
        """A3: All features severely drifted."""
        ref = pd.DataFrame({
            "f1": _make_numeric_reference(),
            "f2": _make_numeric_reference(seed=10),
        })
        cur = pd.DataFrame({
            "f1": _make_numeric_shifted(),
            "f2": _make_numeric_shifted(seed=88),
        })

        result = detect_drift(ref, cur)

        assert result.n_drifted == 2
        assert result.share_drifted == 1.0
        assert result.overall_severity in ("moderate", "severe")

    def test_single_feature_dataframe(self):
        """A6: Single-feature DataFrame."""
        ref = pd.DataFrame({"only_feature": _make_numeric_reference()})
        cur = pd.DataFrame({"only_feature": _make_numeric_shifted()})

        result = detect_drift(ref, cur)

        assert result.n_features == 1
        assert len(result.features) == 1

    def test_no_overlapping_features_raises_value_error(self):
        """Dataset-level drift requires at least one shared feature."""
        ref = pd.DataFrame({"f_ref": _make_numeric_reference()})
        cur = pd.DataFrame({"f_cur": _make_numeric_reference(seed=99)})

        with pytest.raises(
            ValueError,
            match="No overlapping features available for drift detection",
        ):
            detect_drift(ref, cur)


# ---------------------------------------------------------------------------
# Group A5: PSI severity thresholds
# ---------------------------------------------------------------------------


class TestPSISeverityThresholds:
    """A5: PSI threshold boundaries."""

    def test_psi_none(self):
        """PSI < 0.1 → none."""
        # Same distribution, PSI should be near 0
        ref = _make_numeric_reference(n=1000)
        cur = _make_numeric_reference(n=1000, seed=99)
        psi = compute_psi(ref.values, cur.values)
        assert psi < 0.1

    def test_psi_computation_positive(self):
        """PSI for shifted distribution is positive."""
        ref = _make_numeric_reference(n=1000)
        cur = _make_numeric_shifted(n=1000)
        psi = compute_psi(ref.values, cur.values)
        assert psi > 0.0


# ---------------------------------------------------------------------------
# Group B: Feature-level drift
# ---------------------------------------------------------------------------


class TestDetectFeatureDrift:
    """B1–B4: Feature-level drift detection."""

    def test_numeric_significant_drift(self):
        """B1: N(0,1) vs N(2,1) → drift detected."""
        ref = _make_numeric_reference()
        cur = _make_numeric_shifted()

        result = detect_feature_drift(ref, cur, "temperature")

        assert isinstance(result, FeatureDriftResult)
        assert result.drift_detected is True
        assert result.test_name == "psi"
        assert result.p_value is not None
        assert abs(result.reference_summary.mean - 0.0) < 0.2
        assert abs(result.current_summary.mean - 2.0) < 0.2

    def test_numeric_no_drift(self):
        """B2: Same distribution → no drift."""
        ref = _make_numeric_reference()
        cur = _make_numeric_reference(seed=99)

        result = detect_feature_drift(ref, cur, "stable_feature")

        assert result.drift_detected is False
        assert result.severity == "none"

    def test_categorical_drift(self):
        """B3: Shifted categorical proportions → drift."""
        ref = _make_categorical_reference()
        cur = _make_categorical_shifted()

        result = detect_feature_drift(ref, cur, "region")

        assert result.test_name == "chi_squared"
        assert result.drift_detected is True

    def test_categorical_unknown_category_dropped(self, caplog):
        """B4: Unknown category in current is dropped with warning."""
        ref = _make_categorical_reference()
        # Add a new category D to current
        rng = np.random.RandomState(99)
        choices = rng.choice(["A", "B", "C", "D"], size=500, p=[0.2, 0.3, 0.2, 0.3])
        cur = pd.Series(choices, name="category")

        with caplog.at_level(logging.WARNING):
            result = detect_feature_drift(ref, cur, "region")

        assert result.test_name == "chi_squared"
        assert any("dropping" in msg.lower() for msg in caplog.messages)


# ---------------------------------------------------------------------------
# Group C: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """C1–C5: Edge case handling."""

    def test_constant_feature(self):
        """C1: All same values → no drift, no error."""
        ref = pd.Series([5.0] * 100)
        cur = pd.Series([5.0] * 100)

        result = detect_feature_drift(ref, cur, "constant")

        assert result.drift_detected is False
        assert result.severity == "none"

    def test_constant_reference_shifted_current(self):
        """C2: Constant reference, different constant current → drift."""
        ref = pd.Series([5.0] * 100)
        cur = pd.Series([10.0] * 100)

        result = detect_feature_drift(ref, cur, "shifted_constant")

        assert result.drift_detected is True

    def test_nan_few(self):
        """C3: Few NaNs dropped, result still valid."""
        ref = _make_numeric_reference()
        cur = _make_numeric_reference(seed=99)
        # Insert 5% NaN
        ref.iloc[:25] = np.nan

        result = detect_feature_drift(ref, cur, "nan_feature")

        assert isinstance(result, FeatureDriftResult)
        assert result.reference_summary.n_samples == 475  # 500 - 25

    def test_nan_high_rate_warning(self, caplog):
        """C4: >70% NaN triggers warning."""
        ref = pd.Series([np.nan] * 400 + [1.0, 2.0, 3.0] * 34)  # ~80% NaN
        cur = _make_numeric_reference(n=500, seed=99)

        with caplog.at_level(logging.WARNING):
            result = detect_feature_drift(ref, cur, "mostly_nan")

        assert any("high nan rate" in msg.lower() for msg in caplog.messages)
        assert isinstance(result, FeatureDriftResult)

    def test_psi_skipped_low_cardinality(self):
        """C5: <10 unique values → KS fallback."""
        # Only 5 unique values
        rng = np.random.RandomState(42)
        ref = pd.Series(rng.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=200))
        cur = pd.Series(rng.choice([1.0, 2.0, 3.0, 4.0, 5.0], size=200))

        result = detect_feature_drift(ref, cur, "low_card")

        assert result.test_name == "ks"  # PSI skipped

    def test_all_nan_current_raises_value_error(self):
        """All-NaN current data should fail with a clear error."""
        ref = _make_numeric_reference()
        cur = pd.Series([np.nan] * len(ref))

        with pytest.raises(
            ValueError,
            match="no usable current samples after NaN filtering",
        ):
            detect_feature_drift(ref, cur, "all_nan_current")
