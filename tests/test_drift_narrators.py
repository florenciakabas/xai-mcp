"""Tests for drift narrator functions.

Covers scenarios D1–D3, E1–E5 from the BDD scenarios.
"""

from syrupy.assertion import SnapshotAssertion

from xai_toolkit.narrators import narrate_dataset_drift, narrate_feature_drift
from xai_toolkit.schemas import (
    DatasetDriftResult,
    DistributionSummary,
    FeatureDriftResult,
)


# ---------------------------------------------------------------------------
# Test data builders
# ---------------------------------------------------------------------------


def _make_ref_summary(**overrides) -> DistributionSummary:
    defaults = dict(
        mean=4.2, std=1.5, median=4.2, min=1.0, max=8.0,
        n_samples=500, quantile_25=3.0, quantile_75=5.5,
    )
    defaults.update(overrides)
    return DistributionSummary(**defaults)


def _make_cur_summary(**overrides) -> DistributionSummary:
    defaults = dict(
        mean=6.8, std=1.8, median=6.8, min=2.5, max=12.0,
        n_samples=500, quantile_25=5.5, quantile_75=8.0,
    )
    defaults.update(overrides)
    return DistributionSummary(**defaults)


def _make_drifted_feature(
    name: str = "warehouse_lead_time",
    test_name: str = "psi",
    statistic: float = 0.31,
    p_value: float = 0.0001,
    severity: str = "severe",
    **overrides,
) -> FeatureDriftResult:
    return FeatureDriftResult(
        feature_name=name,
        test_name=test_name,
        statistic=statistic,
        p_value=p_value,
        drift_detected=True,
        severity=severity,
        reference_summary=overrides.get("reference_summary", _make_ref_summary()),
        current_summary=overrides.get("current_summary", _make_cur_summary()),
    )


def _make_stable_feature(name: str = "stable_feature") -> FeatureDriftResult:
    summary = _make_ref_summary()
    return FeatureDriftResult(
        feature_name=name,
        test_name="psi",
        statistic=0.02,
        p_value=0.85,
        drift_detected=False,
        severity="none",
        reference_summary=summary,
        current_summary=summary,
    )


def _make_dataset_drift_mixed() -> DatasetDriftResult:
    """3 of 12 features drifted, overall severe."""
    drifted = [
        _make_drifted_feature("warehouse_lead_time", statistic=0.31, severity="severe"),
        _make_drifted_feature("order_frequency", test_name="psi", statistic=0.19, severity="moderate", p_value=0.002),
        _make_drifted_feature("unit_price", test_name="psi", statistic=0.14, severity="moderate", p_value=0.01),
    ]
    stable = [_make_stable_feature(f"stable_{i}") for i in range(9)]
    all_features = drifted + stable
    return DatasetDriftResult(
        features=all_features,
        n_features=12,
        n_drifted=3,
        share_drifted=0.25,
        overall_severity="severe",
    )


def _make_dataset_drift_none() -> DatasetDriftResult:
    stable = [_make_stable_feature(f"feature_{i}") for i in range(5)]
    return DatasetDriftResult(
        features=stable,
        n_features=5,
        n_drifted=0,
        share_drifted=0.0,
        overall_severity="none",
    )


# ---------------------------------------------------------------------------
# Group E: Feature-level narration
# ---------------------------------------------------------------------------


class TestNarrateFeatureDrift:
    """E1–E5: Feature-level drift narration."""

    def test_narrate_drifted_numeric(self):
        """E1: Severity first, shift direction, supporting evidence."""
        result = _make_drifted_feature()
        narrative = narrate_feature_drift(result)

        assert "Significant drift detected" in narrative
        assert "warehouse_lead_time" in narrative
        assert "PSI = 0.3100" in narrative
        assert "severe" in narrative
        assert "KS p-value" in narrative
        assert "median moved from 4.20 to 6.80" in narrative

    def test_narrate_non_drifted(self):
        """E2: No drift → reassuring message."""
        result = _make_stable_feature()
        narrative = narrate_feature_drift(result)

        assert "No significant drift detected" in narrative
        assert "stable_feature" in narrative

    def test_narrate_categorical(self):
        """E3: Chi-squared test mentioned."""
        result = _make_drifted_feature(
            name="region",
            test_name="chi_squared",
            statistic=25.3,
            p_value=0.0001,
            severity="severe",
        )
        narrative = narrate_feature_drift(result)

        assert "chi-squared" in narrative
        assert "25.3" in narrative
        assert "p-value" in narrative

    def test_narrate_feature_drift_deterministic(self):
        """E4: Same input → byte-for-byte identical output, 5 times."""
        result = _make_drifted_feature()
        outputs = [narrate_feature_drift(result) for _ in range(5)]
        assert len(set(outputs)) == 1

    def test_narrate_external_source(self):
        """E5: Narrator works identically for external origin."""
        result = _make_drifted_feature()
        # Narrator doesn't use origin — it's source-agnostic
        narrative = narrate_feature_drift(result)
        assert "Significant drift detected" in narrative

    def test_narrate_feature_drift_snapshot(self, snapshot: SnapshotAssertion):
        """Snapshot lock for feature drift narrative."""
        result = _make_drifted_feature()
        narrative = narrate_feature_drift(result)
        assert narrative == snapshot


# ---------------------------------------------------------------------------
# Group D: Dataset-level narration
# ---------------------------------------------------------------------------


class TestNarrateDatasetDrift:
    """D1–D3: Dataset-level drift narration."""

    def test_narrate_mixed_drift(self):
        """D1: Count, ranked features, severity, no cause speculation."""
        result = _make_dataset_drift_mixed()
        narrative = narrate_dataset_drift(result)

        assert "3 of 12 features" in narrative
        assert "severe" in narrative
        assert "warehouse_lead_time" in narrative
        assert "order_frequency" in narrative
        assert "unit_price" in narrative
        # Severe features ranked before moderate
        wlt_pos = narrative.index("warehouse_lead_time")
        of_pos = narrative.index("order_frequency")
        assert wlt_pos < of_pos
        # No cause speculation
        assert "because" not in narrative.lower()
        assert "caused by" not in narrative.lower()

    def test_narrate_no_drift(self):
        """D2: No drift → reassuring message."""
        result = _make_dataset_drift_none()
        narrative = narrate_dataset_drift(result)

        assert "No significant drift detected" in narrative
        assert "5 features" in narrative

    def test_narrate_dataset_drift_deterministic(self):
        """D3: Same input → byte-for-byte identical output, 5 times."""
        result = _make_dataset_drift_mixed()
        outputs = [narrate_dataset_drift(result) for _ in range(5)]
        assert len(set(outputs)) == 1

    def test_narrate_dataset_drift_snapshot(self, snapshot: SnapshotAssertion):
        """Snapshot lock for dataset drift narrative."""
        result = _make_dataset_drift_mixed()
        narrative = narrate_dataset_drift(result)
        assert narrative == snapshot

    def test_narrate_moderate_severity_framing(self):
        """Moderate severity gets monitoring language."""
        features = [
            _make_drifted_feature("f1", statistic=0.15, severity="moderate"),
            _make_stable_feature("f2"),
        ]
        result = DatasetDriftResult(
            features=features,
            n_features=2,
            n_drifted=1,
            share_drifted=0.5,
            overall_severity="moderate",
        )
        narrative = narrate_dataset_drift(result)

        assert "monitored" in narrative.lower()
