"""Pure drift detection — no MCP imports, no side effects (ADR-001).

This module computes statistical drift tests between a reference (baseline)
distribution and a current (production) distribution. Same role as
explainers.py: thin scipy wrappers returning structured Pydantic models.

Stateless design (ADR-011): the caller provides both DataFrames.
The toolkit does NOT manage baselines or temporal state.

Statistical tests:
  - KS test (scipy.stats.ks_2samp): non-parametric, continuous features.
  - PSI (manual numpy): population stability index, continuous features.
  - Chi-squared (scipy.stats.chisquare): categorical features.

Auto-select logic:
  - Numeric features → PSI (primary) + KS (supporting p_value).
  - If numeric feature has <10 unique values → KS only (PSI skipped).
  - Categorical features → chi-squared.
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

from xai_toolkit.schemas import (
    DatasetDriftResult,
    DistributionSummary,
    FeatureDriftResult,
)

logger = logging.getLogger(__name__)

# --- Severity thresholds (hardcoded, documented) ---

_PSI_MODERATE = 0.1
_PSI_SEVERE = 0.25

_PVALUE_MODERATE = 0.05
_PVALUE_SEVERE = 0.01

# Minimum unique values required for decile-based PSI binning
_PSI_MIN_UNIQUE = 10


def compute_distribution_summary(series: pd.Series) -> DistributionSummary:
    """Compute descriptive statistics for a single feature.

    Args:
        series: A pandas Series (NaN values should be pre-dropped).

    Returns:
        DistributionSummary with mean, std, median, min, max, quartiles.
    """
    if series.empty:
        raise ValueError("Distribution summary requires at least one sample.")

    return DistributionSummary(
        mean=round(float(series.mean()), 6),
        std=round(float(series.std()), 6),
        median=round(float(series.median()), 6),
        min=round(float(series.min()), 6),
        max=round(float(series.max()), 6),
        n_samples=len(series),
        quantile_25=round(float(series.quantile(0.25)), 6),
        quantile_75=round(float(series.quantile(0.75)), 6),
    )


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Population Stability Index between two distributions.

    Uses quantile-based binning on the reference distribution to define
    bin edges, then compares the proportion of values in each bin.

    Args:
        reference: Reference (baseline) values.
        current: Current (production) values.
        n_bins: Number of bins (default: 10 for deciles).

    Returns:
        PSI value (non-negative float). Higher = more drift.
    """
    # Quantile-based bin edges from reference
    edges = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts = np.histogram(reference, bins=edges)[0].astype(float)
    cur_counts = np.histogram(current, bins=edges)[0].astype(float)

    # Convert to proportions, clip to avoid log(0) and division by zero
    eps = 1e-6
    ref_pct = np.clip(ref_counts / ref_counts.sum(), eps, None)
    cur_pct = np.clip(cur_counts / cur_counts.sum(), eps, None)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return round(max(psi, 0.0), 6)


def _severity_from_psi(psi: float) -> str:
    """Map PSI value to severity label."""
    if psi >= _PSI_SEVERE:
        return "severe"
    elif psi >= _PSI_MODERATE:
        return "moderate"
    return "none"


def _severity_from_pvalue(p_value: float) -> str:
    """Map p-value to severity label (used for KS fallback and chi-squared)."""
    if p_value < _PVALUE_SEVERE:
        return "severe"
    elif p_value < _PVALUE_MODERATE:
        return "moderate"
    return "none"


def _drop_nans(
    reference: pd.Series,
    current: pd.Series,
    feature_name: str,
) -> tuple[pd.Series, pd.Series]:
    """Drop NaN values per-feature, warn if >70% dropped.

    Args:
        reference: Reference series (may contain NaN).
        current: Current series (may contain NaN).
        feature_name: Feature name for log messages.

    Returns:
        Tuple of (cleaned reference, cleaned current).
    """
    ref_clean = reference.dropna()
    cur_clean = current.dropna()

    ref_dropped = len(reference) - len(ref_clean)
    cur_dropped = len(current) - len(cur_clean)

    if ref_dropped > 0 or cur_dropped > 0:
        ref_pct = ref_dropped / len(reference) if len(reference) > 0 else 0
        cur_pct = cur_dropped / len(current) if len(current) > 0 else 0

        if ref_pct > 0.7 or cur_pct > 0.7:
            logger.warning(
                "Feature '%s': high NaN rate — reference %.0f%%, current %.0f%%. "
                "Results may be unreliable.",
                feature_name,
                ref_pct * 100,
                cur_pct * 100,
            )
        else:
            logger.info(
                "Feature '%s': dropped %d NaN from reference, %d from current.",
                feature_name,
                ref_dropped,
                cur_dropped,
            )

    return ref_clean, cur_clean


def detect_feature_drift(
    reference: pd.Series,
    current: pd.Series,
    feature_name: str,
) -> FeatureDriftResult:
    """Detect drift for a single feature between reference and current.

    Auto-selects the statistical test based on dtype:
      - Numeric: PSI (primary) + KS (supporting). Falls back to KS-only
        if fewer than 10 unique values in reference.
      - Non-numeric: chi-squared. Categories in current but not in
        reference are dropped with a warning.

    Args:
        reference: Reference (baseline) values for this feature.
        current: Current (production) values for this feature.
        feature_name: Human-readable feature name.

    Returns:
        FeatureDriftResult with test results, severity, and distribution summaries.
    """
    ref_clean, cur_clean = _drop_nans(reference, current, feature_name)
    if ref_clean.empty:
        raise ValueError(
            f"Feature '{feature_name}' has no usable reference samples after NaN filtering."
        )
    if cur_clean.empty:
        raise ValueError(
            f"Feature '{feature_name}' has no usable current samples after NaN filtering."
        )

    is_numeric = pd.api.types.is_numeric_dtype(ref_clean)

    if is_numeric:
        ref_summary = compute_distribution_summary(ref_clean)
        cur_summary = compute_distribution_summary(cur_clean)

        # KS test (always computed for numeric)
        ks_stat, ks_pvalue = stats.ks_2samp(ref_clean.values, cur_clean.values)

        # Check if PSI is feasible
        n_unique = ref_clean.nunique()
        if n_unique >= _PSI_MIN_UNIQUE:
            psi = compute_psi(ref_clean.values, cur_clean.values)
            severity = _severity_from_psi(psi)
            return FeatureDriftResult(
                feature_name=feature_name,
                test_name="psi",
                statistic=round(psi, 6),
                p_value=round(float(ks_pvalue), 6),
                drift_detected=severity != "none",
                severity=severity,
                reference_summary=ref_summary,
                current_summary=cur_summary,
            )
        else:
            # KS-only fallback for low-cardinality numeric features
            logger.info(
                "Feature '%s': only %d unique values, skipping PSI (need >= %d). "
                "Using KS test as primary.",
                feature_name,
                n_unique,
                _PSI_MIN_UNIQUE,
            )
            severity = _severity_from_pvalue(float(ks_pvalue))
            return FeatureDriftResult(
                feature_name=feature_name,
                test_name="ks",
                statistic=round(float(ks_stat), 6),
                p_value=round(float(ks_pvalue), 6),
                drift_detected=severity != "none",
                severity=severity,
                reference_summary=ref_summary,
                current_summary=cur_summary,
            )
    else:
        # Categorical: chi-squared test
        ref_counts = ref_clean.value_counts()
        cur_counts = cur_clean.value_counts()

        # Drop categories in current that are not in reference
        unknown_cats = set(cur_counts.index) - set(ref_counts.index)
        if unknown_cats:
            logger.warning(
                "Feature '%s': dropping %d categories from current not in reference: %s",
                feature_name,
                len(unknown_cats),
                sorted(unknown_cats),
            )
            cur_clean = cur_clean[cur_clean.isin(ref_counts.index)]
            cur_counts = cur_clean.value_counts()

        # Align on reference categories (fill missing with 0)
        all_cats = ref_counts.index
        observed = np.array([cur_counts.get(cat, 0) for cat in all_cats], dtype=float)

        # Expected proportions from reference, scaled to current sample size
        ref_proportions = ref_counts.loc[all_cats].values.astype(float)
        expected = ref_proportions / ref_proportions.sum() * observed.sum()

        # Avoid zero expected (would cause division by zero in chi-squared)
        eps = 1e-6
        expected = np.clip(expected, eps, None)

        chi2_stat, chi2_pvalue = stats.chisquare(observed, f_exp=expected)

        severity = _severity_from_pvalue(float(chi2_pvalue))

        # Distribution summary for categorical: use code-based numeric summary
        # We encode categories as integer codes for summary statistics
        ref_codes = ref_clean.astype("category").cat.codes.astype(float)
        cur_codes = cur_clean.astype("category").cat.codes.astype(float)
        ref_summary = compute_distribution_summary(ref_codes)
        cur_summary = compute_distribution_summary(cur_codes)

        return FeatureDriftResult(
            feature_name=feature_name,
            test_name="chi_squared",
            statistic=round(float(chi2_stat), 6),
            p_value=round(float(chi2_pvalue), 6),
            drift_detected=severity != "none",
            severity=severity,
            reference_summary=ref_summary,
            current_summary=cur_summary,
        )


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
) -> DatasetDriftResult:
    """Detect drift across all features between reference and current DataFrames.

    Iterates over each column, auto-selects the appropriate test,
    and aggregates results. Overall severity is the worst severity
    across all features.

    Args:
        reference: Reference (baseline) DataFrame.
        current: Current (production) DataFrame.

    Returns:
        DatasetDriftResult with per-feature results and aggregate summary.
    """
    features: list[FeatureDriftResult] = []

    for col in reference.columns:
        if col not in current.columns:
            logger.warning("Feature '%s' missing from current DataFrame, skipping.", col)
            continue

        result = detect_feature_drift(
            reference=reference[col],
            current=current[col],
            feature_name=col,
        )
        features.append(result)

    n_features = len(features)
    n_drifted = sum(1 for f in features if f.drift_detected)
    share_drifted = round(n_drifted / n_features, 6) if n_features > 0 else 0.0

    # Overall severity = worst across all features
    severity_order = {"none": 0, "moderate": 1, "severe": 2}
    reverse_order = {0: "none", 1: "moderate", 2: "severe"}
    worst = max(severity_order[f.severity] for f in features) if features else 0
    overall_severity = reverse_order[worst]

    if n_features == 0:
        raise ValueError(
            "No overlapping features available for drift detection. "
            "Ensure current data contains at least one reference feature."
        )

    return DatasetDriftResult(
        features=features,
        n_features=n_features,
        n_drifted=n_drifted,
        share_drifted=share_drifted,
        overall_severity=overall_severity,
    )
