"""Tests for explainers.py — pure computation correctness.

Focused on compute_data_hash (D3-S2).
No model loading, no SHAP computation — fast and fully isolated.
"""

import pandas as pd

from xai_toolkit.explainers import compute_data_hash


class TestComputeDataHash:
    """Tests for compute_data_hash (D3-S2)."""

    def _sample_df(self) -> pd.DataFrame:
        """Minimal DataFrame for deterministic testing."""
        return pd.DataFrame(
            {"feature_a": [1.0, 2.0, 3.0], "feature_b": [0.1, 0.2, 0.3]}
        )

    # --- Shape and format ---

    def test_returns_64_char_hex_string(self):
        """SHA256 hex digest is always exactly 64 hex characters."""
        h = compute_data_hash(self._sample_df())
        assert isinstance(h, str)
        assert len(h) == 64

    def test_output_is_valid_hex(self):
        """Output contains only valid hex characters."""
        h = compute_data_hash(self._sample_df())
        assert all(c in "0123456789abcdef" for c in h)

    # --- Determinism ---

    def test_full_matrix_hash_is_deterministic(self):
        """Same DataFrame → same hash, always."""
        df = self._sample_df()
        assert compute_data_hash(df) == compute_data_hash(df)

    def test_single_row_hash_is_deterministic(self):
        """Same row index → same hash, always."""
        df = self._sample_df()
        h1 = compute_data_hash(df, sample_index=0)
        h2 = compute_data_hash(df, sample_index=0)
        assert h1 == h2

    def test_hash_is_stable_across_5_calls(self):
        """D3-S1: 5 consecutive calls produce the same hash."""
        df = self._sample_df()
        hashes = [compute_data_hash(df, sample_index=1) for _ in range(5)]
        assert len(set(hashes)) == 1, (
            f"Expected 1 unique hash across 5 calls, got {len(set(hashes))}"
        )

    # --- Sensitivity ---

    def test_different_rows_produce_different_hashes(self):
        """Different rows with different values → different hashes."""
        df = self._sample_df()
        assert compute_data_hash(df, sample_index=0) != compute_data_hash(df, sample_index=1)

    def test_full_matrix_differs_from_single_row(self):
        """Hashing the full matrix vs a single row must differ."""
        df = self._sample_df()
        assert compute_data_hash(df) != compute_data_hash(df, sample_index=0)

    def test_tiny_value_change_changes_hash(self):
        """A single-digit change anywhere in the data changes the hash."""
        df1 = pd.DataFrame({"a": [1.0], "b": [2.0]})
        df2 = pd.DataFrame({"a": [1.0], "b": [2.0000001]})
        assert compute_data_hash(df1) != compute_data_hash(df2)
