"""Reproducibility tests — D3-S1.

The core guarantee: same input + same data = same output, every time.
Timestamps are the ONLY field permitted to differ between calls.

These tests exercise the pure Python narrator layer — no MCP server,
no model loading, no SHAP computation. They run fast and in isolation.
"""

import pandas as pd

from xai_toolkit.explainers import compute_data_hash
from xai_toolkit.narrators import (
    narrate_dataset,
    narrate_feature_comparison,
    narrate_model_summary,
    narrate_partial_dependence,
    narrate_prediction,
)
from tests.test_narrators import (
    _make_dataset_description,
    _make_feature_importances,
    _make_model_summary,
    _make_pdp_result,
    _make_shap_result,
)

N_CALLS = 5  # D3-S1 specifies 5 identical calls


class TestNarratorReproducibility:
    """D3-S1: Every narrator produces byte-for-byte identical output across N_CALLS."""

    def _assert_stable(self, outputs: list[str], narrator_name: str) -> None:
        unique = set(outputs)
        assert len(unique) == 1, (
            f"{narrator_name} produced {len(unique)} distinct outputs across "
            f"{N_CALLS} calls — expected exactly 1.\n"
            f"Unique outputs:\n" + "\n---\n".join(unique)
        )

    def test_narrate_prediction_is_stable(self):
        shap_result = _make_shap_result()
        outputs = [narrate_prediction(shap_result, top_n=3) for _ in range(N_CALLS)]
        self._assert_stable(outputs, "narrate_prediction")

    def test_narrate_model_summary_is_stable(self):
        summary = _make_model_summary()
        outputs = [narrate_model_summary(summary) for _ in range(N_CALLS)]
        self._assert_stable(outputs, "narrate_model_summary")

    def test_narrate_feature_comparison_is_stable(self):
        importances = _make_feature_importances()
        outputs = [narrate_feature_comparison(importances) for _ in range(N_CALLS)]
        self._assert_stable(outputs, "narrate_feature_comparison")

    def test_narrate_partial_dependence_is_stable(self):
        pdp = _make_pdp_result()
        outputs = [narrate_partial_dependence(pdp) for _ in range(N_CALLS)]
        self._assert_stable(outputs, "narrate_partial_dependence")

    def test_narrate_dataset_is_stable(self):
        desc = _make_dataset_description()
        outputs = [narrate_dataset(desc) for _ in range(N_CALLS)]
        self._assert_stable(outputs, "narrate_dataset")


class TestDataHashReproducibility:
    """D3-S2: compute_data_hash is stable and sensitive."""

    def _make_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]}
        )

    def test_full_matrix_hash_is_stable_across_5_calls(self):
        df = self._make_df()
        hashes = [compute_data_hash(df) for _ in range(N_CALLS)]
        assert len(set(hashes)) == 1

    def test_single_row_hash_is_stable_across_5_calls(self):
        df = self._make_df()
        hashes = [compute_data_hash(df, sample_index=0) for _ in range(N_CALLS)]
        assert len(set(hashes)) == 1

    def test_different_rows_have_different_hashes(self):
        df = self._make_df()
        h0 = compute_data_hash(df, sample_index=0)
        h1 = compute_data_hash(df, sample_index=1)
        assert h0 != h1, "Different rows must produce different hashes"
