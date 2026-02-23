"""Snapshot tests — D3-S6.

Golden files live in tests/__snapshots__/test_snapshots.ambr.

FIRST RUN (write snapshots):
    uv run python -m pytest tests/test_snapshots.py --snapshot-update -v

SUBSEQUENT RUNS (verify nothing changed):
    uv run python -m pytest tests/test_snapshots.py -v

Any unintended change to a narrator output causes a test failure.
To intentionally update a narrative, re-run with --snapshot-update,
review the diff, and commit the updated .ambr file alongside the code change.
This creates an explicit, reviewable audit trail of narrative changes.
"""

import pytest
from syrupy.assertion import SnapshotAssertion

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


def test_narrate_prediction_snapshot(snapshot: SnapshotAssertion):
    """narrate_prediction output is locked to the golden value."""
    narrative = narrate_prediction(_make_shap_result(), top_n=3)
    assert narrative == snapshot


def test_narrate_model_summary_snapshot(snapshot: SnapshotAssertion):
    """narrate_model_summary output is locked to the golden value."""
    narrative = narrate_model_summary(_make_model_summary())
    assert narrative == snapshot


def test_narrate_feature_comparison_snapshot(snapshot: SnapshotAssertion):
    """narrate_feature_comparison output is locked to the golden value."""
    narrative = narrate_feature_comparison(_make_feature_importances())
    assert narrative == snapshot


def test_narrate_partial_dependence_snapshot(snapshot: SnapshotAssertion):
    """narrate_partial_dependence output is locked to the golden value."""
    narrative = narrate_partial_dependence(_make_pdp_result())
    assert narrative == snapshot


def test_narrate_dataset_snapshot(snapshot: SnapshotAssertion):
    """narrate_dataset output is locked to the golden value."""
    narrative = narrate_dataset(_make_dataset_description())
    assert narrative == snapshot
