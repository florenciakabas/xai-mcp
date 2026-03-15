"""Pure query helpers for persisted result-store data."""

from __future__ import annotations

from pathlib import Path

from xai_toolkit.result_store import (
    load_drift_results,
    load_explanations,
)
from xai_toolkit.schemas import StoredDriftResult, StoredExplanation


def discover_store_model_ids(store_path: Path | None) -> list[str]:
    """Return sorted model IDs that have persisted result artifacts."""
    if store_path is None or not store_path.exists():
        return []
    model_ids: list[str] = []
    for entry in store_path.iterdir():
        if not entry.is_dir():
            continue
        if (
            (entry / "drift_results.parquet").exists()
            or (entry / "explanations.parquet").exists()
        ):
            model_ids.append(entry.name)
    return sorted(model_ids)


def query_drift_results(
    store_path: Path,
    model_id: str | None = None,
    run_id: str | None = None,
) -> list[StoredDriftResult]:
    """Fetch drift rows from the result store with optional model/run filtering."""
    model_ids = [model_id] if model_id is not None else discover_store_model_ids(store_path)
    all_results: list[StoredDriftResult] = []
    for mid in model_ids:
        all_results.extend(load_drift_results(mid, store_path, run_id=run_id))
    return all_results


def query_explanations(
    store_path: Path,
    model_id: str,
    run_id: str | None = None,
) -> list[StoredExplanation]:
    """Fetch explanation rows for a model from the result store."""
    return load_explanations(model_id, store_path, run_id=run_id)
