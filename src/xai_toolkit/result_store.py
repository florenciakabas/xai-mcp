"""Result store — persisted explanation contract (Layer 3).

Pydantic schemas define the stored format. Read/write functions are backed
by parquet files organized as {store_path}/{model_id}/explanations.parquet.

No Spark, no Delta, no Kedro — just pandas + parquet.

The Kedro adapter (Layer 4) writes via catalog instead of these functions,
but uses the same schemas. The MCP server reads via this module.
"""

import logging
from pathlib import Path

import pandas as pd
from xai_toolkit.schemas import (
    StoredDriftResult,
    StoredExplanation,
    StoredModelSummary,
)

logger = logging.getLogger(__name__)


LATEST_RUN_STRATEGY = "latest_successful_by_computed_at"


def resolve_latest_successful_run_id(df: pd.DataFrame) -> str | None:
    """Resolve latest run deterministically by computed_at.

    "Successful" is inferred from persisted rows that have both run_id and
    computed_at fields present. Ties are broken by run_id ascending order after
    sorting descending by timestamp.
    """
    if df.empty:
        return None
    if "run_id" not in df.columns or "computed_at" not in df.columns:
        return None

    run_rows = df[["run_id", "computed_at"]].dropna()
    if run_rows.empty:
        return None

    parsed = run_rows.copy()
    parsed["computed_at"] = pd.to_datetime(parsed["computed_at"], utc=True, errors="coerce")
    parsed = parsed.dropna(subset=["computed_at"])
    if parsed.empty:
        return None

    latest = parsed.sort_values(["computed_at", "run_id"], ascending=[False, False]).iloc[0]
    return str(latest["run_id"])


# ---------------------------------------------------------------------------
# Write functions
# ---------------------------------------------------------------------------


def _ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def save_explanations(
    results: list[StoredExplanation],
    store_path: str | Path,
) -> Path:
    """Save explanation results to parquet.

    Appends to existing file if present (multiple runs coexist).

    Returns:
        Path to the written parquet file.
    """
    store_path = Path(store_path)
    if not results:
        logger.warning("No explanations to save.")
        return store_path

    model_id = results[0].model_id
    model_dir = store_path / model_id
    _ensure_dir(model_dir)

    parquet_path = model_dir / "explanations.parquet"

    new_df = pd.DataFrame([r.model_dump() for r in results])

    if parquet_path.exists():
        try:
            existing_df = pd.read_parquet(parquet_path)
            combined = pd.concat([existing_df, new_df], ignore_index=True)
        except Exception:
            logger.warning("Could not read existing explanations; overwriting.")
            combined = new_df
    else:
        combined = new_df

    combined.to_parquet(parquet_path, index=False)
    logger.info(
        "Saved %d explanations for '%s' to %s (total: %d)",
        len(results), model_id, parquet_path, len(combined),
    )
    return parquet_path


def save_drift_results(
    results: list[StoredDriftResult],
    store_path: str | Path,
) -> Path:
    """Save drift results to parquet.

    Returns:
        Path to the written parquet file.
    """
    store_path = Path(store_path)
    if not results:
        logger.warning("No drift results to save.")
        return store_path

    model_id = results[0].model_id
    model_dir = store_path / model_id
    _ensure_dir(model_dir)

    parquet_path = model_dir / "drift_results.parquet"

    new_df = pd.DataFrame([r.model_dump() for r in results])

    if parquet_path.exists():
        try:
            existing_df = pd.read_parquet(parquet_path)
            combined = pd.concat([existing_df, new_df], ignore_index=True)
        except Exception:
            logger.warning("Could not read existing drift results; overwriting.")
            combined = new_df
    else:
        combined = new_df

    combined.to_parquet(parquet_path, index=False)
    logger.info(
        "Saved %d drift results for '%s' to %s",
        len(results), model_id, parquet_path,
    )
    return parquet_path


def save_model_summaries(
    results: list[StoredModelSummary],
    store_path: str | Path,
) -> Path:
    """Save model summary results to parquet.

    Returns:
        Path to the written parquet file.
    """
    store_path = Path(store_path)
    if not results:
        logger.warning("No model summaries to save.")
        return store_path

    model_id = results[0].model_id
    model_dir = store_path / model_id
    _ensure_dir(model_dir)

    parquet_path = model_dir / "model_summary.parquet"

    new_df = pd.DataFrame([r.model_dump() for r in results])

    if parquet_path.exists():
        try:
            existing_df = pd.read_parquet(parquet_path)
            combined = pd.concat([existing_df, new_df], ignore_index=True)
        except Exception:
            logger.warning("Could not read existing model summaries; overwriting.")
            combined = new_df
    else:
        combined = new_df

    combined.to_parquet(parquet_path, index=False)
    logger.info(
        "Saved %d model summary rows for '%s' to %s",
        len(results), model_id, parquet_path,
    )
    return parquet_path


# ---------------------------------------------------------------------------
# Read functions
# ---------------------------------------------------------------------------


def load_explanation(
    model_id: str,
    sample_index: int,
    store_path: str | Path,
    run_id: str | None = None,
) -> StoredExplanation | None:
    """Load a single explanation by model_id and sample_index.

    Args:
        model_id: Model identifier.
        sample_index: Row index of the sample.
        store_path: Root path of the result store.
        run_id: If provided, filter to this specific run.
            If None, returns the most recent result.

    Returns:
        StoredExplanation or None if not found.
    """
    store_path = Path(store_path)
    parquet_path = store_path / model_id / "explanations.parquet"

    if not parquet_path.exists():
        return None

    try:
        df = pd.read_parquet(parquet_path)
    except Exception:
        logger.warning("Could not read explanations from %s", parquet_path)
        return None

    resolved_run_id = run_id if run_id is not None else resolve_latest_successful_run_id(df)
    mask = df["sample_index"] == sample_index
    if resolved_run_id is not None:
        mask = mask & (df["run_id"] == resolved_run_id)

    filtered = df[mask]
    if filtered.empty:
        return None

    # Most recent by computed_at
    row = filtered.sort_values("computed_at", ascending=False).iloc[0]
    return StoredExplanation(**row.to_dict())


def load_explanations(
    model_id: str,
    store_path: str | Path,
    run_id: str | None = None,
) -> list[StoredExplanation]:
    """Load all explanations for a model.

    Args:
        model_id: Model identifier.
        store_path: Root path of the result store.
        run_id: If provided, filter to this specific run.
            If None, returns results from the most recent run.

    Returns:
        List of StoredExplanation (empty if none found).
    """
    store_path = Path(store_path)
    parquet_path = store_path / model_id / "explanations.parquet"

    if not parquet_path.exists():
        return []

    try:
        df = pd.read_parquet(parquet_path)
    except Exception:
        logger.warning("Could not read explanations from %s", parquet_path)
        return []

    resolved_run_id = run_id if run_id is not None else resolve_latest_successful_run_id(df)
    if resolved_run_id is not None:
        df = df[df["run_id"] == resolved_run_id]

    return [StoredExplanation(**row.to_dict()) for _, row in df.iterrows()]


def load_drift_results(
    model_id: str,
    store_path: str | Path,
    run_id: str | None = None,
) -> list[StoredDriftResult]:
    """Load drift results for a model.

    Args:
        model_id: Model identifier.
        store_path: Root path of the result store.
        run_id: If provided, filter to this specific run.
            If None, returns results from the most recent run.

    Returns:
        List of StoredDriftResult (empty if none found).
    """
    store_path = Path(store_path)
    parquet_path = store_path / model_id / "drift_results.parquet"

    if not parquet_path.exists():
        return []

    try:
        df = pd.read_parquet(parquet_path)
    except Exception:
        logger.warning("Could not read drift results from %s", parquet_path)
        return []

    resolved_run_id = run_id if run_id is not None else resolve_latest_successful_run_id(df)
    if resolved_run_id is not None:
        df = df[df["run_id"] == resolved_run_id]

    return [StoredDriftResult(**row.to_dict()) for _, row in df.iterrows()]
