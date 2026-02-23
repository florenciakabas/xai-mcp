# ADR-008: Pipeline Bridge — Reading Pre-Computed SHAP Artifacts

## Status

Accepted

## Date

2026-02-23

## Context

The existing Kedro explainability pipeline (`xai-xgboost-clf` repo)
computes SHAP values in batch and persists them as:
- `shap_values.npy` — SHAP values for all explained samples
- `shap_expected_value.npy` — baseline (expected) values
- `shap_metadata.json` — feature names, model type, explainer type

During the sprint, our MCP toolkit computed SHAP values on the fly
(directly from model + data) for speed of development. This works for
the PoC but:
1. **Duplicates** computation the pipeline already performs
2. **Misses** the pipeline's correct background data methodology
   (training set vs. our PoC shortcut of test set)
3. **Ignores** the pipeline's model-type-aware explainer selection
4. **Doesn't scale** — on-the-fly SHAP over large datasets is slow

A colleague who built the Kedro pipeline raised a valid concern:
the MCP toolkit should consume pipeline outputs, not bypass them.

## Decision

Add **pipeline bridge functions** to `explainers.py` that read
pre-computed SHAP artifacts from disk and return our standard
Pydantic schemas (`ShapResult`, `FeatureImportance`).

This gives us two modes of operation:
- **On-the-fly** (PoC): `compute_shap_values()` — computes directly
- **From pipeline** (production): `load_shap_from_pipeline()` — reads artifacts

Both return the same `ShapResult`, so narrators and plots work identically
regardless of how the SHAP values were obtained.

## Consequences

### Positive
- Pipeline's batch computation is reused, not duplicated
- Correct SHAP methodology (training set background) is inherited
- Narrators and plots are agnostic to the data source
- `ToolMetadata.source` field makes the provenance auditable
- Colleague's infrastructure is visibly integrated

### Negative
- Feature values are not available from pipeline artifacts alone
  (placeholder zeros are used; callers must join with original data)
- Prediction and probability require the model, which the bridge
  doesn't load (set to placeholder values)
- Two code paths must be maintained and tested

### Trade-off
The bridge functions are intentionally simple readers, not a full
integration layer. They don't handle MLflow artifact fetching,
multi-output models, or Kedro catalog resolution. Those are
documented as future production-path items.

## Alternatives Considered

1. **Only use pipeline artifacts, remove on-the-fly computation**
   Rejected: on-the-fly is essential for interactive exploration
   and for datasets where the pipeline hasn't been run yet.

2. **Import the colleague's code directly as a dependency**
   Rejected: the Kedro pipeline has heavy dependencies (Kedro,
   MLflow, etc.) that we don't want in the MCP server. Reading
   the output files is a cleaner boundary.

3. **Convert pipeline artifacts to our schema at pipeline time**
   Deferred: would require changes to the colleague's repo. The
   bridge approach lets us integrate without modifying their code.
