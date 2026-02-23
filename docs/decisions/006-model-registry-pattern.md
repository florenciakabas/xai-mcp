# ADR-006: ModelRegistry Pattern for Managing Loaded Models

## Status: Accepted
## Date: 2025-02-21

## Context

The server needs to load models, their metadata, and their associated test
data at startup and make them available to tool handlers. Multiple tools
call the same model; we need a single point of loading and access.

## Decision

Use a **Registry pattern** (`registry.py`). A `ModelRegistry` instance is
created once at server startup. Each model is loaded once and stored by ID.
Tool handlers call `registry.get(model_id)` to retrieve a `RegisteredModel`
dataclass containing the fitted model, metadata dict, and test DataFrames.

```python
registry = ModelRegistry()
registry.load_from_disk("xgboost_breast_cancer", MODELS_DIR, DATA_DIR)
registry.load_from_disk("rf_breast_cancer", MODELS_DIR, DATA_DIR)

entry = registry.get("xgboost_breast_cancer")
entry.model          # fitted sklearn/xgboost model
entry.metadata       # dict with accuracy, feature_names, target_names, etc.
entry.X_test         # pd.DataFrame
entry.y_test         # pd.Series
```

## Consequences

- ✅ Models load once — no repeated disk I/O per tool call
- ✅ Tool handlers are decoupled from loading logic — they just ask for a model by ID
- ✅ Easy to add new models without touching tool code (add one `load_from_disk` call)
- ✅ `KeyError` on unknown model_id gives a clear, actionable error message
- ✅ `list_models()` provides introspection for the `list_models` MCP tool
- ⚠️ In production, replace disk-based loading with MLflow registry calls
- ⚠️ Registry is in-memory — a server restart re-loads from disk (acceptable for PoC)

## Production Evolution

```python
# PoC (current):
registry.load_from_disk("xgboost_breast_cancer", MODELS_DIR, DATA_DIR)

# Production (Phase 2):
registry.load_from_mlflow("xgboost_breast_cancer", mlflow_uri, experiment_name)
```

The `RegisteredModel` dataclass interface stays the same — tool handlers
are unaffected by the loading source change.

## Alternatives Considered

- **Global variables per model**: Unscalable, no introspection. Rejected.
- **Load on demand per tool call**: Repeated I/O, non-deterministic latency. Rejected.
- **Dependency injection via FastMCP context**: More complex, no advantage at this scale. Rejected.
