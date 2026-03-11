# ADR-011: Stateless Drift Detection

## Status

Accepted

## Context

The toolkit needs drift detection to compare training (reference) and
production (current) data distributions. The question is whether the
toolkit should manage baseline distributions internally or require
the caller to provide both datasets.

## Decision

**Stateless.** The caller provides both the reference DataFrame and the
current DataFrame. The toolkit does NOT manage baselines, store
distributions, or maintain temporal state.

The two MCP tools (`detect_drift`, `detect_feature_drift`) use the
model registry to access X_train (reference) and X_test (current).
The pure functions in `drift.py` accept raw pandas objects with no
registry dependency.

## Consequences

### Benefits
- Simple: no database, no file system state, no cache invalidation.
- Predictable: same two DataFrames always produce the same result.
- Composable: the caller can compare any two snapshots (train vs. prod,
  January vs. February, product A vs. product B).
- Testable: pure functions with no hidden state.

### Limitations
- The caller must manage baseline storage and versioning externally.
- No built-in temporal drift tracking (e.g., "show me drift over the
  last 6 months").
- No automatic alerting when drift exceeds thresholds.

### Upgrade Path
Future versions could:
1. Accept a registry of baselines (similar to ModelRegistry) that maps
   model_id → reference distribution snapshot.
2. Integrate with Databricks monitoring APIs to pull reference
   distributions automatically.
3. Add a `drift_history` tool that accepts a list of timestamped
   snapshots and shows drift evolution over time.

These are additive changes that do not require breaking the current API.
