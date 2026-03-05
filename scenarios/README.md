# Scenarios — Behavioral Acceptance Criteria

> Scenario-based specifications written *before* implementation.
> Each YAML file describes a user-observable behavior and the conditions
> that must hold. These serve as both acceptance criteria and the
> blueprint for automated tests.

## Directory Structure

| Directory | Concern | Scenarios |
|---|---|---|
| **tools/** | Individual tool behavioral specs | 7 — one per tool + multi-turn flow |
| **error-handling/** | Structured error responses | 3 — model not found, sample out of range, invalid feature |
| **auditability/** | Reproducibility, metadata, snapshots | 3 — deterministic output, audit trail, narrative drift detection |
| **multi-model/** | Model-agnostic verification | 2 — second model works, cross-model comparison |
| **rag/** | Knowledge retrieval & Glass Floor | 4 — loading, search, provenance, graceful degradation |
| **cli/** | CLI adapter parity | 1 — identical output shape to MCP server |
| **onboarding/** | Setup, docs, demo readiness | 6 — zero-setup, golden path demo, handoff, docs completeness |

## Scenario ID Convention

Each scenario has a unique ID. The original sprint-day prefix (e.g., `D1-S2`) is
preserved for traceability to sprint planning, but scenarios are now organized
by the *concern* they validate rather than when they were written.

| Prefix | Origin |
|---|---|
| `D1-S*` through `D5-S*` | Sprint days 1–5 |
| `CP-S*` | Compare predictions (post-sprint) |
| No prefix | RAG and CLI scenarios (phase-based) |
