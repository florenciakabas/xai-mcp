# ADR-004: Pydantic Schemas as Single Source of Truth for Tool Contracts

## Status: Accepted
## Date: 2025-02-22

## Context
We have three layers that must agree on data shapes: explainers (produce data),
narrators (consume data), and the MCP server (transport data). Without a single
source of truth, these layers can drift apart silently.

## Decision
All data contracts are defined as Pydantic v2 models in `schemas.py`. Both
the pure Python functions and the MCP server reference these models. If a
schema changes, it changes in one place.

## Consequences

### Positive
- ✅ Automatic JSON schema generation for MCP tool descriptions
- ✅ Runtime validation catches contract violations immediately
- ✅ IDE autocompletion and type checking across all layers
- ✅ Single file to review when auditing data contracts
- ✅ `.model_dump()` provides consistent serialization

### Negative
- ⚠️ Adding a new field requires updating the Pydantic model, not just the function
- ⚠️ Pydantic v2 migration required if starting from v1 codebase

## Alternatives Considered
1. **TypedDict** — Rejected: no runtime validation, no JSON schema generation
2. **dataclasses** — Rejected: weaker validation, no auto JSON schema
3. **Plain dicts** — Rejected: no type safety, contracts are implicit and fragile
