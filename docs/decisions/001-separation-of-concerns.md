# ADR-001: Separation of Pure Functions from MCP Layer

## Status: Accepted
## Date: 2025-02-21

## Context
We need to expose ML explainability tools via MCP for use in VS Code Copilot.
The question is whether the explainability logic should live inside the MCP
tool handlers or be separated into independent functions.

## Decision
Explainability logic lives in pure Python functions (`explainers.py`,
`narrators.py`, `plots.py`) with **zero MCP imports**. The MCP server
(`server.py`) is a thin adapter that calls these functions and wraps
their outputs for the MCP protocol.

## Consequences

### Positive
- ✅ Functions are testable with `pytest` alone — no MCP server needed
- ✅ Functions are reusable outside MCP (notebooks, REST APIs, Kedro nodes, scripts)
- ✅ MCP layer can be swapped (stdio → HTTP) without touching business logic
- ✅ Clear dependency direction: `server.py` depends on pure functions, never reverse
- ✅ Aligns with existing Kedro pipeline's modular node design (future integration)

### Negative
- ⚠️ Slight overhead of maintaining two layers (adapter + pure functions)
- ⚠️ Must ensure Pydantic schemas stay in sync between MCP tool definitions
  and pure function signatures

### Mitigations
- Pydantic schemas (`schemas.py`) serve as the single source of truth (see ADR-004)
- Tests validate both layers independently and together

## Alternatives Considered
1. **Logic directly in MCP tool handlers** — Rejected: creates tight coupling,
   makes functions untestable without an MCP server, prevents reuse.
2. **Framework that merges both layers** — Rejected: reduces portability and
   makes it harder to swap transports or integrate with other systems.
