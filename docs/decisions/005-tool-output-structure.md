# ADR-005: Consistent Tool Output Structure (narrative + evidence + metadata)

## Status: Accepted
## Date: 2025-02-22

## Context
We have 6 MCP tools, each returning different types of data. The consuming
LLM (Sonnet 4.5 in Copilot) needs a predictable response shape to present
results consistently. Without a standard structure, each tool becomes a
special case requiring unique handling logic.

## Decision
Every MCP tool returns a `ToolResponse` with five fields:

```python
{
    "narrative":   str,   # Plain English interpretation (REQUIRED)
    "evidence":    dict,  # Structured numeric data backing the narrative
    "metadata":    dict,  # Audit trail: model_id, timestamp, tool_version, data_hash
    "plot_base64": str,   # Optional base64-encoded PNG visualization
    "grounded":    bool,  # Always True for tool responses (epistemic label)
}
```

The `narrative` is the primary output — what the LLM presents to the user.
The `evidence` is structured data for programmatic consumption or follow-up.
The `metadata` enables auditability and reproducibility.
The `grounded` flag is an epistemic label for the consuming LLM: `True` signals
that this answer was computed deterministically from a registered model and is
audit-ready. The LLM is instructed to prepend a disclaimer on any response it
generates *without* calling a tool — i.e. when `grounded=True` is absent from
the response entirely. This gives users a visible, consistent signal about
whether they are reading a verified computation or general knowledge.

## Consequences

### Positive
- ✅ LLM always knows to look for `narrative` and present it
- ✅ Copilot instructions can say "present the narrative verbatim" — one rule
- ✅ Audit trail is automatic for every tool call
- ✅ Evidence enables follow-up questions without recomputation
- ✅ Structure is extensible — new fields don't break existing consumers

### Negative
- ⚠️ Some tools return very different evidence shapes (SHAP values vs. PDP curves)
- ⚠️ `evidence` is typed as `dict` rather than a specific model (flexibility vs. safety)

## Alternatives Considered
1. **Free-form dict per tool** — Rejected: LLM would need per-tool parsing logic
2. **Narrative only** — Rejected: loses structured data for programmatic use
3. **Strongly typed evidence per tool** — Considered for future: would require
   discriminated union or per-tool response types. Current `dict` is pragmatic for PoC.
