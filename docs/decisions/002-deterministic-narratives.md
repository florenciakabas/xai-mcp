# ADR-002: Deterministic Narratives Over LLM-Generated Interpretation

## Status: Accepted
## Date: 2025-02-21

## Context
Our MCP tools return explainability results to an LLM (Sonnet 4.5 in Copilot).
The question is: who generates the English explanation — the LLM or our code?

SHAP values are precise numerical quantities. A SHAP value of +0.23 for
"water_cut" means exactly: this feature pushed the prediction 0.23 toward the
positive class, relative to the base rate. Allowing an LLM to interpret these
values introduces risk of hallucination, inconsistency, or loss of precision.

## Decision
All English narratives are generated **deterministically by Python code** in
`narrators.py`. The LLM's role is limited to:
1. Understanding the user's question and choosing the right tool
2. Presenting the pre-computed narrative conversationally

The LLM does NOT interpret, rephrase, or recompute the narrative content.

## Consequences

### Positive
- ✅ **Reproducible:** same data + same model = same English explanation, every time
- ✅ **Auditable:** narratives can be snapshot-tested and version-controlled
- ✅ **Trustworthy:** no risk of LLM hallucinating feature importance or direction
- ✅ **Testable:** `pytest` can assert exact narrative content
- ✅ **Regulatory-friendly:** consistent outputs support compliance requirements

### Negative
- ⚠️ Narratives may sound less "natural" than LLM-generated prose
- ⚠️ Template maintenance required when adding new explanation types
- ⚠️ Less flexible for unexpected question formats (LLM could adapt, templates can't)

### Mitigations
- Invest in well-crafted templates that read naturally
- The LLM still adds conversational polish when presenting (greeting, context, follow-up)
- Copilot instructions tell the LLM to present narratives verbatim (see copilot-instructions.md)

## Alternatives Considered
1. **LLM generates explanations from raw SHAP values** — Rejected: non-reproducible,
   risks hallucination, untestable output, fails audit requirements.
2. **Hybrid: Python generates data, LLM generates prose** — Rejected: still
   non-reproducible. Even small LLM rephrasing can change meaning of quantitative
   statements. Gains in naturalness don't justify loss of determinism.
3. **LLM post-processes with strict constraints** — Rejected: constraints are
   brittle; "don't change the numbers" is hard to enforce reliably across LLM versions.
