"""Instruction-serving pure functions (ADR-001: no MCP imports).

These functions provide methodology and protocol content to LLMs that
don't have access to AGENTS.md or SKILL.md (e.g., Databricks Apps).
"""

from pathlib import Path

_GLASS_FLOOR_CONTENT = """\
# Glass Floor Separation Principles (ADR-009)

The Glass Floor protocol ensures transparent separation between deterministic
model outputs and AI-interpreted business context.

## Layer 1 — Deterministic (Model Explanation)

- Computed by pure Python functions, reproducible, audit-ready.
- Present the tool narrative EXACTLY as returned — do not modify, paraphrase,
  or supplement SHAP values beyond what the narrative states.
- Every tool response includes `grounded: true`.
- Label: 📊 **Model Explanation** (deterministic, grounded)

## Layer 2 — AI-Interpreted (Business Context)

- Retrieved from business documents via `retrieve_business_context`.
- Provenance label is always `ai-interpreted`.
- Any synthesis by the LLM is NOT deterministic and must be clearly
  distinguished from grounded tool outputs.
- Label: 📋 **Business Context** (AI-interpreted from [source_document], [section])

## Separation Rules

1. **Never blend** Layer 1 and Layer 2 into a single paragraph.
2. **Never modify** the deterministic narrative based on business context.
3. Layer 2 content should be **verified by the user** before acting on it.
4. If `retrieve_business_context` returns low-relevance results, say so
   rather than presenting weakly matched content as authoritative.
5. The deterministic explanation must **stand alone and unmodified**.
   Business context is ADDITIVE only.

## Why "Glass Floor"?

The name means the separation is transparent: users can always see which
layer produced which content and verify accordingly. The floor is glass —
you can see through it, but it holds firm. Deterministic facts stay on top;
interpreted context stays below, clearly visible but structurally separate.

## Provenance

- Layer 1: `grounded: true` — computed deterministically from registered model
- Layer 2: `provenance_label: "ai-interpreted"` — synthesized from retrieved
  business documents, always citing `source_document` and `section_heading`
"""


def get_methodology_content(root_dir: Path) -> str:
    """Read the XAI workflow methodology from SKILL.md.

    Args:
        root_dir: Project root directory containing skills/xai-workflow/SKILL.md.

    Returns:
        The full content of SKILL.md.

    Raises:
        FileNotFoundError: If SKILL.md is missing.
    """
    path = root_dir / "skills" / "xai-workflow" / "SKILL.md"
    return path.read_text(encoding="utf-8")


def get_glass_floor_principles() -> str:
    """Return curated Glass Floor principles (ADR-009 + SKILL.md protocol).

    Returns:
        A string describing the Glass Floor separation principles.
    """
    return _GLASS_FLOOR_CONTENT
