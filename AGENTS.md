# AGENTS.md — Instructions for AI Agents Working in This Repository

## Project Summary (one sentence)

**xai-toolkit** exposes ML model explainability as plain-English narratives
via MCP, so an LLM in VS Code Copilot can answer "Why did the model flag
this well as high risk?" with a deterministic, SHAP-backed explanation.

## Skills

| Skill | Location | When to use |
|---|---|---|
| `xai-workflow` | `skills/xai-workflow/SKILL.md` | Conducting explainability analyses — defines tool sequencing, Glass Floor protocol, and abbreviation rules |

Read the relevant skill before starting analysis work. The skill contains
procedural expertise that ensures consistent, high-quality explanations.

## Commands

```bash
# Run all tests (use this exact form — avoids Windows canonicalize-path errors)
uv run python -m pytest

# Run a specific test file
uv run python -m pytest tests/test_narrators.py -v

# Run tests matching a keyword
uv run python -m pytest -k "waterfall" -v

# Train models (only needed if models/ is empty)
uv run python scripts/train_toy_model.py      # XGBoost
uv run python scripts/train_rf_model.py       # RandomForest

# Start the MCP server locally
uv run python -m xai_toolkit.server

# CLI interface (same functions, no MCP dependency)
uv run python -m xai_toolkit.cli models
uv run python -m xai_toolkit.cli explain --model xgboost_breast_cancer --sample 0
uv run python -m xai_toolkit.cli context --query "high risk biopsy"
```

## The Golden Rule

**The LLM is the presenter, not the analyst.**

All analysis is done by pure Python functions. The LLM's only job is to
choose the right MCP tool and present the pre-computed result conversationally.
Same question + same data = same answer, always.

## Architecture (six layers, one direction)

```
User question → server.py → explainers.py → narrators.py → response
                    ↓              ↓
               schemas.py      plots.py
                    ↓
              registry.py
```

- **`server.py`** — MCP adapter. Thin routing only. No computation here.
- **`explainers.py`** — SHAP/PDP computation + data hashing. Two modes:
  on-the-fly (`compute_shap_values`) or from pre-computed pipeline artifacts
  (`load_shap_from_pipeline`). Both return the same `ShapResult` schema.
- **`narrators.py`** — Structured data → deterministic English. No LLM calls.
- **`plots.py`** — matplotlib → base64 PNG. Three types: SHAP bar, waterfall,
  PDP+ICE overlay.
- **`schemas.py`** — Pydantic models. Single source of truth for all contracts.
- **`registry.py`** — `ModelRegistry`: load/list/get models. Central loader.
- **`pipeline_compat.py`** — Bridge to colleague's Kedro pipeline artifacts.
  Reads `shap_values.npy`, `shap_expected_value.npy`, and `shap_metadata.json`.
  Includes `_detect_model_type()` attributed to Tamás's original implementation.
- **`knowledge.py`** — Business context retrieval (ADR-009). Loads markdown
  documents from `knowledge/`, chunks by heading, searches via TF-IDF.
  Pure Python, no MCP imports. Supports "Glass Floor" provenance pattern.
- **`cli.py`** — CLI adapter. Same pure functions as `server.py`, outputs
  JSON to stdout. Zero MCP dependency. For scripting, CI/CD, and
  environments where MCP servers are unavailable.

## MCP Tools (8 total)

| Tool | Question it answers | Plot |
|---|---|---|
| `list_models` | What models are available? | — |
| `describe_dataset` | What does the data look like? | — |
| `summarize_model` | What does this model do overall? | — |
| `compare_features` | Which features matter most? | — |
| `explain_prediction` | Why was sample N classified as X? | SHAP bar |
| `explain_prediction_waterfall` | Show the full SHAP breakdown | Waterfall |
| `get_partial_dependence` | How does feature F affect predictions? | PDP+ICE |
| `retrieve_business_context` | What should I do about this? (ADR-009) | — |

## Tool Output Contract

Every **success** returns: `narrative`, `evidence`, `metadata`, `plot_base64`,
`grounded` (always `True`). Every **failure** returns: `error_code`, `message`,
`available`, `suggestion`. See `schemas.py` for exact Pydantic definitions.

The `grounded: True` flag signals deterministic computation. If the LLM answers
without calling a tool, it must prepend a ⚠️ disclaimer.

`retrieve_business_context` returns a `KnowledgeSearchResult` (not `ToolResponse`)
with `provenance_label: "ai-interpreted"`. The Glass Floor Protocol (ADR-009)
requires the LLM to present deterministic output (Layer 1) separately from
business context (Layer 2). See `.github/copilot-instructions.md` rule 6.

## Design Decisions (docs/decisions/)

| ADR | Decision |
|---|---|
| 001 | Pure functions separated from MCP layer |
| 002 | Deterministic narratives — no LLM calls |
| 003 | stdio transport (PoC) → Streamable HTTP (production) |
| 004 | Pydantic schemas as single source of truth |
| 005 | Every tool returns narrative + evidence + metadata + optional plot |
| 006 | ModelRegistry pattern — one central loader |
| 007 | Single-agent architecture — LLM routes via tool descriptions |
| 008 | Pipeline bridge — read pre-computed SHAP artifacts, don't recompute |
| 009 | RAG augmentation with Glass Floor provenance separation |

## How to Add a New Model

```bash
# 1. Train and save (adapt train_rf_model.py for your model type)
uv run python scripts/train_rf_model.py

# 2. Add one line to server.py startup:
registry.load_from_disk("your_model_id", MODELS_DIR, DATA_DIR)

# 3. Run the test suite — all tools work automatically:
uv run python -m pytest tests/test_second_model.py -v
```

No other code changes needed. The registry + parametrized tests handle the rest.

## Coding Standards

- Python 3.11+, type hints on all public signatures
- Pydantic v2 for all data contracts
- pytest with parametrized tests for multi-model coverage
- Google-style docstrings on all public functions
- Scenario YAML written before implementing each feature (see `scenarios/`)

## What NOT to Do

- **No computation in `server.py`** — it is an adapter only
- **No LLM calls in `narrators.py`** — narratives must be deterministic
- **No MCP/FastMCP imports outside `server.py`**
- **No changes to tool output structure without updating `schemas.py` first**
- **No new narrator without a snapshot test** to lock its output
- **No hard-coded file paths in tests** — use fixtures and `pathlib`
- **Do not auto-generate or overwrite this file** — update it alongside code changes
