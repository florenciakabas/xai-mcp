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

# Run a specific test file or keyword
uv run python -m pytest tests/test_<module>.py -v
uv run python -m pytest -k "<keyword>" -v

# Train models (only needed if models/ is empty)
uv run python scripts/train_toy_model.py      # XGBoost
uv run python scripts/train_rf_model.py       # RandomForest
uv run python scripts/train_lubricant_model.py # LightGBM

# Start the MCP server locally
uv run python -m xai_toolkit.server

# CLI interface (same functions, no MCP dependency)
uv run python -m xai_toolkit.cli models
uv run python -m xai_toolkit.cli explain --model xgboost_breast_cancer --sample 0
uv run python -m xai_toolkit.cli drift --model xgboost_breast_cancer
uv run python -m xai_toolkit.cli feature-drift --model xgboost_breast_cancer --feature "mean radius"
uv run python -m xai_toolkit.cli context --query "high risk biopsy"
```

## Architecture (seven layers, one direction)

```
User question → server.py → explainers.py → narrators.py → response
                    ↓        drift.py ↗          ↑
               schemas.py      plots.py
                    ↓
              registry.py
```

- **`server.py`** — MCP adapter. Thin routing only. No computation here.
- **`explainers.py`** — SHAP/PDP computation + data hashing + intrinsic
  importance extraction. Two modes: on-the-fly (`compute_shap_values`) or from
  pre-computed pipeline artifacts (`load_shap_from_pipeline`). Both return the
  same `ShapResult` schema. `extract_intrinsic_importances()` handles models
  with `coef_` or `feature_importances_` (adapted from Tamas's Kedro pipeline).
  On-the-fly SHAP uses training data as background when available (TD-14 fix).
- **`drift.py`** — Pure drift detection (ADR-011). Thin scipy wrappers:
  KS test, PSI, chi-squared. Auto-selects test by dtype. Stateless — caller
  provides both reference and current DataFrames. Same role as `explainers.py`.
- **`narrators.py`** — Structured data → deterministic English. No LLM calls.
  Includes `narrate_intrinsic_importance()` for coefficient/importance narratives,
  and `narrate_feature_drift()` / `narrate_dataset_drift()` for drift results.
- **`plots.py`** — matplotlib → base64 PNG. Three types: SHAP bar, waterfall,
  PDP+ICE overlay.
- **`schemas.py`** — Pydantic models. Single source of truth for all contracts.
- **`registry.py`** — `ModelRegistry`: load/list/get models. Central loader.
  Loads `X_train` alongside `X_test` when available (for SHAP background).
- **`pipeline_compat.py`** — Bridge to colleague's Kedro pipeline artifacts.
  Reads `shap_values.npy`, `shap_expected_value.npy`, and `shap_metadata.json`.
  Includes `_detect_model_type()` attributed to Tamás's original implementation.
- **`knowledge.py`** — Business context retrieval (ADR-009). Loads markdown
  documents from `knowledge/`, chunks by heading, searches via TF-IDF.
  Pure Python, no MCP imports. Supports "Glass Floor" provenance pattern.
- **`cli.py`** — CLI adapter. Same pure functions as `server.py`, outputs
  JSON to stdout. Zero MCP dependency. For scripting, CI/CD, and
  environments where MCP servers are unavailable.
- **`result_store.py`** — Persists explanation/drift results to JSON for
  cross-session retrieval via `list_drift_alerts` / `list_explained_samples`.
- **`store_queries.py`** — Query helpers for the result store (filtering,
  sorting, aggregation).
- **`store_briefing.py`** — Generates `standard_briefing` narratives from
  stored results.
- **`skill_registry.py`** — Discovers and serves `SKILL.md` files for the
  `list_skills` / `get_skill` tools.
- **`notebook_wrappers.py`** — Convenience wrappers for notebook workflows
  (register_in_memory, one-call explain).
- **`sampling.py`** — Smart sample selection (outliers, boundary cases,
  high-SHAP samples) for batch explanation.
- **`kedro_adapter/`** — Kedro pipeline nodes that wrap xai-toolkit pure
  functions for integration with existing Kedro pipelines.

## MCP Tools (18 total)

| Tool | Question it answers | Plot |
|---|---|---|
| `list_models` | What models are available? | — |
| `describe_dataset` | What does the data look like? | — |
| `summarize_model` | What does this model do overall? | — |
| `compare_features` | Which features matter most? | — |
| `explain_prediction` | Why was sample N classified as X? | SHAP bar |
| `explain_prediction_waterfall` | Show the full SHAP breakdown | Waterfall |
| `get_partial_dependence` | How does feature F affect predictions? | PDP+ICE |
| `compare_predictions` | Do two models agree on sample N? | — |
| `detect_drift` | Has the model's input data drifted? (ADR-011) | — |
| `detect_feature_drift` | How has feature F's distribution changed? (ADR-011) | — |
| `retrieve_business_context` | What should I do about this? (ADR-009) | — |
| `get_xai_methodology` | How should I sequence my analysis? | — |
| `get_glass_floor` | How do I separate model facts from business context? | — |
| `list_drift_alerts` | Which features are currently drifting? | — |
| `list_explained_samples` | What samples have been explained recently? | — |
| `standard_briefing` | Give me a quick model health summary | — |
| `list_skills` | What analysis skills are available? | — |
| `get_skill` | Show me the details of a specific skill | — |

## Tool Output Contract

Every **success** returns: `narrative`, `evidence`, `metadata`, `plot_base64`,
`grounded` (always `True`). Every **failure** returns: `error_code`, `message`,
`available`, `suggestion`. See `schemas.py` for exact Pydantic definitions.

`retrieve_business_context` returns a `KnowledgeSearchResult` (not `ToolResponse`)
with `provenance_label: "ai-interpreted"`. See the Glass Floor Protocol in
`.github/copilot-instructions.md` for behavioral presentation rules.

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
| 010 | Kedro pipeline integration map — function-by-function mapping |
| 011 | Stateless drift detection — caller provides both DataFrames |

## Server Initialization

The server uses a factory pattern (TD-12): importing `server.py` does NOT
load models or knowledge. Call `init_server()` explicitly before running:

```python
from xai_toolkit.server import init_server, mcp
init_server()          # loads models + knowledge base
mcp.run(transport=...) # start serving
```

The `if __name__ == "__main__"` block handles this automatically.
Tests use a session-scoped `conftest.py` fixture that calls `init_server()`.

### Configurable Transport

Set `XAI_TRANSPORT` env var to switch between stdio (default) and HTTP:

```bash
# stdio (default, VS Code Copilot)
uv run python -m xai_toolkit.server

# HTTP (Databricks Apps)
XAI_TRANSPORT=http XAI_PORT=9000 uv run python -m xai_toolkit.server
```

## Registered Models

| Model ID | Framework | Domain |
|---|---|---|
| `xgboost_breast_cancer` | XGBoost | Breast cancer screening |
| `rf_breast_cancer` | RandomForest | Breast cancer screening |
| `lgbm_lubricant_quality` | LightGBM | Lubricant quality (O&G demo) |

## How to Add a New Model

```bash
# 1. Train and save (adapt train_rf_model.py for your model type)
uv run python scripts/train_rf_model.py

# 2. Add one line to init_server() in server.py:
registry.load_from_disk("your_model_id", models_dir, data_dir)

# 3. Run the test suite — all tools work automatically:
uv run python -m pytest tests/test_second_model.py -v
```

No other code changes needed. The registry + parametrized tests handle the rest.

## Coding Standards

- Python 3.11+, type hints on all public signatures
- Pydantic v2 for all data contracts
- pytest with parametrized tests for multi-model coverage
- Google-style docstrings on all public functions
- Tests cover narrators, explainers, drift, plots, snapshots, reproducibility,
  pipeline compat, knowledge, CLI, server errors, second model, comparisons,
  intrinsic importances, and SHAP background data

## What NOT to Do

- **No computation in `server.py`** — it is an adapter only
- **No LLM calls in `narrators.py`** — narratives must be deterministic
- **No MCP/FastMCP imports outside `server.py`**
- **No changes to tool output structure without updating `schemas.py` first**
- **No new narrator without a snapshot test** to lock its output
- **No hard-coded file paths in tests** — use fixtures and `pathlib`
- **Do not auto-generate or overwrite this file** — update it alongside code changes
