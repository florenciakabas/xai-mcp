# AGENTS.md — Source of Truth for AI Agents in This Repository

## What This Project Is

**xai-toolkit** is a Python package that provides ML model explainability as
plain-English narratives backed by quantitative evidence. It is exposed as an
MCP (Model Context Protocol) server that LLMs in VS Code Copilot agent mode
can call directly.

The user asks a question like "Why did the model flag this well as high risk?"
and gets back a deterministic English explanation powered by SHAP analysis — no
plots to decipher, no code to run.

## Architecture — The Golden Rule

**The LLM is the presenter, not the analyst.**

All analysis is done by pure Python functions. The LLM's only job is:
1. Understand the user's question
2. Choose the right MCP tool
3. Present the pre-computed result conversationally

This makes outputs **reproducible**: same question + same data = same answer, always.

## Project Structure

```
xai-mcp/
├── src/xai_toolkit/
│   ├── explainers.py        # SHAP/PDP computation + data hashing (no MCP)
│   ├── narrators.py         # Structured data → English (deterministic, no LLM)
│   ├── plots.py             # matplotlib → base64 PNG (3 plot types)
│   ├── schemas.py           # Pydantic models — single source of truth
│   ├── registry.py          # ModelRegistry: load/list/get models
│   └── server.py            # MCP server — thin adapter ONLY
├── tests/
│   ├── test_narrators.py    # 32 narrator unit tests
│   ├── test_plots.py        # 12 plot tests
│   ├── test_explainers.py   # 8 data hash tests
│   ├── test_reproducibility.py  # 8 reproducibility tests (D3-S1)
│   ├── test_server_errors.py    # ~28 structured error tests (D3-S3–S5)
│   ├── test_snapshots.py        # 5 golden-file snapshot tests (D3-S6)
│   └── test_second_model.py     # ~17 multi-model parametrized tests (D4-S1)
├── models/                  # xgboost_breast_cancer.joblib, rf_breast_cancer.joblib
├── data/                    # breast_cancer_test_X.csv, breast_cancer_test_y.csv
├── docs/
│   ├── decisions/           # ADR-001 through ADR-007
│   └── scalability-path.md  # PoC → Production roadmap
├── specs/tools.spec.md      # Tool specifications (written before implementation)
├── scenarios/               # YAML acceptance criteria (Holdout Pattern), day1–day5
├── scripts/
│   ├── train_toy_model.py   # Train XGBoost on breast cancer data
│   └── train_rf_model.py    # Train RandomForest on breast cancer data
└── .vscode/mcp.json         # MCP server config for VS Code Copilot
```

## Available MCP Tools (7 total)

| Tool | Question it answers | Plot |
|---|---|---|
| `list_models` | What models are available? | — |
| `describe_dataset` | What does the data look like? | — |
| `summarize_model` | What does this model do overall? | — |
| `compare_features` | Which features matter most? | — |
| `explain_prediction` | Why was sample N classified as X? | SHAP bar chart |
| `explain_prediction_waterfall` | Show me the full SHAP breakdown | SHAP waterfall |
| `get_partial_dependence` | How does feature F affect predictions? | PDP + ICE overlay |

## Tool Output Contract (ADR-005)

Every successful tool call returns:
```python
{
    "narrative":   str,   # Plain English interpretation — AUTHORITATIVE
    "evidence":    dict,  # Structured numeric data backing the narrative
    "metadata":    dict,  # model_id, model_type, timestamp, tool_version, data_hash
    "plot_base64": str,   # Optional base64 PNG (None if not applicable)
    "grounded":    bool,  # Always True — epistemic label (see below)
}
```

`grounded: True` signals that this answer was computed deterministically from a
registered model. It is present on every tool response by definition — if a tool
was called, computation happened. The LLM is instructed: if it answers a question
without calling a tool, it must prepend a ⚠️ disclaimer, because `grounded: True`
will be absent from its response. This gives users a visible, consistent signal
about whether they are reading a verified result or general knowledge.

Every failed tool call returns:
```python
{
    "error_code": str,        # e.g. "MODEL_NOT_FOUND"
    "message":    str,        # Human-readable explanation
    "available":  list[str],  # Valid options the user can try
    "suggestion": str | None, # Closest match for typo-style errors
}
```

## Key Design Decisions (see docs/decisions/ for full rationale)

| ADR | Decision |
|---|---|
| 001 | Pure functions separated from MCP layer — testable without a server |
| 002 | Deterministic narratives — no LLM calls, same input = same output |
| 003 | stdio transport for PoC → Streamable HTTP for production (one flag) |
| 004 | Pydantic schemas are the single source of truth for all contracts |
| 005 | Every tool returns narrative + evidence + metadata + optional plot |
| 006 | ModelRegistry pattern — one central loader, decoupled from tool handlers |
| 007 | Single-agent architecture — LLM routes via tool descriptions, no supervisor |

## How to Add a New Model

1. Train it and save with the same convention:
   ```bash
   uv run python scripts/train_rf_model.py   # adapt for your model
   ```
2. Add one line to `server.py` startup:
   ```python
   registry.load_from_disk("your_model_id", MODELS_DIR, DATA_DIR)
   ```
3. Run the test suite — all tools should work automatically:
   ```bash
   uv run python -m pytest tests/test_second_model.py -v
   ```
   No other code changes needed.

## Coding Standards

- Python 3.11+, type hints on all signatures
- Pydantic v2 for all data contracts
- pytest for all tests — prefer parametrized tests for multi-model coverage
- Google-style docstrings on all public functions
- No MCP imports outside `server.py`
- No LLM calls in `narrators.py` — narratives must be deterministic
- Write a scenario YAML before implementing each feature

## What NOT to Do

- Do NOT put computation logic in `server.py` — it is an adapter only
- Do NOT use LLM calls to generate narratives
- Do NOT import MCP/FastMCP in any file except `server.py`
- Do NOT modify tool output structure without updating `schemas.py` first
- Do NOT merge a new narrator without a snapshot test to lock its output
