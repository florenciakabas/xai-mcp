# xai-toolkit

**ML model explainability as plain-English narratives, exposed via MCP.**

Ask your model *why* in natural language. Get a deterministic, reproducible
English answer backed by SHAP analysis — directly inside VS Code Copilot.

```
User: "Why was patient 42 classified as malignant?"

Copilot: The model classified this sample as malignant (probability: 0.91)
         primarily because of 3 factors: worst_radius = 23.4 (pushing toward
         the positive class by +0.28), worst_concave_points = 0.18 (+0.19),
         and mean_concavity = 0.15 (+0.14). The top opposing factor is
         mean_smoothness = 0.08 (pushing away from the positive class by -0.06).
```

No SHAP plots to decipher. No code to write. English that a decision-maker can act on.

---

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Train the toy models (run once)
uv run python scripts/train_toy_model.py
uv run python scripts/train_rf_model.py

# 3. Run the test suite (should show 100+ passing)
uv run python -m pytest tests/ -v

# 4. Open VS Code — the MCP server starts automatically via .vscode/mcp.json
#    Open Copilot chat in agent mode and ask:
#    "What models are available?"
```

**Requirements:** Python 3.11+, [uv](https://docs.astral.sh/uv/), VS Code with GitHub Copilot

---

## What It Does

Seven MCP tools answer the most common explainability questions:

| Ask | Tool | Returns |
|---|---|---|
| "What models are available?" | `list_models` | Model IDs, types, accuracy |
| "Tell me about the data" | `describe_dataset` | Sample count, class distribution, stats |
| "What does this model do?" | `summarize_model` | Model type, accuracy, top 5 features |
| "Which features matter most?" | `compare_features` | Ranked feature importance with magnitudes |
| "Why was sample N classified as X?" | `explain_prediction` | Narrative + SHAP bar chart |
| "Show me the full SHAP breakdown" | `explain_prediction_waterfall` | Narrative + waterfall plot |
| "How does feature F affect predictions?" | `get_partial_dependence` | Narrative + PDP/ICE plot |

Every response includes a complete audit trail: model ID, timestamp, tool version,
and a SHA256 hash of the input data. Same question + same data = same answer, every time.

---

## Architecture

```
VS Code Copilot (Sonnet 4.5)
    │  natural language question
    ▼
MCP Client (built into Copilot)
    │  JSON-RPC over stdio
    ▼
xai-toolkit MCP Server  (server.py — thin adapter only)
    │
    ├── explainers.py   SHAP values, PDP/ICE, global importance, data hashing
    ├── narrators.py    Structured data → deterministic English paragraphs
    ├── plots.py        matplotlib → base64 PNG (bar, waterfall, PDP+ICE)
    ├── schemas.py      Pydantic contracts — single source of truth
    └── registry.py     ModelRegistry — load and serve multiple model types
```

**Design principle:** The LLM is the presenter, not the analyst.
All computation and narrative generation happens in pure Python.
The LLM chooses the right tool and wraps the pre-computed result conversationally.
This guarantees reproducibility — the LLM cannot hallucinate SHAP values.

---

## Project Layout

```
xai-mcp/
├── src/xai_toolkit/         # Source package
├── tests/                   # 100+ pytest tests
├── docs/
│   ├── decisions/           # 7 Architecture Decision Records
│   └── scalability-path.md  # PoC → Production roadmap
├── scenarios/               # YAML acceptance criteria (day1–day5)
├── scripts/                 # Model training scripts
├── models/                  # Trained model artifacts
├── data/                    # Test datasets
└── .vscode/mcp.json         # MCP server configuration
```

---

## Running Tests

```bash
# Full test suite
uv run python -m pytest tests/ -v

# Write snapshot golden files (run once after first install)
uv run python -m pytest tests/test_snapshots.py --snapshot-update -v

# Just the fast unit tests (no model loading)
uv run python -m pytest tests/test_narrators.py tests/test_explainers.py tests/test_reproducibility.py -v

# Second model integration tests (requires trained RF model)
uv run python -m pytest tests/test_second_model.py -v
```

---

## Adding a New Model

1. Train and save your model following the convention in `scripts/train_rf_model.py`
2. Add one line to the startup block in `server.py`:
   ```python
   registry.load_from_disk("your_model_id", MODELS_DIR, DATA_DIR)
   ```
3. Run `uv run python -m pytest tests/test_second_model.py -v` — all tools
   should work for your new model with zero code changes

See `AGENTS.md` for full coding standards and architecture guidance.

---

## Architecture Decision Records

Seven decisions documented in `docs/decisions/`:

| ADR | Decision |
|---|---|
| 001 | Pure functions separated from MCP layer |
| 002 | Deterministic narratives — no LLM calls |
| 003 | stdio → Streamable HTTP migration path |
| 004 | Pydantic schemas as single source of truth |
| 005 | Consistent tool output structure |
| 006 | ModelRegistry pattern |
| 007 | Single-agent architecture |

---

## Production Path

This local PoC becomes a production service by changing infrastructure, not code.
See [`docs/scalability-path.md`](docs/scalability-path.md) for the full roadmap,
including Databricks integration, MLflow model registry, Unity Catalog data access,
and the integration path with the existing Kedro-based XAI pipeline.

**Estimated effort to production: 2–4 weeks** (platform team, not application code).

---

## Related

- [FastMCP](https://github.com/jlowin/fastmcp) — MCP server framework used here
- [SHAP](https://shap.readthedocs.io/) — explainability library
- [Model Context Protocol](https://modelcontextprotocol.io/) — the standard this implements
