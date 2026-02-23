# Scalability Path: From Local PoC to Production Service

> **Audience:** Engineering leads, data science managers, and architects  
> **Purpose:** Show that this prototype is not a throwaway — it is a foundation.  
> **Bottom line:** The Python code, test suite, and user experience are production-ready today.
> What changes for production is infrastructure, not logic.

---

## 1. What We Built (The PoC)

A local MCP server that exposes ML model explainability as plain-English narratives,
callable from VS Code Copilot agent mode via natural language.

**What it does today:**
- Accepts natural language questions: *"Why was well #47 flagged as high risk?"*
- Calls the right explainability tool automatically (SHAP, PDP/ICE, global importance)
- Returns a deterministic, reproducible English narrative backed by quantitative evidence
- Supports two model types (XGBoost, RandomForest) with identical interfaces
- Every response carries a full audit trail: model ID, timestamp, tool version, data hash

**What it runs on today:**
- Developer laptop (Windows/Mac/Linux)
- Transport: stdio (local process)
- Models: pickle files on disk
- Data: CSV files on disk
- Auth: none (local only)
- Users: one (the developer running VS Code)

---

## 2. What Changes for Production

Each row below is an independent upgrade. They can be done in any order.

| Component | PoC (Now) | Production | Effort | Owner |
|---|---|---|---|---|
| **Transport** | stdio (local process) | Streamable HTTP | 1–2 days | Platform |
| **Hosting** | Developer laptop | Databricks App or container | 3–5 days | Platform |
| **Models** | `.joblib` files on disk | MLflow Model Registry | 2–3 days | ML Eng |
| **Data** | CSV files on disk | Unity Catalog tables | 2–3 days | Data Eng |
| **Auth** | None | OAuth via Databricks SSO | 3–5 days | Platform |
| **Concurrency** | Single user | Multi-user (HTTP handles this) | 0 days (free with transport change) | — |
| **Governance** | None | Unity Catalog permissions + audit logging | 3–5 days | Governance |
| **CI/CD** | Manual `uv run pytest` | GitHub Actions pipeline | 1–2 days | DevOps |

**Total estimated effort to production: 2–4 weeks** depending on platform team availability.
The application code requires zero changes for most of these upgrades.

### Transport upgrade (the most important one — it's also the easiest)

The entire user experience — the natural language interface, the English narratives,
the SHAP analysis — stays identical. Only one line changes:

```python
# PoC (today):
mcp.run(transport="stdio")

# Production:
mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
```

The VS Code `mcp.json` configuration changes from pointing at a local process
to pointing at a URL:

```json
// PoC:
{ "command": "uv", "args": ["run", "python", "-m", "xai_toolkit.server"] }

// Production:
{ "url": "https://xai-toolkit.internal.company.com/mcp" }
```

The Copilot user experience is **identical** in both cases.

---

## 3. What Stays the Same (The Important Part)

This is the section that matters most for long-term value.

**Everything in `src/xai_toolkit/` requires no changes for production:**

| Module | Role | Production status |
|---|---|---|
| `explainers.py` | SHAP computation, PDP/ICE, data hashing | ✅ Production-ready today |
| `narrators.py` | Deterministic English generation | ✅ Production-ready today |
| `plots.py` | matplotlib → base64 PNG | ✅ Production-ready today |
| `schemas.py` | Pydantic data contracts | ✅ Production-ready today |
| `registry.py` | Model loading/management | ✅ Interface stable; swap loading source |

**Everything in `tests/` transfers directly to CI:**
- 99 tests, all passing
- Snapshot tests prevent narrative drift
- Error handling tests prevent silent failures
- Parametrized multi-model tests ensure new models work before deployment

**The user experience is already production-quality:**
- Deterministic: same question + same data = same answer, every time
- Auditable: every response carries model ID, timestamp, tool version, and data hash
- Graceful errors: unknown models, out-of-range indices, and typos all return helpful guidance

---

## 4. Databricks Integration Path

This is the natural production home for this toolkit given existing infrastructure.

### Phase 2: MCP Server on Databricks Apps

```
User (VS Code Copilot)
    │
    │ HTTPS / MCP Streamable HTTP
    ▼
Databricks App (xai-toolkit MCP server)
    │                    │
    ▼                    ▼
MLflow Model Registry   Unity Catalog (feature data)
(pre-trained models)    (production datasets)
```

**What this enables:**
- Any engineer with VS Code + Copilot access can query any registered model
- No data leaves Databricks — the MCP server runs inside the security perimeter
- Model versioning handled by MLflow — the server always serves the "production" alias
- Unity Catalog permissions control which users can query which models

### Phase 2 Registry Change

The `ModelRegistry.load_from_disk()` method gets a sibling:

```python
# Current:
registry.load_from_disk("xgboost_breast_cancer", models_dir, data_dir)

# Phase 2 (interface stays identical for tool handlers):
registry.load_from_mlflow(
    model_id="xgboost_breast_cancer",
    mlflow_uri="databricks://...",
    registered_model_name="well_risk_classifier",
    alias="production",
)
```

Tool handlers call `registry.get("xgboost_breast_cancer")` in both cases —
they are completely unaware of whether the model came from disk or MLflow.

### Phase 2 Data Change

```python
# Current:
X_test = pd.read_csv("data/breast_cancer_test_X.csv")

# Phase 2:
X_test = spark.table("catalog.schema.well_features_test").toPandas()
```

One line in `load_from_mlflow()`. Nothing else changes.

---

## 5. Integration with the Existing Kedro XAI Pipeline

An existing pipeline (`EMOrg-Prd/xai-xgboost-clf`) already computes SHAP values
in production batch runs. This toolkit does **not** replace it. They solve
different halves of the same problem:

```
EXISTING PIPELINE (the engine)         THIS MCP SERVER (the steering wheel)
─────────────────────────────          ──────────────────────────────────────
Batch computation (kedro run)     →    Interactive queries (natural language)
Output: SHAP artifacts on disk    →    Output: English narratives in chat
Audience: data scientists         →    Audience: any engineer with VS Code
Invocation: CLI + config          →    Invocation: "Why was this well flagged?"
Strength: reproducible compute    →    Strength: accessible communication
```

**Phase 2 integration:** The MCP server reads pre-computed SHAP artifacts from
the Kedro pipeline's output store (MLflow / artifact store) instead of computing
SHAP on demand. The pipeline is the computation engine; this is the access layer.

```
Kedro pipeline (nightly batch)
    → computes SHAP for all wells
    → saves artifacts to MLflow         ← already exists and works

MCP Server (on-demand, interactive)
    → receives "Why was well #47 flagged?"
    → reads pre-computed SHAP from MLflow  ← replaces on-demand SHAP call
    → narrates → returns to user
```

**This means:**
- The pipeline's computation stays completely untouched
- The MCP server adds a new consumer, not a new computation path
- Latency improves dramatically (read vs. compute)
- The Kedro pipeline's existing parameterization and hooks become integration points

---

## 6. Multi-Agent Architecture (When It Becomes Relevant)

**Current:** Single MCP server, single LLM routes via tool descriptions. This is
correct for the current scope — 7 tools, one domain, one team.

**When to evolve:**

```
Phase 1 (Now):
  VS Code Copilot → [xai-toolkit MCP server, 7 tools]

Phase 2 (When tool count exceeds ~15):
  VS Code Copilot → [xai-toolkit MCP server]    explainability domain
                  → [data-access MCP server]     Unity Catalog queries
                  → [compliance MCP server]       audit and reporting

Phase 3 (Autonomous workflows):
  Orchestration layer (LangGraph or similar)
    → spawns specialist agents per domain
    → coordinates multi-step workflows
    → e.g., "Review all flagged wells this week and draft an inspection report"
```

The jump from Phase 1 to Phase 2 requires no changes to the xai-toolkit codebase —
just connecting additional MCP servers to Copilot via `mcp.json`.

---

## Summary

| Question | Answer |
|---|---|
| Is this production-ready code? | **Yes.** The application logic, tests, and schemas are production-quality. |
| What needs to change for production? | Infrastructure only: transport, hosting, model source, data source. |
| How long to production? | **2–4 weeks** with platform team support. |
| Does this replace the existing Kedro pipeline? | **No.** It adds a natural language access layer on top of it. |
| Can we add new models without code changes? | **Yes.** Register them in `server.py` startup. All tools work automatically. |
| What happens if a tool fails? | Structured error with guidance. The LLM presents it helpfully. |
| Can we audit what explanation was given? | **Yes.** Every response includes model ID, timestamp, tool version, and data hash. |
