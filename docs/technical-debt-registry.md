# XAI MCP Toolkit — Technical Debt Registry

> **Created:** 2026-03-04
> **Sources:** Claude Code audit + manual codebase review
> **Status:** Active — track remediation as items are resolved
> **Last updated:** 2026-03-08 — TD-14 resolved (X_train as SHAP background)

---

## Summary

The sprint produced a well-architected PoC with strong separation of concerns and comprehensive test coverage. The technical debt below is typical for a 5-day sprint and mostly falls into three buckets: (1) type safety gaps in schemas, (2) missing CI/CD infrastructure, and (3) known PoC shortcuts in the RAG module. Nothing here is blocking the management presentation, but addressing these items before scaling to production will significantly reduce maintenance burden.

---

## TD-01: `evidence: dict` escape hatch in ToolResponse

**Source:** Claude Code audit (ADR-005 acknowledged)
**Severity:** Medium | **Effort:** Moderate | **Area:** Schemas

`ToolResponse.evidence` is typed as `dict`, so Pydantic validates it exists but not its shape. Each tool puts a different structure inside (serialized `ShapResult`, `{"features": [...]}`, partial `PartialDependenceResult`). The LLM consumer must pattern-match on contents.

**Remediation:** Introduce a discriminated union via `Annotated[..., Field(discriminator='evidence_type')]` or per-tool response subtypes. Start by adding an `evidence_type: str` field to ToolResponse that names the schema, then incrementally type each evidence shape. This is ADR-005's documented upgrade path.

**Trade-off:** More correct typing adds complexity and makes adding new tools slightly harder. For PoC, the current approach is pragmatic.

---

## TD-02: No semantic validators on Pydantic fields

**Source:** Claude Code audit
**Severity:** Medium (High in clinical/production context) | **Effort:** Quick | **Area:** Schemas

No `@field_validator` constraints exist. Examples of missing guards:
- `probability` is not bounded to `[0.0, 1.0]`
- `data_hash` is not validated as 64 hex characters
- `relevance_score` is not bounded to `[0.0, 1.0]`

**Remediation:** Add `@field_validator` or `Field(ge=0.0, le=1.0)` constraints to `probability`, `relevance_score`, and a regex pattern validator for `data_hash`. These are quick wins with high safety value.

```python
# Example — 5 minutes per field
probability: float = Field(ge=0.0, le=1.0, description="...")
data_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$", description="...")
```

---

## TD-03: No `Literal` / `Enum` for constrained string fields

**Source:** Claude Code audit
**Severity:** Low-Medium | **Effort:** Quick | **Area:** Schemas

`FeatureImportance.direction` is `str` but should be `Literal["positive", "negative"]`. `ErrorResponse.error_code` lists valid codes in the description but accepts any string. A typo like `"MODL_NOT_FOUND"` would pass validation silently.

**Remediation:** Replace with `Literal` types or `StrEnum`:

```python
from typing import Literal

direction: Literal["positive", "negative"] = Field(...)

class ErrorCode(StrEnum):
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    SAMPLE_OUT_OF_RANGE = "SAMPLE_OUT_OF_RANGE"
    FEATURE_NOT_FOUND = "FEATURE_NOT_FOUND"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"
```

---

## TD-04: Field descriptions serve dual purpose (dev docs + LLM instructions)

**Source:** Claude Code audit
**Severity:** Low | **Effort:** Moderate | **Area:** Schemas, Testing

`Field(description=...)` strings are read by both developers and the LLM consuming tool outputs. Editing a description for developer clarity could unknowingly change LLM behavior. No test asserts that descriptions contain required instructional content.

**Remediation:** Add snapshot tests or explicit string-match tests for critical Field descriptions. Alternatively, separate LLM-facing instructions from developer docs using a custom metadata field.

---

## TD-05: Pydantic validates producers but not consumers

**Source:** Claude Code audit
**Severity:** Low-Medium | **Effort:** Moderate | **Area:** Schemas, Type Safety

Pydantic validates when data is constructed (in `explainers.py`, `server.py`) but narrators access `.shap_values`, `.prediction`, etc. without consumption-side validation. A schema rename would produce an `AttributeError` at runtime. No `mypy` or `pyright` runs in CI to catch this statically.

**Remediation:** Two-pronged:
1. Add `mypy` or `pyright` to CI (see TD-09)
2. The existing narrator tests implicitly test consumption paths, but adding explicit property-access tests for schema consumers would make this defense more explicit

---

## TD-06: Timestamp non-determinism in ToolMetadata

**Source:** Claude Code audit
**Severity:** Low | **Effort:** Quick | **Area:** Schemas, Testing

`ToolMetadata.timestamp` uses `default_factory=lambda: datetime.now(...)`. This is the only field that varies between calls. Snapshot-testing a full `ToolResponse` would break on timestamp. No `exclude` mechanism or timestamp-stripped comparison helper exists.

**Remediation:** Add a `model_config` with a custom serializer that can strip timestamp for comparison, or a test utility:

```python
def strip_timestamp(response: dict) -> dict:
    """Remove volatile fields for deterministic comparison."""
    r = response.copy()
    if "metadata" in r:
        r["metadata"] = {k: v for k, v in r["metadata"].items() if k != "timestamp"}
    return r
```

---

## TD-07: Glass Floor — template rigidity with no graceful degradation

**Source:** Claude Code audit (ADR-002 acknowledged)
**Severity:** Medium | **Effort:** Significant | **Area:** Narrators

Narratives use rigid f-string templates. If a user asks a nuanced follow-up the templates don't cover, the system goes silent and the LLM must disclaim with the epistemic warning. There is no intermediate path between "fully deterministic" and "fully disclaimed."

**Remediation:** Consider a "template coverage" registry that explicitly declares what question patterns each narrator handles. When a question falls outside coverage, return a structured "partial coverage" response that tells the LLM which parts are grounded and which require disclaimer. This is an architectural decision worth a new ADR.

**Trade-off:** Adding flexibility risks undermining the determinism guarantee that is the core value proposition.

---

## TD-08: Glass Floor — Layer 2 non-determinism and TF-IDF quality

**Source:** Claude Code audit (ADR-009 acknowledged)
**Severity:** Medium | **Effort:** Moderate | **Area:** Knowledge/RAG

Two related issues:
1. Layer 2 is non-deterministic — two identical queries could produce different LLM synthesis of retrieved chunks. The provenance label helps, but there's no verification that the LLM faithfully represented the chunks.
2. TF-IDF retrieval quality is a known PoC limitation. `min_score` defaults to `0.0`, meaning any chunk with any keyword overlap is returned regardless of relevance quality. Poor retrieval silently degrades Layer 2 quality.

**Remediation:**
- Set a meaningful `min_score` threshold (empirically determined from the knowledge corpus) rather than returning everything
- Add a "retrieval confidence" field to `KnowledgeSearchResult` that flags when the best match score is below a warning threshold
- For production: swap TF-IDF internals to sentence-transformer embeddings (the interface via `search_chunks()` doesn't change — ADR-001 separation pays off here)

---

## TD-09: No CI/CD pipeline

**Source:** Codebase review
**Severity:** High | **Effort:** Moderate | **Area:** Infrastructure

The `.github/` directory contains only `copilot-instructions.md` — no GitHub Actions workflows. Tests run only locally via `uv run python -m pytest`. This means:
- No automated regression gate on PRs
- No lint enforcement (ruff is installed but has no config and no CI step)
- No type checking in CI
- No snapshot drift detection on merge

**Remediation:** Create `.github/workflows/ci.yml`:
```yaml
# Minimum viable CI
- pytest with coverage threshold
- ruff check + ruff format --check
- mypy --strict (or pyright) on src/
- syrupy snapshot validation (fail if snapshots are stale)
```

This is the single highest-leverage item for code quality going forward.

---

## TD-10: No static type checking (mypy/pyright)

**Source:** Codebase review
**Severity:** Medium | **Effort:** Moderate | **Area:** Type Safety

Neither `mypy` nor `pyright` appears in dependencies or config. `RegisteredModel.model` is typed as `object` rather than a `Protocol` or typed interface. The codebase uses type hints extensively but they're never machine-verified.

**Remediation:**
1. Add `mypy` to dev dependencies
2. Add `[tool.mypy]` section to `pyproject.toml` with `strict = true` (or start with `--check-untyped-defs`)
3. Define a `ModelProtocol` for the model interface (must have `predict_proba`, etc.)
4. Run in CI (see TD-09)

---

## TD-11: Split/inconsistent dev dependencies

**Source:** Codebase review
**Severity:** Low | **Effort:** Quick | **Area:** Configuration

Dev dependencies are split across two locations in `pyproject.toml`:
- `[project.optional-dependencies] dev = ["pytest", "pytest-cov"]`
- `[dependency-groups] dev = ["ruff>=0.15.2", "syrupy>=5.1.0"]`

This means `pip install -e ".[dev]"` gets pytest but not ruff/syrupy, while `uv sync --group dev` gets ruff/syrupy but not pytest-cov. New contributors will hit confusing missing-dependency errors.

**Remediation:** Consolidate all dev dependencies into `[dependency-groups] dev` (the modern uv approach) or into `[project.optional-dependencies] dev` (the pip-compatible approach). Pick one, document the install command in README.

---

## TD-12: Module-level side effects in server.py

**Source:** Codebase review
**Severity:** Medium | **Effort:** Moderate | **Area:** Architecture

`server.py` loads models and the knowledge base at import time (module-level code). This means:
- Importing the module for testing triggers model loading
- A missing model file causes an import-time print (not a proper error)
- The knowledge base is a module-level global (`_knowledge_base`)

**Remediation:** Wrap initialization in a factory function or lazy initialization pattern. The MCP server startup can call the factory; tests can call it with test fixtures. This also enables "bring your own model/data" more cleanly.

---

## TD-13: Repeated error-handling boilerplate in server.py

**Source:** Codebase review
**Severity:** Low | **Effort:** Quick | **Area:** Code Quality

Every tool function repeats the same `try: registry.get(model_id) / except KeyError` pattern with identical error construction. This is ~8 lines duplicated across 7 tools.

**Remediation:** Extract a decorator or helper:

```python
def _require_model(model_id: str) -> RegisteredModel:
    """Get model or raise structured error. Use in every tool."""
    try:
        return registry.get(model_id)
    except KeyError:
        available = [m["model_id"] for m in registry.list_models()]
        raise ToolError(
            _build_error("MODEL_NOT_FOUND", f"Model '{model_id}' not registered.", available)
        )
```

---

## ~~TD-14: On-the-fly SHAP uses test set as background data~~ → RESOLVED

See Resolved Items below.

---

## TD-15: Snapshot test brittleness

**Source:** Claude Code audit (by design)
**Severity:** Low | **Effort:** Low | **Area:** Testing

Any intentional narrative template change requires `--snapshot-update` + manual git review. This is a feature (audit trail) but creates friction for iterative narrative quality improvement.

**Remediation:** This is more of a workflow note than a bug. Document the snapshot update process in CONTRIBUTING.md. Consider grouping snapshot tests so that narrative improvements can be reviewed as a coherent changeset.

---

## TD-16: No ruff configuration

**Source:** Codebase review
**Severity:** Low | **Effort:** Quick | **Area:** Code Quality

Ruff is in dev dependencies but `pyproject.toml` has no `[tool.ruff]` section. Without explicit config, ruff uses defaults which may not match project conventions.

**Remediation:** Add explicit ruff config:
```toml
[tool.ruff]
target-version = "py311"
line-length = 99

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
```

---

## TD-17: No logging configuration outside knowledge.py

**Source:** Codebase review
**Severity:** Low | **Effort:** Quick | **Area:** Observability

Only `knowledge.py` uses Python's `logging` module. `server.py`, `explainers.py`, and `narrators.py` have no logging. Errors in model loading are caught with bare `print()` statements. In production, structured logging is essential for debugging MCP tool failures.

**Remediation:** Add `logger = logging.getLogger(__name__)` to all modules. Replace print statements with `logger.warning()`. Configure a log format in the server entrypoint.

---

## Prioritized Remediation Plan

### Phase 1: Quick wins (1-2 hours)
Items that improve safety and quality with minimal effort:
- **TD-02** — Add semantic validators (`Field(ge=, le=, pattern=)`)
- **TD-03** — Add `Literal` types for constrained strings
- **TD-11** — Consolidate dev dependencies
- **TD-16** — Add ruff configuration
- **TD-06** — Add timestamp-stripping test utility

### Phase 2: CI/CD foundation (half day)
The highest-leverage single investment:
- **TD-09** — Create GitHub Actions CI workflow
- **TD-10** — Add mypy with initial config (can start permissive, tighten over time)
- **TD-17** — Add logging to all modules

### Phase 3: Architectural improvements (1-2 days)
Items that improve maintainability for the scaling phase:
- **TD-12** — Refactor server.py initialization to factory pattern
- **TD-13** — Extract model-lookup boilerplate into decorator/helper
- **TD-08** — Add retrieval confidence threshold to RAG module
- ~~**TD-14**~~ — ✅ Resolved (2026-03-08)

### Phase 4: Strategic decisions (requires ADR)
Items that need architectural discussion before implementation:
- **TD-01** — Typed evidence (discriminated union vs current dict)
- **TD-07** — Template coverage and graceful degradation strategy
- **TD-04** — Separating LLM instructions from developer docs
- **TD-05** — Static type checking coverage for schema consumers

---

## Items Explicitly NOT Technical Debt

These are documented, conscious design choices:
- **Deterministic narratives over LLM-generated ones** (ADR-002) — core value proposition
- **TF-IDF over embeddings** (ADR-009) — documented PoC trade-off with clear upgrade path
- **stdio transport** (ADR-003) — correct for PoC; production switches to streamable-http
- **Two models only** — intentional scope for sprint; ModelRegistry supports arbitrary models
- **argparse over click/typer in CLI** — zero-dependency choice, appropriate for the adapter

---

## Resolved Items

| ID | Item | Resolution |
|---|---|---|
| TD-02 | No semantic validators | Added `Field(ge=, le=, pattern=)` bounds to probability, accuracy, importance, data_hash, relevance_score |
| TD-03 | No Literal/Enum for constrained strings | Added `Literal` types for direction, source, provenance_label, retrieval_method; `ErrorCode(StrEnum)` for error codes |
| TD-06 | Timestamp non-determinism | Created `_testing.py` with `strip_volatile_fields()` utility |
| TD-10 | No type checking / `model: object` | Added `ClassifierProtocol(Protocol)` with `predict` + `predict_proba`; added `py.typed` marker; added `[tool.mypy]` config |
| TD-11 | Split dev dependencies | Consolidated all dev deps into `[dependency-groups] dev` |
| TD-13 | Repeated error-handling boilerplate | Extracted `_require_model()` helper; all 7 tools now use it |
| TD-16 | No ruff configuration | Added `[tool.ruff]` with select rules, isort, line-length |
| TD-14 | On-the-fly SHAP uses test set as background | Registry loads `X_train` when available; all SHAP callers pass `background_data=entry.X_train`; training scripts save `X_train` to disk. Adapted from Tamas's `explainability_node()` methodology (ADR-010). |
| TD-17 | No logging outside knowledge.py | Added `logger = logging.getLogger(__name__)` to all modules; replaced `print()` with `logger.warning()` |
