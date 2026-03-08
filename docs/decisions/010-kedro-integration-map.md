# ADR-010: Kedro Pipeline Integration Map

## Status
Accepted

## Context
The XAI MCP toolkit was built as a natural language access layer for ML model
explainability. A senior principal colleague (Tamas) independently built a
Kedro-based explainability pipeline (`xai-xgboost-clf`) that handles SHAP
computation, intrinsic interpretability, model detection, and visualization.

Management requires that our toolkit visibly integrates and builds upon his
work rather than reimplementing functionality from scratch. This ADR documents
our systematic review of his codebase, what we adopt, what we adapt, and
where we diverge (with rationale).

## Decision

### Function-by-function mapping

| His function (nodes.py) | Signature / Purpose | Our equivalent | Status | Rationale |
|---|---|---|---|---|
| `_detect_model_type(model)` → `str` | Detects xgboost, lightgbm, tree, sklearn, keras, pytorch, pyfunc via type string + duck typing | `pipeline_compat.detect_model_type()` | **Adopted** (already in codebase) | Same detection order, same logic, same lazy imports. Public API with attribution comment citing source repo. |
| `_handle_intrinsically_explainable_model(model, feature_names, save_dir, X_array)` → `(List[str], str)` | Extracts `coef_` or `feature_importances_` from models, saves as `.npy` + metadata JSON | `explainers.extract_intrinsic_importances()` | **Adopted** (Phase 1) | Core logic preserved: check `coef_` first (flatten), then `feature_importances_`. We return `list[FeatureImportance]` in memory instead of saving to disk — per ADR-001. |
| `_choose_explainer_by_type(explainer_type, model, model_type, X_background, pred_callable, gpu_enabled, use_gradient)` → `(explainer, callable)` | Routes to TreeExplainer, DeepExplainer, GradientExplainer, or KernelExplainer based on model type. Detects unsupported Keras activations (gelu, swish, selu). | `explainers.compute_shap_values()` uses `shap.Explainer()` auto-dispatch | **Adapted** | His dispatch is more explicit with named explainer types. Ours uses SHAP's built-in `Explainer()` which auto-detects. Both achieve the same result for our supported model types (tree-based). His Keras/PyTorch/deep learning paths are not needed for our current scope. |
| `_prepare_background(X_array, feature_names, background_size, sample_seed, explainer_type, use_exact_threshold, approx_threshold)` → `DataFrame` | Selects background samples from training data with configurable thresholds, KernelExplainer summarization | `explainers.compute_shap_values()` samples `min(50, len(X))` | **Adapted** (Phase 2 enhances) | His implementation is more sophisticated (tiered thresholds, seed control). Our PoC uses a simpler approach. Phase 2 adopts his correct methodology of using X_train as background. |
| `_prepare_X(X)` → `(feature_names, X_array)` | Normalizes DataFrame/ndarray input | Implicit in our functions (we require DataFrame) | **Diverge** | Our API contract requires `pd.DataFrame` (ADR-004), making this normalization unnecessary. |
| `_call_explainer_on_batch(explainer, batch, pred_callable, background)` → SHAP values | Tries three calling conventions: callable, `.shap_values()`, manual prediction delta | Not needed | **Diverge** | We use `shap.Explainer()` which handles calling conventions internally. His triple-fallback addresses edge cases with older SHAP APIs and exotic model types we don't support. |
| `_compute_shap_batches(explainer, X_for_shap, batch_size, pred_callable, background)` → list of (sv, batch) | Batched SHAP computation for large datasets | Not needed for PoC | **Diverge** | Our PoC explains single samples on-the-fly. His batched approach is for bulk pipeline runs. Our pipeline bridge (ADR-008) reads his batch outputs. |
| `_extract_expected_value_from_creation(explainer)` / `_from_batches(...)` | Multi-step extraction of SHAP baseline from various explainer APIs | `compute_shap_values()` reads `shap_explanation.base_values[0]` | **Diverge** | His extraction handles TF EagerTensors, multi-output, and edge cases from deep learning explainers. Our models return standard numpy arrays. |
| `_normalize_expected_value(expected_value_full, n_samples, n_outputs)` | Normalizes baseline to (n_samples,) or (n_samples, n_outputs) shape | Not needed | **Diverge** | He saves per-sample baselines for plotting. We use a scalar baseline for narrative generation. |
| `_aggregate_shap_values(explanations, feature_names)` → ndarray | Concatenates batched SHAP outputs, handles multi-output | Not needed for PoC | **Diverge** | Same as batching — this is pipeline infrastructure. Our pipeline bridge reads the concatenated output. |
| `_save_shap_outputs(...)` / `_save_expected_value(...)` / `_save_metadata(...)` | Disk persistence of SHAP artifacts | `load_shap_from_pipeline()` reads these outputs | **Complementary** | He writes, we read. This is the ADR-008 pipeline bridge in action. |
| `_log_to_mlflow(...)` / `_log_intrinsic_to_mlflow(...)` | Uploads artifacts to MLflow tracking server | Not used | **Diverge** | We don't depend on MLflow. Our pipeline bridge reads disk artifacts regardless of MLflow status. |
| `explainability_node(X_train, X_test, y_test, model, params)` → dict | Kedro node orchestrator: prepare data, detect model, run SHAP, save artifacts | No direct equivalent | **Diverge** | This is the Kedro node wrapper. We are not a Kedro plugin — we consume his outputs via the pipeline bridge. |
| `_make_predictable_callable(model)` → callable | Creates `lambda x: model.predict(x)` wrapper | `compute_shap_values()` passes `model.predict_proba` directly | **Adapted** | Same goal, different mechanism. We pass `predict_proba` as a callable to `shap.Explainer()`. |

**Utility functions (xai_utils.py):**

| His function | Purpose | Our equivalent | Status |
|---|---|---|---|
| `_normalize_base_values(expected_value, n_outputs, n_samples)` | Coerces baseline to consistent shape | Not needed | **Diverge** — we work with scalar baselines |
| `make_explanation(values, base_values, data_row, feature_names)` → `shap.Explanation` | Constructs Explanation objects for plotting | `plots.py` constructs Explanation inline | **Adapted** — same goal, different location |
| `get_active_run_info(run_id, run_name)` / `make_artifact_filename(...)` | MLflow run resolution and artifact naming | Not used | **Diverge** — no MLflow dependency |

**Plot functions (xai_plots.py):**

| His function | Purpose | Our equivalent | Status |
|---|---|---|---|
| `_generate_plots_from_artifacts(save_dir, plot_types, ...)` | 5 plot types: summary, bar, waterfall, force, decision. Loads from disk. | `plots.py`: 3 plot types (bar, waterfall, PDP+ICE). In-memory. | **Complementary** | His plots are visual artifacts from pipeline runs. Ours are inline base64 PNGs for MCP responses. Different audiences. |
| `_load_shap_artifacts(save_dir)` | Reads .npy + metadata from disk | `load_shap_from_pipeline()` in explainers.py | **Adopted** — same logic, with attribution |
| `_load_intrinsic_artifacts(save_dir)` | Reads intrinsic importances from disk | Not yet — Phase 1 adds in-memory equivalent | **Adopted** (Phase 1) |

**Data science (data_science/nodes.py):**

| His function | Purpose | Our equivalent | Status |
|---|---|---|---|
| `XGBoostModeling.log_feature_importance()` | Extracts `feature_importances_` and logs to MLflow | `extract_intrinsic_importances()` (Phase 1) | **Adopted** — same extraction, different output target |
| `xgb_training_node(...)` | Full XGBoost training with HalvingRandomSearchCV | `scripts/train_toy_model.py` | **Diverge** — our training is a simple script, not a Kedro node |

**Configuration (parameters_xai.yml):**

| His parameter | Our equivalent | Status |
|---|---|---|
| `background_size: 5000` | Hardcoded `min(50, len(X))` | **Phase 2** — will make configurable |
| `sampling_size: 1000` | Not needed (single-sample on-the-fly) | **Diverge** |
| `batch_size: 10000` | Not needed (single-sample) | **Diverge** |
| `sample_seed: 42` | Hardcoded `random_state=42` | Same value, different mechanism |
| `use_exact_threshold: 1000` | Not needed | **Diverge** |
| `approx_threshold: 50000` | Not needed | **Diverge** |
| `explainer_type: "Tree"` | Auto-detected via `shap.Explainer()` | **Adapted** |
| `mlflow_log_artifacts: true` | No MLflow | **Diverge** |

### What we adopt directly
- **Intrinsic explainability path** (Phase 1): His `_handle_intrinsically_explainable_model()` extracts `coef_` and `feature_importances_` directly from models. We adopt this as `extract_intrinsic_importances()` returning in-memory Pydantic schemas.
- **X_train as SHAP background distribution** (Phase 2): His `explainability_node()` correctly uses `X_train` for background and `X_test` for explanation. We adopt this methodology, fixing TD-14.
- **Model type detection** (already adopted): `pipeline_compat.detect_model_type()` is adapted from his `_detect_model_type()` with attribution.

### What we adapt (different implementation, same goal)
- **Explainer dispatch**: His `_choose_explainer_by_type()` explicitly routes to TreeExplainer, DeepExplainer, etc. Our `compute_shap_values()` uses `shap.Explainer()` auto-dispatch. Same result for tree-based models; his approach supports more exotic model types.
- **Background preparation**: His `_prepare_background()` has tiered thresholds and seed control. Our approach is simpler (`min(50, len(X))`). Phase 2 adds X_train support without the full threshold machinery.
- **Narrative generation**: He produces 5 plot types (visual). We produce deterministic English text (ADR-002). These are complementary, not competing — different audiences and consumption patterns.
- **Artifact persistence**: He saves `.npy` files + metadata JSON to disk for pipeline reproducibility. We return Pydantic schemas in memory for real-time MCP responses. Our pipeline bridge reads his disk artifacts.

### Where we diverge (with rationale)
- **Disk I/O vs in-memory**: His pipeline saves `.npy` files and reads them back for plotting; our on-the-fly path computes and returns in-memory Pydantic schemas. Both paths are valid; our pipeline bridge (ADR-008) reads his artifacts when they exist.
- **Error handling style**: His error handling uses bare `print()` statements (acknowledged in his email as "code cleanup pending"); ours uses structured logging per TD-17 and returns `ErrorResponse` schemas.
- **MLflow dependency**: His pipeline deeply integrates with MLflow for artifact tracking and run management. We deliberately avoid this dependency — we read his output artifacts regardless of MLflow status.
- **Kedro dependency**: His pipeline is a Kedro plugin with node/pipeline definitions. We are a standalone MCP server that can consume Kedro pipeline outputs but does not require Kedro to run.
- **Batching/sampling**: His pipeline handles datasets up to 50K+ rows with batching, sampling, and threshold-based strategy selection. Our PoC explains single samples on-the-fly. For production scale, we defer to his pipeline for bulk computation.
- **Visual vs textual narratives**: His output is 5 plot types (summary, bar, waterfall, force, decision). Our output is deterministic English text (ADR-002). These serve different purposes and are complementary.

### Existing integration points
- **`pipeline_compat.py`** — Contains `detect_model_type()` adapted from his `_detect_model_type()` with explicit attribution comment citing source repo, file, and line numbers.
- **`explainers.py` pipeline bridge** — `load_shap_from_pipeline()` and `load_global_importance_from_pipeline()` read his pipeline output artifacts (`.npy`, `shap_metadata.json`), implementing ADR-008.

## Consequences
- Toolkit gains intrinsic explainability (Phase 1), closing the most visible feature gap identified by Tamas.
- SHAP methodology improves with training data as background (Phase 2), resolving TD-14.
- Integration is documented at function-level granularity, auditable for management review.
- Attribution is explicit and traceable — every adopted function cites the source repo, file, and function name.
- The divergence rationale prevents future "why didn't you just use his code?" questions by documenting the architectural reasons (ADR-001 separation of concerns, ADR-002 deterministic narratives, no MLflow/Kedro dependencies).
