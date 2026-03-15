# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # XAI Toolkit — Lubricant Quality Demo
# MAGIC
# MAGIC **Audience:** Reliability Engineering / Data Science teams using Kedro + LightGBM + Databricks
# MAGIC
# MAGIC This notebook demonstrates the full xai-toolkit workflow:
# MAGIC 1. Train a LightGBM lubricant quality classifier
# MAGIC 2. Register the model
# MAGIC 3. Generate deterministic, SHAP-backed explanations
# MAGIC 4. Detect seasonal drift
# MAGIC 5. Retrieve domain knowledge (Glass Floor protocol)
# MAGIC
# MAGIC **No MCP server required** — all functions are pure Python.

# COMMAND ----------

# Setup: install xai-toolkit and restart Python
# %pip install /Workspace/path/to/xai_toolkit-0.1.0-py3-none-any.whl lightgbm
# dbutils.library.restartPython()  # noqa: F821

import sys
from pathlib import Path

# Fallback for local development: add src/ to path
_src = Path.cwd().parent / "src" if (Path.cwd().parent / "src").exists() else Path("src")
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import numpy as np
import pandas as pd

print("Setup complete.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Generate Synthetic Lubricant Data
# MAGIC
# MAGIC We simulate oil analysis data with 12 features from ASTM/ISO test methods.
# MAGIC Training data uses winter baselines; test data introduces summer drift
# MAGIC (temperature → viscosity shifts, humidity → water content, equipment wear → particles).

# COMMAND ----------

FEATURE_NAMES = [
    "viscosity_40c", "viscosity_100c", "total_acid_number", "water_content_ppm",
    "flash_point_c", "oxidation_stability", "particle_count_4um", "iron_ppm",
    "copper_ppm", "foam_tendency_ml", "color_astm", "demulsibility_min",
]
TARGET_NAMES = ["in_spec", "degraded"]


def generate_samples(n_samples, rng, summer=False):
    degraded_fraction = 0.30
    n_degraded = int(n_samples * degraded_fraction)
    labels = np.array([0] * (n_samples - n_degraded) + [1] * n_degraded)
    rng.shuffle(labels)

    data = {
        "viscosity_40c": rng.normal(68.0, 3.0, n_samples),
        "viscosity_100c": rng.normal(10.5, 0.8, n_samples),
        "total_acid_number": rng.normal(0.5, 0.1, n_samples),
        "water_content_ppm": rng.normal(80, 20, n_samples),
        "flash_point_c": rng.normal(220, 8, n_samples),
        "oxidation_stability": rng.normal(300, 30, n_samples),
        "particle_count_4um": rng.normal(50, 15, n_samples),
        "iron_ppm": rng.normal(10, 3, n_samples),
        "copper_ppm": rng.normal(2, 0.8, n_samples),
        "foam_tendency_ml": rng.normal(20, 8, n_samples),
        "color_astm": rng.normal(2.0, 0.5, n_samples),
        "demulsibility_min": rng.normal(15, 4, n_samples),
    }

    for i in range(n_samples):
        if labels[i] == 1:
            severity = rng.uniform(0.5, 1.0)
            data["total_acid_number"][i] += severity * rng.uniform(1.0, 3.0)
            data["water_content_ppm"][i] += severity * rng.uniform(100, 400)
            data["flash_point_c"][i] -= severity * rng.uniform(15, 40)
            data["oxidation_stability"][i] -= severity * rng.uniform(80, 180)
            data["particle_count_4um"][i] += severity * rng.uniform(40, 120)
            data["iron_ppm"][i] += severity * rng.uniform(10, 40)
            data["copper_ppm"][i] += severity * rng.uniform(3, 10)
            data["foam_tendency_ml"][i] += severity * rng.uniform(15, 50)
            data["color_astm"][i] += severity * rng.uniform(1.5, 4.0)
            data["demulsibility_min"][i] += severity * rng.uniform(10, 30)
            direction = rng.choice([-1, 1])
            data["viscosity_40c"][i] += direction * severity * rng.uniform(5, 15)
            data["viscosity_100c"][i] += direction * severity * rng.uniform(1, 4)

    if summer:
        data["viscosity_40c"] -= rng.normal(3.0, 1.0, n_samples)
        data["viscosity_100c"] -= rng.normal(0.5, 0.2, n_samples)
        data["water_content_ppm"] += rng.normal(40, 10, n_samples)
        data["particle_count_4um"] += rng.normal(12, 4, n_samples)

    X = pd.DataFrame(data, columns=FEATURE_NAMES)
    y = pd.Series(labels, name="target")
    return X, y


rng = np.random.RandomState(42)
X_train, y_train = generate_samples(500, rng, summer=False)
X_test, y_test = generate_samples(150, rng, summer=True)

print(f"Training: {len(X_train)} samples ({y_train.sum()} degraded)")
print(f"Test:     {len(X_test)} samples ({y_test.sum()} degraded)")
X_train.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Train LightGBM Classifier

# COMMAND ----------

from lightgbm import LGBMClassifier

model = LGBMClassifier(
    n_estimators=100, max_depth=5, learning_rate=0.1,
    random_state=42, verbose=-1,
)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Test accuracy: {accuracy:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Register Model
# MAGIC
# MAGIC `register_in_memory()` wraps the trained model with its metadata so all
# MAGIC xai-toolkit functions work seamlessly. In production, `load_from_disk()`
# MAGIC reads the same artifacts from MLflow or a mounted volume.

# COMMAND ----------

from xai_toolkit.registry import ModelRegistry

registry = ModelRegistry()
registry.register_in_memory(
    model_id="lgbm_lubricant_quality",
    model=model,
    X_test=X_test,
    y_test=y_test,
    X_train=X_train,
    feature_names=FEATURE_NAMES,
    target_names=TARGET_NAMES,
)
print("Registered models:", registry.list_models())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Model Summary + Intrinsic Importances

# COMMAND ----------

from xai_toolkit.explainers import compute_model_summary, extract_intrinsic_importances
from xai_toolkit.narrators import narrate_model_summary, narrate_intrinsic_importance

reg = registry.get("lgbm_lubricant_quality")

summary = compute_model_summary(reg)
print(narrate_model_summary(summary))
print()

intrinsic = extract_intrinsic_importances(reg)
if intrinsic:
    print(narrate_intrinsic_importance(intrinsic, reg.feature_names))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Single Prediction Explanation
# MAGIC
# MAGIC Find a degraded sample, compute SHAP values, and generate a deterministic
# MAGIC English narrative explaining *why* the model flagged it.

# COMMAND ----------

from xai_toolkit.explainers import compute_shap_values
from xai_toolkit.narrators import narrate_prediction
from xai_toolkit.plots import plot_shap_bar

# Find a degraded sample
degraded_idx = int(y_test[y_test == 1].index[0])
shap_result = compute_shap_values(reg, sample_index=degraded_idx)

print(narrate_prediction(shap_result, reg.target_names))
print()

# Display SHAP bar chart
plot_b64 = plot_shap_bar(shap_result, reg.feature_names)
if plot_b64:
    import base64
    from IPython.display import display, HTML
    img_tag = f'<img src="data:image/png;base64,{plot_b64}" width="700"/>'
    display(HTML(img_tag))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Drift Detection
# MAGIC
# MAGIC Compare winter training distribution against summer test data.
# MAGIC Uses KS test, PSI, and chi-squared (auto-selected by dtype).

# COMMAND ----------

from xai_toolkit.drift import detect_drift
from xai_toolkit.narrators import narrate_dataset_drift, narrate_feature_drift

drift_result = detect_drift(X_train, X_test, reg.feature_names)
print(narrate_dataset_drift(drift_result))
print()

# Per-feature narratives for drifted features
for feat in drift_result.features:
    if feat.drifted:
        print(narrate_feature_drift(feat))
        print()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Partial Dependence Plot
# MAGIC
# MAGIC How does `total_acid_number` affect the model's predictions?
# MAGIC PDP shows the average effect; ICE curves show individual variation.

# COMMAND ----------

from xai_toolkit.explainers import compute_partial_dependence
from xai_toolkit.plots import plot_pdp_ice

pdp_result = compute_partial_dependence(reg, feature_name="total_acid_number")

pdp_b64 = plot_pdp_ice(pdp_result, feature_name="total_acid_number")
if pdp_b64:
    import base64
    from IPython.display import display, HTML
    img_tag = f'<img src="data:image/png;base64,{pdp_b64}" width="700"/>'
    display(HTML(img_tag))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Glass Floor Demo: Deterministic Facts + Domain Knowledge
# MAGIC
# MAGIC **Layer 1 (Model Facts):** SHAP-backed narrative — deterministic, reproducible.
# MAGIC
# MAGIC **Layer 2 (Domain Knowledge):** Retrieved from `knowledge/` markdown files via
# MAGIC TF-IDF search. Provenance-labeled as `ai-interpreted` so the LLM (or user)
# MAGIC knows these are contextual guidelines, not model outputs.

# COMMAND ----------

# Layer 1: Deterministic SHAP narrative (already computed above)
print("=" * 60)
print("LAYER 1 — MODEL FACTS (grounded, deterministic)")
print("=" * 60)
print(narrate_prediction(shap_result, reg.target_names))
print()

# Layer 2: Knowledge base retrieval
print("=" * 60)
print("LAYER 2 — DOMAIN KNOWLEDGE (ai-interpreted, retrieved)")
print("=" * 60)

try:
    from xai_toolkit.knowledge import load_knowledge_base, search_chunks
    kb_path = Path("knowledge/")
    if not kb_path.exists():
        kb_path = Path.cwd().parent / "knowledge"
    kb = load_knowledge_base(kb_path)
    results = search_chunks(kb, "high acid number degraded lubricant action", top_k=3)
    for chunk, score in results:
        print(f"\n[score={score:.2f}] {chunk.source_document} > {chunk.section_heading}")
        print(chunk.text[:300])
except Exception as e:
    print(f"Knowledge base not available (expected in some environments): {e}")
    print("In production, knowledge/ directory is mounted or bundled with the wheel.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Integration Paths
# MAGIC
# MAGIC | Surface | How | Key Benefit |
# MAGIC |---|---|---|
# MAGIC | **This Notebook** | Direct Python calls | Full control, exploration, prototyping |
# MAGIC | **MCP Server** | `uv run python -m xai_toolkit.server` | VS Code Copilot / Claude integration |
# MAGIC | **VS Code Copilot** | MCP tool calls in chat | Conversational explainability |
# MAGIC | **Kedro Pipeline** | `kedro_adapter/` nodes | Batch SHAP as pipeline stage |
# MAGIC | **CLI** | `uv run python -m xai_toolkit.cli` | Scripting, CI/CD, automation |
