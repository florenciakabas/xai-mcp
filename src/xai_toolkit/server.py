"""MCP server — thin adapter layer (ADR-001).

This is the ONLY file that imports MCP/FastMCP. It wires together:
  - registry.py (model loading)
  - explainers.py (SHAP computation)
  - narrators.py (English generation)
  - schemas.py (data contracts)

The server itself does NO computation or narrative generation.
It is an adapter: receive request → call pure functions → return response.
"""

from datetime import datetime, timezone
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from xai_toolkit.explainers import compute_shap_values
from xai_toolkit.narrators import narrate_prediction
from xai_toolkit.registry import ModelRegistry
from xai_toolkit.schemas import ExplainPredictionResponse, ToolMetadata

# --- Server setup ---

mcp = FastMCP(
    "xai-toolkit",
    instructions=(
        "ML model explainability server. Provides plain-English explanations "
        "of model predictions backed by SHAP analysis. Call tools to get "
        "deterministic, reproducible explanations — do not interpret SHAP "
        "values yourself."
    ),
)

# --- Model registry initialization ---

ROOT = Path(__file__).parent.parent.parent  # xai-toolkit/
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

registry = ModelRegistry()

# Load available models at startup
try:
    registry.load_from_disk("xgboost_breast_cancer", MODELS_DIR, DATA_DIR)
except FileNotFoundError as e:
    print(f"Warning: Could not load model at startup: {e}")


# --- MCP Tools ---


@mcp.tool()
def explain_prediction(model_id: str, sample_index: int) -> dict:
    """Explain why a single sample received its classification.

    Returns a plain-English narrative explaining which features drove
    the model's prediction for the given sample, backed by SHAP values.

    Args:
        model_id: ID of a registered model (e.g., "xgboost_breast_cancer").
        sample_index: Row index in the test dataset to explain (0-based).

    Returns:
        A dict with 'narrative' (English explanation), 'evidence' (SHAP data),
        and 'metadata' (audit trail).
    """
    # Get model from registry
    try:
        entry = registry.get(model_id)
    except KeyError as e:
        return {"error": str(e)}

    # Validate sample index
    try:
        shap_result = compute_shap_values(
            model=entry.model,
            X=entry.X_test,
            sample_index=sample_index,
            target_names=entry.metadata.get("target_names"),
        )
    except IndexError as e:
        return {"error": str(e)}

    # Generate narrative
    narrative = narrate_prediction(shap_result, top_n=3)

    # Build response
    response = ExplainPredictionResponse(
        narrative=narrative,
        evidence=shap_result.model_dump(),
        metadata=ToolMetadata(
            model_id=model_id,
            model_type=entry.metadata.get("model_type", "unknown"),
            sample_index=sample_index,
            dataset_size=len(entry.X_test),
        ),
    )

    return response.model_dump()


# --- Entrypoint ---

if __name__ == "__main__":
    mcp.run(transport="stdio")
