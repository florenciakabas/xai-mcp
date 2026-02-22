"""Pydantic schemas — single source of truth for tool contracts (ADR-004).

These models define the shape of data flowing between layers:
  explainers.py → narrators.py → server.py

No MCP imports here. These are pure data contracts.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


# --- Explainer output (what SHAP computation produces) ---


class ShapResult(BaseModel):
    """Output from SHAP computation for a single sample."""

    prediction: int = Field(description="Predicted class (0 or 1)")
    prediction_label: str = Field(description="Human-readable class name")
    probability: float = Field(description="Predicted probability for the positive class")
    base_value: float = Field(description="SHAP base value (average model output)")
    shap_values: dict[str, float] = Field(
        description="Feature name → SHAP value mapping"
    )
    feature_values: dict[str, float] = Field(
        description="Feature name → actual value for this sample"
    )
    feature_names: list[str] = Field(description="All feature names in order")


# --- Tool response (what the MCP tool returns) ---


class ToolMetadata(BaseModel):
    """Audit metadata attached to every tool response."""

    model_id: str
    model_type: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    tool_version: str = "0.1.0"
    sample_index: int | None = None
    dataset_size: int | None = None


class ExplainPredictionResponse(BaseModel):
    """Full response from the explain_prediction tool."""

    narrative: str = Field(description="Plain English explanation")
    evidence: dict = Field(description="Structured SHAP data backing the narrative")
    metadata: ToolMetadata
    plot_base64: str | None = Field(
        default=None, description="Optional base64-encoded PNG"
    )
