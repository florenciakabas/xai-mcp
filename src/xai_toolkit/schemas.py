"""Pydantic schemas — single source of truth for tool contracts (ADR-004).

These models define the shape of data flowing between layers:
  explainers.py → narrators.py → server.py

No MCP imports here. These are pure data contracts.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


# --- Explainer outputs (what computation produces) ---


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


class FeatureImportance(BaseModel):
    """A single feature's global importance."""

    name: str
    importance: float = Field(description="Mean absolute SHAP value")
    direction: str = Field(description="'positive' or 'negative' average effect")
    mean_shap: float = Field(description="Mean SHAP value (signed)")


class ModelSummary(BaseModel):
    """Summary statistics about a model."""

    model_type: str
    accuracy: float
    n_features: int
    n_train_samples: int
    n_test_samples: int
    target_names: list[str]
    top_features: list[FeatureImportance]


class PartialDependenceResult(BaseModel):
    """Output from partial dependence + ICE computation."""

    feature_name: str
    feature_values: list[float] = Field(description="Grid of feature values")
    predictions: list[float] = Field(description="Mean prediction at each grid point (PDP)")
    ice_curves: list[list[float]] = Field(
        default_factory=list,
        description="Per-sample prediction curves (ICE). Each inner list is one sample's predictions across the grid.",
    )
    feature_min: float
    feature_max: float
    prediction_min: float
    prediction_max: float


class DatasetDescription(BaseModel):
    """Summary statistics about a dataset."""

    n_samples: int
    n_features: int
    feature_names: list[str]
    class_distribution: dict[str, int] = Field(
        description="Class label → count mapping"
    )
    missing_values: int = Field(description="Total missing values across all features")
    feature_stats: dict[str, dict[str, float]] = Field(
        description="Feature name → {mean, std, min, max}"
    )


# --- Tool response (what MCP tools return) — ADR-005 ---


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


class ToolResponse(BaseModel):
    """Generic response for all tools (ADR-005).

    Every tool returns this same shape: narrative + evidence + metadata.
    This consistency means the LLM always knows what to expect.
    """

    narrative: str = Field(description="Plain English interpretation")
    evidence: dict = Field(description="Structured data backing the narrative")
    metadata: ToolMetadata
    plot_base64: str | None = Field(
        default=None, description="Optional base64-encoded PNG"
    )
