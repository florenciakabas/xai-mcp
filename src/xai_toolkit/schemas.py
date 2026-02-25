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
    """Audit metadata attached to every tool response.

    Every field here answers a question an auditor might ask:
      - model_id          → which model made this prediction?
      - model_type         → what kind of model is it?
      - timestamp          → when was this explanation generated?
      - tool_version       → which version of the toolkit produced this?
      - sample_index       → which sample was explained?
      - dataset_size       → how large was the dataset?
      - data_hash          → was this the exact same data as before?
      - source             → was this computed on-the-fly or from pipeline?
      - detected_type      → how did the pipeline detect the model type?
      - explainer_type     → which SHAP explainer was used?
      - n_rows_explained   → how many samples were explained?

    The pipeline-related fields (detected_type, explainer_type,
    n_rows_explained) are populated when reading from pre-computed Kedro
    pipeline artifacts. They are None for on-the-fly computation.
    This schema is designed to be compatible with the metadata format
    produced by the Kedro explainability pipeline developed by Tamas (xai-xgboost-clf).
    """

    model_id: str
    model_type: str
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    tool_version: str = "0.1.0"
    sample_index: int | None = None
    dataset_size: int | None = None
    data_hash: str | None = Field(
        default=None,
        description=(
            "SHA256 hex digest of the input data used to generate this explanation. "
            "Enables audit trail: same hash guarantees same underlying data. "
            "64 hex characters."
        ),
    )
    source: str = Field(
        default="on_the_fly",
        description=(
            "How this explanation was generated. "
            "'on_the_fly': computed in real-time from model + data (PoC). "
            "'pipeline': read from pre-computed Kedro pipeline artifacts (production)."
        ),
    )
    detected_type: str | None = Field(
        default=None,
        description=(
            "Model type as detected by the pipeline's _detect_model_type(). "
            "E.g., 'xgboost', 'lightgbm', 'tree', 'keras'. "
            "Populated when source='pipeline'; None for on-the-fly."
        ),
    )
    explainer_type: str | None = Field(
        default=None,
        description=(
            "SHAP explainer used by the pipeline. "
            "E.g., 'tree', 'kernel', 'deep', 'auto'. "
            "Populated when source='pipeline'; None for on-the-fly."
        ),
    )
    n_rows_explained: int | None = Field(
        default=None,
        description=(
            "Number of samples the pipeline computed SHAP for. "
            "Populated when source='pipeline'; None for on-the-fly."
        ),
    )


class ToolResponse(BaseModel):
    """Generic response for all tools (ADR-005).

    Every tool returns this same shape: narrative + evidence + metadata.
    This consistency means the LLM always knows what to expect.

    The `grounded` flag is the epistemic label for the consuming LLM:
      - grounded=True  → this answer was computed deterministically from a
                         registered model. It is reproducible and audit-ready.
      - grounded=False → reserved for future use (e.g. a conversational
                         fallback tool). Currently all tool responses are
                         grounded=True by definition: if a tool was called,
                         computation happened.

    The LLM is instructed (via server instructions and copilot-instructions.md)
    to prepend a disclaimer on any response it generates WITHOUT calling a tool.
    That disclaimer is triggered by the *absence* of grounded=True in the
    response — because ungrounded answers don't go through this schema at all.
    """

    narrative: str = Field(description="Plain English interpretation")
    evidence: dict = Field(description="Structured data backing the narrative")
    metadata: ToolMetadata
    plot_base64: str | None = Field(
        default=None, description="Optional base64-encoded PNG"
    )
    grounded: bool = Field(
        default=True,
        description=(
            "True when this response was computed deterministically from a "
            "registered model. Always True for tool responses. Signals to "
            "the LLM that the narrative is audit-ready and should be "
            "presented as a verified result."
        ),
    )


class ErrorResponse(BaseModel):
    """Structured error returned when a tool call fails (D3-S3, S4, S5).

    Follows the same structural pattern as ToolResponse so the LLM
    always receives a predictable shape regardless of success or failure.
    The 'available' field tells the user what valid options exist,
    and 'suggestion' provides a closest-match hint for typos.

    Error codes are SCREAMING_SNAKE_CASE for machine-readability.
    Messages are plain English for human-readability.
    """

    error_code: str = Field(
        description=(
            "Machine-readable error type. One of: "
            "MODEL_NOT_FOUND | SAMPLE_OUT_OF_RANGE | FEATURE_NOT_FOUND | UNKNOWN_ERROR"
        )
    )
    message: str = Field(description="Human-readable explanation of what went wrong")
    available: list[str] = Field(
        default_factory=list,
        description=(
            "Valid options the user can try instead. "
            "For MODEL_NOT_FOUND: list of registered model IDs. "
            "For SAMPLE_OUT_OF_RANGE: valid index range as a string. "
            "For FEATURE_NOT_FOUND: list of all feature names."
        ),
    )
    suggestion: str | None = Field(
        default=None,
        description="Closest-match suggestion (populated for typo-style errors like FEATURE_NOT_FOUND)",
    )


# --- Knowledge / RAG schemas (ADR-009) ---


class KnowledgeChunk(BaseModel):
    """A single chunk of retrieved business context.

    Each chunk traces back to a specific section of a specific document,
    enabling the "Glass Floor" provenance requirement (ADR-009).

    Example:
        >>> chunk = KnowledgeChunk(
        ...     text="Values > 14.0 mm are clinically significant...",
        ...     source_document="clinical_protocol.md",
        ...     document_id="PROTO-2024-BC-001",
        ...     section_heading="mean_radius",
        ...     chunk_index=4,
        ...     relevance_score=0.82,
        ... )
        >>> chunk.source_document
        'clinical_protocol.md'
    """

    text: str = Field(description="The chunk content")
    source_document: str = Field(description="Filename the chunk was extracted from")
    document_id: str = Field(
        default="",
        description="Document ID from metadata (e.g., 'PROTO-2024-BC-001')",
    )
    section_heading: str = Field(
        description="The heading under which this chunk appeared"
    )
    chunk_index: int = Field(description="Position of this chunk within the document")
    relevance_score: float = Field(
        default=0.0,
        description="Retrieval relevance score (0.0–1.0). Higher is more relevant.",
    )


class KnowledgeSearchResult(BaseModel):
    """Result from searching the knowledge base.

    Returned by the retrieve_business_context MCP tool.
    Contains the retrieved chunks plus metadata for audit trail.

    Example:
        >>> result = KnowledgeSearchResult(
        ...     chunks=[chunk],
        ...     query="high risk malignant case",
        ...     documents_searched=["clinical_protocol.md"],
        ...     retrieval_method="tfidf",
        ... )
    """

    chunks: list[KnowledgeChunk] = Field(
        description="Retrieved chunks, ordered by relevance (highest first)"
    )
    query: str = Field(description="The original search query")
    documents_searched: list[str] = Field(
        description="List of document filenames that were searched"
    )
    retrieval_method: str = Field(
        default="tfidf",
        description="Retrieval algorithm used (for audit trail)",
    )
    provenance_label: str = Field(
        default="ai-interpreted",
        description=(
            "Epistemic label for the Glass Floor pattern. "
            "Always 'ai-interpreted' — signals that any synthesis "
            "from these chunks is LLM-generated, not deterministic."
        ),
    )
