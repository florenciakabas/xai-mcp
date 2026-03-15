"""Pydantic schemas — single source of truth for tool contracts (ADR-004).

These models define the shape of data flowing between layers:
  explainers.py → narrators.py → server.py

No MCP imports here. These are pure data contracts.

Type discipline:
  - Literal types constrain string fields to documented valid values.
  - Field validators enforce semantic bounds (probabilities ∈ [0,1], hashes = 64 hex).
  - StrEnum for error codes ensures machine-readability without silent typos.
"""

from datetime import datetime, timezone
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


# --- Explainer outputs (what computation produces) ---


class ShapResult(BaseModel):
    """Output from SHAP computation for a single sample."""

    prediction: int = Field(ge=0, description="Predicted class (0 or 1)")
    prediction_label: str = Field(min_length=1, description="Human-readable class name")
    probability: float = Field(
        ge=0.0, le=1.0,
        description="Predicted probability for the positive class",
    )
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
    importance: float = Field(ge=0.0, description="Mean absolute SHAP value")
    direction: Literal["positive", "negative"] = Field(
        description="Average effect direction on the positive class"
    )
    mean_shap: float = Field(description="Mean SHAP value (signed)")


class ModelSummary(BaseModel):
    """Summary statistics about a model."""

    model_type: str
    accuracy: float = Field(ge=0.0, le=1.0)
    n_features: int = Field(ge=1)
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
        pattern=r"^[a-f0-9]{64}$",
        description=(
            "SHA256 hex digest of the input data used to generate this explanation. "
            "Enables audit trail: same hash guarantees same underlying data. "
            "64 hex characters."
        ),
    )
    source: Literal["on_the_fly", "pipeline", "precomputed"] = Field(
        default="on_the_fly",
        description=(
            "How this explanation was generated. "
            "'on_the_fly': computed in real-time from model + data (PoC). "
            "'pipeline': read from pre-computed Kedro pipeline artifacts (production). "
            "'precomputed': retrieved from the result store (batch pipeline output)."
        ),
    )
    batch_run_id: str | None = Field(
        default=None,
        description=(
            "Identifier of the batch pipeline run that produced this result. "
            "Populated when source='precomputed'; None for on-the-fly."
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
    run_id: str | None = Field(
        default=None,
        description=(
            "Resolved run identifier used for retrieval-first responses. "
            "Typically equals batch_run_id when source='precomputed'."
        ),
    )
    resolved_run_strategy: str = Field(
        default="not_applicable",
        description=(
            "How run_id was resolved. "
            "Examples: 'latest_successful_by_computed_at', "
            "'explicit_run_id', 'not_applicable'."
        ),
    )
    provenance: Literal["precomputed", "on_the_fly", "pipeline", "unknown"] = Field(
        default="on_the_fly",
        description=(
            "User-facing provenance label for trust/audit. "
            "Mirrors the computation path."
        ),
    )
    data_source: str = Field(
        default="model_registry",
        description=(
            "Primary source backing the answer. "
            "Examples: 'model_registry', 'result_store', 'pipeline_artifacts'."
        ),
    )
    applied_skills: list[dict[str, str]] = Field(
        default_factory=list,
        description=(
            "Skills applied while producing this response. "
            "Each entry contains {'id': <skill_id>, 'version': <version>}."
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


class ErrorCode(StrEnum):
    """Machine-readable error codes for tool failures.

    Using StrEnum (Python 3.11+) ensures that error codes are:
      - Constrained to a known set at construction time.
      - Serialized as plain strings in JSON (StrEnum inherits from str).
      - Comparable with == against string literals for backward compatibility.
    """

    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    SAMPLE_OUT_OF_RANGE = "SAMPLE_OUT_OF_RANGE"
    FEATURE_NOT_FOUND = "FEATURE_NOT_FOUND"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


class ErrorResponse(BaseModel):
    """Structured error returned when a tool call fails (D3-S3, S4, S5).

    Follows the same structural pattern as ToolResponse so the LLM
    always receives a predictable shape regardless of success or failure.
    The 'available' field tells the user what valid options exist,
    and 'suggestion' provides a closest-match hint for typos.

    Error codes are SCREAMING_SNAKE_CASE for machine-readability.
    Messages are plain English for human-readability.
    """

    error_code: ErrorCode = Field(
        description="Machine-readable error type from the ErrorCode enum."
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


# --- Prediction comparison schemas ---


class SingleModelPrediction(BaseModel):
    """One model's prediction for a single sample."""

    model_id: str
    model_type: str
    predicted_class: int = Field(ge=0, description="Predicted class (0 or 1)")
    predicted_label: str = Field(min_length=1, description="Human-readable class name")
    probability: float = Field(
        ge=0.0, le=1.0, description="Probability for positive class"
    )
    top_features: list[dict] = Field(
        description=(
            "Top N SHAP contributors. Each dict has 'name', 'shap_value', "
            "'feature_value', 'direction'."
        )
    )


class PredictionComparison(BaseModel):
    """Result of comparing predictions from two models on the same sample."""

    per_model: list[SingleModelPrediction] = Field(
        description="Per-model prediction details"
    )
    agreement: bool = Field(
        description="True if both models predict the same class"
    )
    confidence_gap: float = Field(
        ge=0.0, le=1.0,
        description="Absolute difference in predicted probabilities",
    )
    shared_top_features: list[str] = Field(
        description="Feature names appearing in both models' top contributors"
    )
    divergent_features: dict[str, list[str]] = Field(
        description=(
            "Features unique to each model's top contributors. "
            "Keys are model_ids, values are lists of feature names."
        )
    )


# --- Drift detection schemas ---


class DistributionSummary(BaseModel):
    """Summary statistics of a feature's distribution.

    Captures enough about a distribution to narrate meaningfully
    without serializing raw data. Used for both reference and current
    distributions in drift results.
    """

    mean: float
    std: float
    median: float
    min: float
    max: float
    n_samples: int = Field(gt=0)
    quantile_25: float
    quantile_75: float


class FeatureDriftResult(BaseModel):
    """Per-feature drift detection result.

    For numeric features, test_name is "psi" (primary) or "ks" (fallback
    when PSI cannot be computed due to low cardinality). p_value always
    holds the KS test p-value as supporting evidence.

    For categorical features, test_name is "chi_squared" and p_value
    holds the chi-squared p-value.

    Severity thresholds:
      PSI: <0.1 none, 0.1–0.25 moderate, ≥0.25 severe
      KS fallback: p≥0.05 none, 0.01≤p<0.05 moderate, p<0.01 severe
      Chi-squared: p≥0.05 none, 0.01≤p<0.05 moderate, p<0.01 severe
    """

    feature_name: str
    test_name: Literal["ks", "psi", "chi_squared"]
    statistic: float = Field(ge=0.0)
    p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    drift_detected: bool
    severity: Literal["none", "moderate", "severe"]
    reference_summary: DistributionSummary
    current_summary: DistributionSummary


class DatasetDriftResult(BaseModel):
    """Aggregate drift detection across all features in a dataset.

    The origin field records where the drift results came from:
      - "on_the_fly": computed by this toolkit from raw data.
      - "external": pre-computed by an external system (e.g., Evidently,
        Databricks monitoring) and fed into the toolkit for narration.
    """

    features: list[FeatureDriftResult]
    n_features: int = Field(gt=0)
    n_drifted: int = Field(ge=0)
    share_drifted: float = Field(ge=0.0, le=1.0)
    overall_severity: Literal["none", "moderate", "severe"]
    origin: Literal["on_the_fly", "external"] = Field(
        default="on_the_fly",
        description=(
            "How these drift results were produced. "
            "'on_the_fly': computed by the toolkit from raw DataFrames. "
            "'external': pre-computed by an external system and passed in."
        ),
    )


# --- Persisted result-store schemas ---


class StoredExplanation(BaseModel):
    """Persisted row-level explanation from a batch run."""

    run_id: str
    model_id: str
    sample_index: int
    prediction: int
    prediction_label: str
    probability: float
    narrative: str
    top_features: str = Field(description="JSON string of top feature entries.")
    shap_values: str = Field(description="JSON string feature->SHAP mapping.")
    feature_values: str = Field(description="JSON string feature->value mapping.")
    data_hash: str
    computed_at: str


class StoredDriftResult(BaseModel):
    """Persisted per-feature drift result from a batch run."""

    run_id: str
    model_id: str
    feature_name: str
    test_name: str
    statistic: float
    p_value: float | None = None
    drift_detected: bool
    severity: str
    narrative: str
    overall_narrative: str
    overall_severity: str
    computed_at: str


class StoredModelSummary(BaseModel):
    """Persisted global model summary row from a batch run."""

    run_id: str
    model_id: str
    feature_name: str
    importance: float
    rank: int
    narrative: str
    model_type: str
    computed_at: str


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
        ge=0.0,
        description="Retrieval relevance score. Higher is more relevant.",
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
    retrieval_method: Literal["tfidf", "embedding"] = Field(
        default="tfidf",
        description="Retrieval algorithm used (for audit trail)",
    )
    provenance_label: Literal["ai-interpreted", "deterministic"] = Field(
        default="ai-interpreted",
        description=(
            "Epistemic label for the Glass Floor pattern. "
            "'ai-interpreted' signals that any synthesis from these chunks "
            "is LLM-generated, not deterministic."
        ),
    )
