"""Model registry — load, manage, and list available models (ADR-006).

Design pattern: Registry — a central lookup that decouples model loading
from model usage. Consumers ask for a model by ID; the registry handles
the how/where of loading.

At load time, each model is introspected using detect_model_type()
(adapted from the Kedro explainability pipeline) to produce a
pipeline-compatible type string stored as 'detected_type' in metadata.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import joblib
import numpy as np
import pandas as pd

from xai_toolkit.pipeline_compat import detect_model_type

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model interface contract
# ---------------------------------------------------------------------------


@runtime_checkable
class ClassifierProtocol(Protocol):
    """Structural type for fitted classifiers used by the toolkit.

    Design pattern: Protocol (PEP 544) — defines the interface a model must
    satisfy without requiring inheritance. This is Python's equivalent of
    C++ concepts or Go interfaces: structural (duck) typing made explicit.

    Any model that has predict() and predict_proba() satisfies this protocol,
    whether it's sklearn, XGBoost, LightGBM, or a custom wrapper.
    """

    def predict(self, X: pd.DataFrame) -> np.ndarray: ...
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


@dataclass
class RegisteredModel:
    """A model and its associated metadata and data."""

    model_id: str
    model: ClassifierProtocol
    metadata: dict
    X_test: pd.DataFrame
    y_test: pd.Series
    X_train: pd.DataFrame | None = None


class ModelRegistry:
    """Central registry for loaded models.

    Example:
        >>> registry = ModelRegistry()
        >>> registry.load_from_disk("xgboost_breast_cancer", models_dir, data_dir)
        >>> model_entry = registry.get("xgboost_breast_cancer")
        >>> model_entry.model.predict(...)
    """

    def __init__(self) -> None:
        self._models: dict[str, RegisteredModel] = {}

    @staticmethod
    def _validate_supported_model(model: object, model_id: str) -> None:
        """Require classifier-style models that fit the current explainer contract."""
        if not isinstance(model, ClassifierProtocol):
            raise ValueError(
                f"Model '{model_id}' is unsupported. "
                "Only classifiers with both predict() and predict_proba() are supported."
            )
        classes = getattr(model, "classes_", None)
        if classes is not None and len(np.asarray(classes)) != 2:
            raise ValueError(
                f"Model '{model_id}' is unsupported. "
                "Only binary classifiers are supported."
            )

    @staticmethod
    def _validate_loaded_artifacts(
        model_id: str,
        metadata: dict,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        X_train: pd.DataFrame | None,
    ) -> None:
        """Validate the loaded metadata and tabular artifacts agree exactly."""
        feature_names = metadata.get("feature_names")
        if not isinstance(feature_names, list) or not feature_names:
            raise ValueError(
                f"Model '{model_id}' metadata must include a non-empty 'feature_names' list."
            )
        if list(X_test.columns) != feature_names:
            raise ValueError(
                f"Model '{model_id}' test data columns must exactly match metadata "
                "feature_names in the same order."
            )
        if X_train is not None and list(X_train.columns) != feature_names:
            raise ValueError(
                f"Model '{model_id}' training data columns must exactly match metadata "
                "feature_names in the same order."
            )
        if len(y_test) != len(X_test):
            raise ValueError(
                f"Model '{model_id}' test target length must match test feature rows."
            )
        target_names = metadata.get("target_names")
        if target_names is not None and len(target_names) != 2:
            raise ValueError(
                f"Model '{model_id}' is unsupported. "
                "Only binary classification metadata with exactly 2 target_names is supported."
            )

    def load_from_disk(
        self,
        model_id: str,
        models_dir: Path,
        data_dir: Path,
    ) -> None:
        """Load a model, its metadata, and test data from disk.

        Expects:
            models_dir / {model_id}.joblib
            models_dir / {model_id}_meta.json
            data_dir / {dataset_name}_test_X.csv
            data_dir / {dataset_name}_test_y.csv

        Args:
            model_id: Unique identifier (e.g., "xgboost_breast_cancer").
            models_dir: Path to directory containing .joblib and _meta.json.
            data_dir: Path to directory containing test CSV files.
        """
        # Load model
        model_path = models_dir / f"{model_id}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = joblib.load(model_path)
        self._validate_supported_model(model, model_id)

        # Load metadata
        meta_path = models_dir / f"{model_id}_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        with open(meta_path) as f:
            metadata = json.load(f)

        # Load test data
        dataset_name = metadata.get("dataset_name", model_id)
        X_test = pd.read_csv(data_dir / f"{dataset_name}_test_X.csv")
        y_test = pd.read_csv(data_dir / f"{dataset_name}_test_y.csv").squeeze()

        # Load training data if available (TD-14: used as SHAP background).
        # Adapted from Tamas's explainability_node() which correctly uses
        # X_train for background distribution and X_test for samples to explain.
        X_train = None
        train_path = data_dir / f"{dataset_name}_train_X.csv"
        if train_path.exists():
            X_train = pd.read_csv(train_path)
            logger.info("Loaded training data for '%s' (%d samples)", model_id, len(X_train))
        else:
            logger.debug("No training data found at %s; SHAP will use test data as background.", train_path)

        self._validate_loaded_artifacts(
            model_id=model_id,
            metadata=metadata,
            X_test=X_test,
            y_test=y_test,
            X_train=X_train,
        )

        # Detect model type using pipeline-compatible logic (Tamas's
        # _detect_model_type from the Kedro explainability pipeline).
        # This ensures our metadata matches what the pipeline would produce.
        detected_type = detect_model_type(model)
        metadata["detected_type"] = detected_type

        self._models[model_id] = RegisteredModel(
            model_id=model_id,
            model=model,
            metadata=metadata,
            X_test=X_test,
            y_test=y_test,
            X_train=X_train,
        )
        logger.info(
            "Registered model '%s' (type=%s, detected=%s, features=%d, test_samples=%d, train_samples=%s)",
            model_id,
            metadata.get("model_type", "unknown"),
            detected_type,
            len(X_test.columns),
            len(X_test),
            len(X_train) if X_train is not None else "N/A",
        )

    def get(self, model_id: str) -> RegisteredModel:
        """Retrieve a registered model by ID.

        Raises:
            KeyError: If model_id is not registered, with a message
                listing available models.
        """
        if model_id not in self._models:
            available = list(self._models.keys())
            raise KeyError(
                f"Model '{model_id}' not found. "
                f"Available models: {available}"
            )
        return self._models[model_id]

    def list_models(self) -> list[dict]:
        """List all registered models with summary metadata."""
        return [
            {
                "model_id": entry.model_id,
                "model_type": entry.metadata.get("model_type", "unknown"),
                "detected_type": entry.metadata.get("detected_type", "unknown"),
                "dataset_name": entry.metadata.get("dataset_name", "unknown"),
                "feature_count": len(entry.metadata.get("feature_names", [])),
                "accuracy": entry.metadata.get("accuracy"),
            }
            for entry in self._models.values()
        ]
