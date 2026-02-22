"""Model registry — load, manage, and list available models (ADR-006).

Design pattern: Registry — a central lookup that decouples model loading
from model usage. Consumers ask for a model by ID; the registry handles
the how/where of loading.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd


@dataclass
class RegisteredModel:
    """A model and its associated metadata and data."""

    model_id: str
    model: object  # fitted sklearn/xgboost model
    metadata: dict
    X_test: pd.DataFrame
    y_test: pd.Series


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

        self._models[model_id] = RegisteredModel(
            model_id=model_id,
            model=model,
            metadata=metadata,
            X_test=X_test,
            y_test=y_test,
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
                "dataset_name": entry.metadata.get("dataset_name", "unknown"),
                "feature_count": len(entry.metadata.get("feature_names", [])),
                "accuracy": entry.metadata.get("accuracy"),
            }
            for entry in self._models.values()
        ]
