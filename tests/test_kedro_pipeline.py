"""Integration test for Kedro pipeline with SequentialRunner + MemoryDataset.

Requires kedro to be installed. Tests are skipped if kedro is not available.
"""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from xai_toolkit.result_store import StoredDriftResult, StoredExplanation, StoredModelSummary

kedro = pytest.importorskip("kedro", reason="kedro not installed")

from kedro.io import DataCatalog, MemoryDataset
from kedro.runner import SequentialRunner

from xai_toolkit.kedro_adapter import create_xai_pipeline


@pytest.fixture
def pipeline_inputs():
    """Create all inputs needed for the XAI pipeline."""
    X, y = make_classification(
        n_samples=100, n_features=5, n_informative=3,
        random_state=42,
    )
    feature_names = [f"feat_{i}" for i in range(5)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_df, y)

    metadata = {
        "model_type": "RandomForestClassifier",
        "feature_names": feature_names,
        "target_names": ["benign", "malignant"],
        "accuracy": 0.95,
        "n_train_samples": 100,
        "n_test_samples": 100,
    }

    params = {
        "model_id": "test_rf",
        "run_id": "integration-test-001",
        "n_samples": 5,
        "strategy": "random",
        "random_state": 42,
        "target_names": ["benign", "malignant"],
    }

    return {
        "model": model,
        "X_df": X_df,
        "y_series": y_series,
        "metadata": metadata,
        "params": params,
    }


@pytest.fixture
def catalog(pipeline_inputs):
    """Build a Kedro DataCatalog with MemoryDatasets."""
    return DataCatalog(
        datasets={
            "xai_model": MemoryDataset(data=pipeline_inputs["model"]),
            "xai_X_test": MemoryDataset(data=pipeline_inputs["X_df"]),
            "xai_y_test": MemoryDataset(data=pipeline_inputs["y_series"]),
            "xai_X_train": MemoryDataset(data=pipeline_inputs["X_df"]),
            "xai_X_scoring": MemoryDataset(data=pipeline_inputs["X_df"]),
            "xai_model_metadata": MemoryDataset(data=pipeline_inputs["metadata"]),
            "params:xai": MemoryDataset(data=pipeline_inputs["params"]),
        }
    )


def _get_output(outputs: dict, key: str):
    """Extract data from runner output (may be MemoryDataset or raw value)."""
    val = outputs[key]
    if hasattr(val, "load"):
        return val.load()
    return val


class TestKedroXAIPipeline:
    """Integration test: full pipeline with SequentialRunner."""

    def test_pipeline_runs_to_completion(self, catalog):
        pipeline = create_xai_pipeline()
        runner = SequentialRunner()
        outputs = runner.run(pipeline, catalog)

        # xai_sample_indices is an intermediate output consumed by batch_explain,
        # so Kedro only returns terminal (free) outputs
        assert "xai_explanations" in outputs
        assert "xai_drift_results" in outputs
        assert "xai_model_summary" in outputs

    def test_explanations_schema(self, catalog):
        pipeline = create_xai_pipeline()
        outputs = SequentialRunner().run(pipeline, catalog)

        df = _get_output(outputs, "xai_explanations")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5  # n_samples=5

        required_cols = set(StoredExplanation.model_fields.keys())
        assert required_cols.issubset(set(df.columns))

        # Validate each row can be deserialized
        for _, row in df.iterrows():
            exp = StoredExplanation(**row.to_dict())
            assert exp.model_id == "test_rf"
            assert exp.run_id == "integration-test-001"
            shap_dict = json.loads(exp.shap_values)
            assert len(shap_dict) == 5

    def test_drift_results_schema(self, catalog):
        pipeline = create_xai_pipeline()
        outputs = SequentialRunner().run(pipeline, catalog)

        df = _get_output(outputs, "xai_drift_results")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5  # one per feature

        required_cols = set(StoredDriftResult.model_fields.keys())
        assert required_cols.issubset(set(df.columns))

        for _, row in df.iterrows():
            dr = StoredDriftResult(**row.to_dict())
            assert dr.model_id == "test_rf"

    def test_model_summary_schema(self, catalog):
        pipeline = create_xai_pipeline()
        outputs = SequentialRunner().run(pipeline, catalog)

        df = _get_output(outputs, "xai_model_summary")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5  # one per feature

        required_cols = set(StoredModelSummary.model_fields.keys())
        assert required_cols.issubset(set(df.columns))

        ranks = df["rank"].tolist()
        assert ranks == list(range(1, 6))

    def test_explanations_count_matches_n_samples(self, catalog):
        """The number of explanations equals the n_samples parameter."""
        pipeline = create_xai_pipeline()
        outputs = SequentialRunner().run(pipeline, catalog)
        df = _get_output(outputs, "xai_explanations")
        assert len(df) == 5  # params.n_samples = 5


class TestPipelineOutputsAreStoreCompatible:
    """Verify pipeline outputs can be saved/loaded via result_store."""

    def test_explanations_round_trip(self, catalog, tmp_path):
        from xai_toolkit.result_store import save_explanations, load_explanations

        outputs = SequentialRunner().run(create_xai_pipeline(), catalog)
        df = _get_output(outputs, "xai_explanations")

        # Convert DataFrame rows to StoredExplanation and save
        explanations = [StoredExplanation(**row.to_dict()) for _, row in df.iterrows()]
        save_explanations(explanations, tmp_path)

        loaded = load_explanations("test_rf", tmp_path)
        assert len(loaded) == 5
        assert loaded[0].model_id == "test_rf"
