"""Kedro pipeline factory — assembles XAI nodes into a Pipeline.

Usage::

    from xai_toolkit.kedro_adapter import create_xai_pipeline

    pipeline = create_xai_pipeline()

The pipeline expects these catalog entries:
  - xai_model: fitted classifier
  - xai_X_test: test feature DataFrame
  - xai_y_test: test target Series
  - xai_X_train: training feature DataFrame
  - xai_X_scoring: scoring/production DataFrame (for drift)
  - xai_model_metadata: dict with feature_names, target_names, etc.

And these parameters (in parameters/xai.yml):
  - xai.model_id: str
  - xai.run_id: str (optional, auto-generated if absent)
  - xai.n_samples: int (default 100)
  - xai.strategy: "random" | "uncertainty" (default "random")
  - xai.random_state: int (default 42)
  - xai.target_names: list[str] (default ["class_0", "class_1"])
"""

try:
    from kedro.pipeline import Pipeline, node

    def create_xai_pipeline() -> Pipeline:
        """Create the XAI batch pipeline.

        Nodes:
          1. sample_indices — select which rows to explain
          2. batch_explain — SHAP + narration for sampled rows
          3. detect_drift — drift detection + narration
          4. model_summary — global feature importance
        """
        from xai_toolkit.kedro_adapter.nodes import (
            batch_explain_node,
            detect_drift_node,
            model_summary_node,
            sample_indices_node,
        )

        return Pipeline(
            [
                node(
                    func=sample_indices_node,
                    inputs=["xai_model", "xai_X_test", "params:xai"],
                    outputs="xai_sample_indices",
                    name="xai_sample_indices",
                ),
                node(
                    func=batch_explain_node,
                    inputs=[
                        "xai_model",
                        "xai_X_test",
                        "xai_y_test",
                        "xai_X_train",
                        "xai_sample_indices",
                        "params:xai",
                    ],
                    outputs="xai_explanations",
                    name="xai_batch_explain",
                ),
                node(
                    func=detect_drift_node,
                    inputs=["xai_X_train", "xai_X_scoring", "params:xai"],
                    outputs="xai_drift_results",
                    name="xai_detect_drift",
                ),
                node(
                    func=model_summary_node,
                    inputs=[
                        "xai_model",
                        "xai_X_test",
                        "xai_model_metadata",
                        "params:xai",
                    ],
                    outputs="xai_model_summary",
                    name="xai_model_summary",
                ),
            ]
        )

except ImportError:
    def create_xai_pipeline():
        """Kedro is not installed — raise a helpful error."""
        raise ImportError(
            "Kedro is required for the XAI pipeline adapter. "
            "Install with: pip install 'xai-toolkit[kedro]'"
        )
