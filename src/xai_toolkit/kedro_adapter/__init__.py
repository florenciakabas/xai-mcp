"""Kedro adapter — thin Layer 4 adapter for batch XAI pipeline.

Exports `create_xai_pipeline` for integration into a Kedro project.
All computation is delegated to pure functions in Layer 1–2.

Usage in a partner Kedro project::

    # src/partner_project/pipeline_registry.py
    from xai_toolkit.kedro_adapter import create_xai_pipeline

    def register_pipelines():
        return {
            "xai": create_xai_pipeline(),
            ...
        }
"""

from xai_toolkit.kedro_adapter.pipeline import create_xai_pipeline

__all__ = ["create_xai_pipeline"]
