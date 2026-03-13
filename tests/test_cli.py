"""Tests for the CLI adapter (cli.py).

Validates that the CLI produces identical output shapes to the MCP server,
confirming ADR-001's separation of concerns: same pure functions, different adapter.

Strategy:
    - Test argument parsing in isolation (no models needed).
    - Test command handlers with the same fixtures as MCP tool tests.
    - Verify JSON output shape matches ToolResponse / ErrorResponse / KnowledgeSearchResult.
"""

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

import pytest

from xai_toolkit.cli import (
    build_parser,
    cmd_context,
    cmd_explain,
    cmd_feature_drift,
    cmd_models,
    cmd_pdp,
    main,
)
from xai_toolkit.knowledge import load_knowledge_base
from xai_toolkit.registry import ModelRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def parser() -> object:
    """A fresh argument parser."""
    return build_parser()


@pytest.fixture()
def knowledge_dir(tmp_path: Path) -> Path:
    """Temporary knowledge directory with one document."""
    kb_dir = tmp_path / "knowledge"
    kb_dir.mkdir()
    (kb_dir / "test_protocol.md").write_text(
        dedent("""\
            # Test Protocol
            ## Document ID: TEST-001
            ## Section A
            High risk cases require biopsy referral within 48 hours.
            ## Section B
            Low risk patients continue standard screening.
        """),
        encoding="utf-8",
    )
    return kb_dir


def _capture_cli_json(handler, args, dependency) -> dict:
    """Run a CLI handler and parse its JSON output."""
    buf = StringIO()
    with patch("sys.stdout", buf):
        handler(args, dependency)
    return json.loads(buf.getvalue())


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


class TestArgumentParsing:
    """Verify argparse setup without touching any models."""

    def test_models_command(self, parser: object) -> None:
        args = parser.parse_args(["models"])
        assert args.command == "models"

    def test_explain_requires_model_and_sample(self, parser: object) -> None:
        args = parser.parse_args(["explain", "--model", "test", "--sample", "0"])
        assert args.model == "test"
        assert args.sample == 0

    def test_explain_no_plot_flag(self, parser: object) -> None:
        args = parser.parse_args(["explain", "--model", "x", "--sample", "1", "--no-plot"])
        assert args.no_plot is True

    def test_context_requires_query(self, parser: object) -> None:
        args = parser.parse_args(["context", "--query", "biopsy referral"])
        assert args.query == "biopsy referral"
        assert args.top_k == 5  # default

    def test_context_custom_top_k(self, parser: object) -> None:
        args = parser.parse_args(["context", "--query", "test", "--top-k", "3"])
        assert args.top_k == 3

    def test_features_default_top_n(self, parser: object) -> None:
        args = parser.parse_args(["features", "--model", "test"])
        assert args.top_n == 10

    def test_pdp_requires_feature(self, parser: object) -> None:
        args = parser.parse_args(["pdp", "--model", "x", "--feature", "mean radius"])
        assert args.feature == "mean radius"

    def test_no_command_exits(self, parser: object) -> None:
        with pytest.raises(SystemExit):
            parser.parse_args([])


# ---------------------------------------------------------------------------
# Context command (knowledge base, no models needed)
# ---------------------------------------------------------------------------


class TestContextCommand:
    """Test the RAG context command independently of model registry."""

    def test_context_returns_json_with_chunks(self, knowledge_dir: Path) -> None:
        kb = load_knowledge_base(knowledge_dir)
        args = build_parser().parse_args(["context", "--query", "biopsy referral"])

        buf = StringIO()
        with patch("sys.stdout", buf):
            cmd_context(args, kb)

        result = json.loads(buf.getvalue())
        assert "chunks" in result
        assert "query" in result
        assert result["query"] == "biopsy referral"
        assert result["retrieval_method"] == "tfidf"
        assert result["provenance_label"] == "ai-interpreted"

    def test_context_returns_relevant_chunks(self, knowledge_dir: Path) -> None:
        kb = load_knowledge_base(knowledge_dir)
        args = build_parser().parse_args(["context", "--query", "biopsy"])

        buf = StringIO()
        with patch("sys.stdout", buf):
            cmd_context(args, kb)

        result = json.loads(buf.getvalue())
        assert len(result["chunks"]) > 0
        assert "biopsy" in result["chunks"][0]["text"].lower()

    def test_context_empty_kb_returns_empty_chunks(self, tmp_path: Path) -> None:
        kb = load_knowledge_base(tmp_path / "nonexistent")
        args = build_parser().parse_args(["context", "--query", "anything"])

        buf = StringIO()
        with patch("sys.stdout", buf):
            cmd_context(args, kb)

        result = json.loads(buf.getvalue())
        assert result["chunks"] == []
        assert result["documents_searched"] == []


# ---------------------------------------------------------------------------
# Models command (tests registry output shape)
# ---------------------------------------------------------------------------


class TestModelsCommand:
    """Test list_models output via CLI."""

    def test_empty_registry_narrative(self) -> None:
        registry = ModelRegistry()
        args = build_parser().parse_args(["models"])

        buf = StringIO()
        with patch("sys.stdout", buf):
            cmd_models(args, registry)

        result = json.loads(buf.getvalue())
        assert result["narrative"] == "No models are currently registered."
        assert result["grounded"] is True


# ---------------------------------------------------------------------------
# Output shape validation
# ---------------------------------------------------------------------------


class TestOutputShape:
    """Verify CLI output matches the schema contracts from schemas.py."""

    def test_context_output_matches_knowledge_search_result_schema(
        self, knowledge_dir: Path
    ) -> None:
        """The JSON shape must match KnowledgeSearchResult.model_fields."""
        kb = load_knowledge_base(knowledge_dir)
        args = build_parser().parse_args(["context", "--query", "risk"])

        buf = StringIO()
        with patch("sys.stdout", buf):
            cmd_context(args, kb)

        result = json.loads(buf.getvalue())
        required_fields = {"chunks", "query", "documents_searched", "retrieval_method", "provenance_label"}
        assert required_fields.issubset(result.keys())

    def test_models_output_matches_tool_response_schema(self) -> None:
        """The JSON shape must match ToolResponse.model_fields."""
        registry = ModelRegistry()
        args = build_parser().parse_args(["models"])

        buf = StringIO()
        with patch("sys.stdout", buf):
            cmd_models(args, registry)

        result = json.loads(buf.getvalue())
        required_fields = {"narrative", "evidence", "metadata", "grounded"}
        assert required_fields.issubset(result.keys())


# ---------------------------------------------------------------------------
# Cross-adapter parity — the ADR-001 litmus test
# ---------------------------------------------------------------------------


class TestCrossAdapterParity:
    """Prove CLI and pure functions produce identical narratives.

    This is the architectural thesis test: if ADR-001's separation of
    concerns works correctly, then the English narrative must be
    byte-for-byte identical regardless of adapter (CLI, MCP, or
    direct Python import).

    Pattern used: Back-to-back comparison (not snapshot).
    We call the pure functions directly as "ground truth", then call
    the same functions through the CLI adapter, and assert equality.
    """

    @pytest.fixture()
    def _registry(self) -> ModelRegistry:
        """Load models using the same paths as the CLI."""
        from xai_toolkit.cli import DATA_DIR, MODELS_DIR

        registry = ModelRegistry()
        registry.load_from_disk("xgboost_breast_cancer", MODELS_DIR, DATA_DIR)
        return registry

    def test_explain_narrative_matches_direct_call(
        self, _registry: ModelRegistry
    ) -> None:
        """CLI adapter passes narrator output through unchanged.

        We patch compute_shap_values to return a fixed result,
        isolating the adapter test from SHAP's minor floating-point
        non-determinism (which is tested separately in
        test_reproducibility.py).
        """
        from xai_toolkit.explainers import compute_shap_values
        from xai_toolkit.narrators import narrate_prediction

        entry = _registry.get("xgboost_breast_cancer")

        # Compute once — this is the single source of truth
        fixed_shap_result = compute_shap_values(
            model=entry.model,
            X=entry.X_test,
            sample_index=0,
            target_names=entry.metadata.get("target_names"),
        )
        expected_narrative = narrate_prediction(fixed_shap_result, top_n=3)

        # Patch so the CLI uses the exact same SHAP result
        with patch(
            "xai_toolkit.cli.compute_shap_values",
            return_value=fixed_shap_result,
        ):
            args = build_parser().parse_args([
                "explain", "--model", "xgboost_breast_cancer",
                "--sample", "0", "--no-plot",
            ])
            buf = StringIO()
            with patch("sys.stdout", buf):
                cmd_explain(args, _registry)

        cli_output = json.loads(buf.getvalue())

        assert cli_output["narrative"] == expected_narrative

    def test_context_provenance_always_ai_interpreted(
        self,
    ) -> None:
        """RAG output must always carry the Glass Floor label."""
        from xai_toolkit.cli import KNOWLEDGE_DIR

        kb = load_knowledge_base(KNOWLEDGE_DIR)
        args = build_parser().parse_args([
            "context", "--query", "high risk biopsy",
        ])
        buf = StringIO()
        with patch("sys.stdout", buf):
            cmd_context(args, kb)

        result = json.loads(buf.getvalue())
        assert result["provenance_label"] == "ai-interpreted"

    def test_pdp_feature_not_found_matches_server_error_payload(
        self, _registry: ModelRegistry
    ) -> None:
        from xai_toolkit.server import get_partial_dependence

        args = build_parser().parse_args([
            "pdp", "--model", "xgboost_breast_cancer", "--feature", "mean_radus",
        ])
        cli_output = _capture_cli_json(cmd_pdp, args, _registry)
        server_output = get_partial_dependence(
            model_id="xgboost_breast_cancer",
            feature_name="mean_radus",
            include_plot=False,
        )

        assert cli_output == server_output

    def test_feature_drift_feature_not_found_matches_server_error_payload(
        self, _registry: ModelRegistry
    ) -> None:
        from xai_toolkit.server import detect_feature_drift

        args = build_parser().parse_args([
            "feature-drift", "--model", "xgboost_breast_cancer", "--feature", "mean_radus",
        ])
        cli_output = _capture_cli_json(cmd_feature_drift, args, _registry)
        server_output = detect_feature_drift(
            model_id="xgboost_breast_cancer",
            feature_name="mean_radus",
        )

        assert cli_output == server_output
