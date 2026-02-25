"""Tests for the knowledge base module (ADR-009).

Maps to scenarios:
    - scenarios/rag/knowledge_base_loads.yaml
    - scenarios/rag/retrieve_business_context.yaml
    - scenarios/rag/graceful_degradation.yaml

Tests follow the same pytest conventions as test_narrators.py:
    - Fixtures create minimal test data
    - Each test validates one assertion from a scenario
    - Parametrize where multiple cases exist
"""

from pathlib import Path
from textwrap import dedent

import pytest

from xai_toolkit.knowledge import (
    KnowledgeBase,
    _chunk_markdown,
    _extract_document_id,
    _tokenize,
    load_knowledge_base,
    search_chunks,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


SAMPLE_PROTOCOL = dedent("""\
    # Breast Cancer Screening Protocol

    ## Document ID: PROTO-2024-BC-001
    ## Effective Date: 2024-01-15

    ## 1. Risk Classification Thresholds

    - High Risk: probability >= 0.85 → Immediate biopsy referral
    - Moderate Risk: probability 0.60–0.84 → Additional imaging
    - Low Risk: probability < 0.60 → Standard screening schedule

    ## 2. Feature-Specific Clinical Guidelines

    ### mean_radius
    - Values > 14.0 mm are clinically significant
    - When mean_radius is a top SHAP contributor AND value > 14.0:
      recommend ultrasound measurement verification before biopsy

    ### texture_mean
    - Elevated texture (> 20.0) suggests irregular cell architecture
    - When combined with high mean_radius: increase urgency tier

    ## 3. Operational Rules

    All high-risk classifications must be reviewed within 48 hours.
    Every explanation must include the model confidence and top 3 features.
""")


@pytest.fixture()
def knowledge_dir(tmp_path: Path) -> Path:
    """Create a temporary knowledge directory with a sample protocol."""
    kb_dir = tmp_path / "knowledge"
    kb_dir.mkdir()
    (kb_dir / "clinical_protocol.md").write_text(SAMPLE_PROTOCOL, encoding="utf-8")
    return kb_dir


@pytest.fixture()
def empty_knowledge_dir(tmp_path: Path) -> Path:
    """Create an empty knowledge directory (no .md files)."""
    kb_dir = tmp_path / "knowledge"
    kb_dir.mkdir()
    return kb_dir


@pytest.fixture()
def loaded_kb(knowledge_dir: Path) -> KnowledgeBase:
    """A knowledge base loaded from the sample protocol."""
    return load_knowledge_base(knowledge_dir)


# ---------------------------------------------------------------------------
# Scenario: knowledge_base_loads.yaml — Document parsing
# ---------------------------------------------------------------------------


class TestDocumentParsing:
    """Validates: documents_parsed_into_chunks, each_chunk_has_field, etc."""

    def test_extract_document_id_present(self) -> None:
        text = "## Document ID: PROTO-2024-BC-001\nSome content"
        assert _extract_document_id(text) == "PROTO-2024-BC-001"

    def test_extract_document_id_missing(self) -> None:
        assert _extract_document_id("No ID here") == ""

    def test_chunk_markdown_splits_by_heading(self) -> None:
        chunks = _chunk_markdown(SAMPLE_PROTOCOL, "test.md")
        assert len(chunks) > 3  # scenario: chunk_count_greater_than: 3

    def test_each_chunk_has_required_fields(self, loaded_kb: KnowledgeBase) -> None:
        """Scenario assertion: each_chunk_has_field for text, source_document, etc."""
        for chunk in loaded_kb.chunks:
            assert chunk.text, "Chunk text must not be empty"
            assert chunk.source_document == "clinical_protocol.md"
            assert chunk.section_heading  # non-empty
            assert isinstance(chunk.chunk_index, int)

    def test_document_id_propagated_to_chunks(self, loaded_kb: KnowledgeBase) -> None:
        """Scenario assertion: each_chunk_has_field: document_id."""
        for chunk in loaded_kb.chunks:
            assert chunk.document_id == "PROTO-2024-BC-001"

    def test_document_names_tracked(self, loaded_kb: KnowledgeBase) -> None:
        assert "clinical_protocol.md" in loaded_kb.document_names

    def test_chunks_ordered_by_index(self, loaded_kb: KnowledgeBase) -> None:
        indices = [c.chunk_index for c in loaded_kb.chunks]
        assert indices == sorted(indices)


# ---------------------------------------------------------------------------
# Scenario: knowledge_base_loads.yaml — Search functionality
# ---------------------------------------------------------------------------


class TestSearch:
    """Validates: search_for 'biopsy', search_returns_relevant_chunks, etc."""

    def test_search_returns_results_for_biopsy(self, loaded_kb: KnowledgeBase) -> None:
        """Scenario assertion: search_for 'biopsy' returns relevant chunks."""
        results = search_chunks(loaded_kb, "biopsy")
        assert len(results) > 0

    def test_top_result_mentions_biopsy(self, loaded_kb: KnowledgeBase) -> None:
        """Scenario assertion: top_result_mentions 'biopsy'."""
        results = search_chunks(loaded_kb, "biopsy referral")
        top_chunk, _score = results[0]
        assert "biopsy" in top_chunk.text.lower()

    def test_results_sorted_by_score_descending(self, loaded_kb: KnowledgeBase) -> None:
        results = search_chunks(loaded_kb, "risk classification threshold")
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_results(self, loaded_kb: KnowledgeBase) -> None:
        results = search_chunks(loaded_kb, "risk", top_k=2)
        assert len(results) <= 2

    def test_search_for_feature_finds_relevant_section(
        self, loaded_kb: KnowledgeBase
    ) -> None:
        """Search for 'mean_radius' should find the feature guidelines section."""
        results = search_chunks(loaded_kb, "mean_radius clinically significant")
        top_chunk, _ = results[0]
        assert "14.0" in top_chunk.text or "mean_radius" in top_chunk.text.lower()

    def test_empty_query_returns_no_results(self, loaded_kb: KnowledgeBase) -> None:
        results = search_chunks(loaded_kb, "")
        assert results == []

    @pytest.mark.parametrize(
        "query,expected_term",
        [
            ("biopsy referral high risk", "biopsy"),
            ("texture irregular cell", "texture"),
            ("48 hours review operational", "48"),
        ],
    )
    def test_various_queries_find_relevant_content(
        self, loaded_kb: KnowledgeBase, query: str, expected_term: str
    ) -> None:
        results = search_chunks(loaded_kb, query)
        assert len(results) > 0
        top_text = results[0][0].text.lower()
        assert expected_term in top_text


# ---------------------------------------------------------------------------
# Scenario: graceful_degradation.yaml
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """Validates: system works when no knowledge base exists."""

    def test_missing_directory_returns_empty_kb(self, tmp_path: Path) -> None:
        """Scenario: knowledge_dir_exists: false → no error."""
        kb = load_knowledge_base(tmp_path / "nonexistent")
        assert len(kb.chunks) == 0
        assert kb.document_names == []

    def test_empty_directory_returns_empty_kb(self, empty_knowledge_dir: Path) -> None:
        kb = load_knowledge_base(empty_knowledge_dir)
        assert len(kb.chunks) == 0

    def test_search_on_empty_kb_returns_empty(self) -> None:
        kb = KnowledgeBase()
        results = search_chunks(kb, "anything")
        assert results == []


# ---------------------------------------------------------------------------
# Tokenizer unit tests
# ---------------------------------------------------------------------------


class TestTokenizer:
    def test_basic_tokenization(self) -> None:
        tokens = _tokenize("Values > 14.0 mm are clinically significant")
        assert "values" in tokens
        assert "14" in tokens
        assert "clinically" in tokens

    def test_lowercases(self) -> None:
        tokens = _tokenize("SHAP Values")
        assert tokens == ["shap", "values"]

    def test_empty_string(self) -> None:
        assert _tokenize("") == []


# ---------------------------------------------------------------------------
# Multi-document test
# ---------------------------------------------------------------------------


class TestMultiDocument:
    """Validates knowledge base works with multiple documents."""

    def test_two_documents_both_searchable(self, tmp_path: Path) -> None:
        kb_dir = tmp_path / "knowledge"
        kb_dir.mkdir()
        (kb_dir / "protocol_a.md").write_text(
            "# Protocol A\n## Section\nAlpha biopsy procedure", encoding="utf-8"
        )
        (kb_dir / "protocol_b.md").write_text(
            "# Protocol B\n## Section\nBeta radiation guidelines", encoding="utf-8"
        )

        kb = load_knowledge_base(kb_dir)
        assert len(kb.document_names) == 2

        # Search should find content from both
        results_a = search_chunks(kb, "biopsy")
        assert any("protocol_a.md" in c.source_document for c, _ in results_a)

        results_b = search_chunks(kb, "radiation")
        assert any("protocol_b.md" in c.source_document for c, _ in results_b)
