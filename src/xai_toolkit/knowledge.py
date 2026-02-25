"""Knowledge base for business context retrieval (ADR-009).

Loads markdown documents from a directory, chunks them by section heading,
and provides keyword-based search (TF-IDF) for retrieval.

This module is a pure Python layer — no MCP imports (ADR-001).
It follows the same pattern as explainers.py and narrators.py:
    knowledge.py (pure functions) → server.py (MCP adapter)

Design decisions:
    - Chunking by markdown heading: preserves semantic boundaries.
    - TF-IDF over embeddings: sufficient for PoC, zero infrastructure.
    - Upgrade path: swap search_chunks() internals to use embeddings
      without changing the interface.

Example:
    >>> kb = load_knowledge_base(Path("knowledge/"))
    >>> results = search_chunks(kb, "biopsy referral", top_k=3)
    >>> results.chunks[0].section_heading
    'Risk Classification Thresholds'
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures (internal, not exposed via MCP)
# ---------------------------------------------------------------------------


@dataclass
class _RawChunk:
    """Internal representation of a document chunk before scoring.

    Not a Pydantic model because this never crosses the MCP boundary.
    Pydantic schemas (KnowledgeChunk, KnowledgeSearchResult) are used
    only at the adapter layer when building tool responses.
    """

    text: str
    source_document: str
    document_id: str
    section_heading: str
    chunk_index: int


@dataclass
class KnowledgeBase:
    """Container for loaded and indexed documents.

    Holds the raw chunks plus the TF-IDF index for search.
    Created by load_knowledge_base(), consumed by search_chunks().

    This is a dataclass (not Pydantic) because it contains internal
    index state that should never be serialized to JSON or cross
    the MCP boundary.
    """

    chunks: list[_RawChunk] = field(default_factory=list)
    document_names: list[str] = field(default_factory=list)
    # TF-IDF index: built lazily by _build_index()
    _idf: dict[str, float] = field(default_factory=dict, repr=False)
    _tf_vectors: list[dict[str, float]] = field(default_factory=list, repr=False)
    _indexed: bool = field(default=False, repr=False)


# ---------------------------------------------------------------------------
# Document loading and chunking
# ---------------------------------------------------------------------------


def _extract_document_id(text: str) -> str:
    """Extract a document ID from markdown metadata lines.

    Looks for patterns like '## Document ID: PROTO-2024-BC-001'.
    Returns empty string if no ID is found.

    Example:
        >>> _extract_document_id("## Document ID: PROTO-2024-BC-001")
        'PROTO-2024-BC-001'
        >>> _extract_document_id("No ID here")
        ''
    """
    match = re.search(r"Document ID:\s*(.+)", text)
    return match.group(1).strip() if match else ""


def _chunk_markdown(text: str, source_filename: str) -> list[_RawChunk]:
    """Split a markdown document into chunks at heading boundaries.

    Each chunk contains the text under a single heading (any level).
    The document preamble (text before the first heading) becomes
    chunk 0 with section_heading="preamble".

    Args:
        text: Raw markdown content.
        source_filename: Filename for provenance tracking.

    Returns:
        List of _RawChunk objects, one per section.

    Example:
        >>> chunks = _chunk_markdown("# Title\\n## Sec A\\nfoo\\n## Sec B\\nbar", "doc.md")
        >>> len(chunks)
        3
        >>> chunks[1].section_heading
        'Sec A'
    """
    document_id = _extract_document_id(text)

    # Split on markdown headings (any level: #, ##, ###, etc.)
    # Keep the heading with its section.
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    sections: list[tuple[str, str]] = []  # (heading, body)
    last_end = 0
    last_heading = "preamble"

    for match in heading_pattern.finditer(text):
        # Capture text between previous heading and this one
        body = text[last_end : match.start()].strip()
        if body:
            sections.append((last_heading, body))

        last_heading = match.group(2).strip()
        last_end = match.end()

    # Capture final section
    body = text[last_end:].strip()
    if body:
        sections.append((last_heading, body))

    return [
        _RawChunk(
            text=body,
            source_document=source_filename,
            document_id=document_id,
            section_heading=heading,
            chunk_index=i,
        )
        for i, (heading, body) in enumerate(sections)
    ]


def load_knowledge_base(knowledge_dir: Path) -> KnowledgeBase:
    """Load all markdown files from a directory into a searchable knowledge base.

    Scans for .md files, chunks each by heading, and returns a KnowledgeBase
    ready for search_chunks(). The TF-IDF index is built lazily on first search.

    Args:
        knowledge_dir: Path to directory containing .md files.

    Returns:
        KnowledgeBase with all chunks loaded. Empty if directory doesn't exist
        or contains no .md files (graceful degradation per scenario).

    Raises:
        No exceptions — returns empty KnowledgeBase on any failure.

    Example:
        >>> kb = load_knowledge_base(Path("knowledge/"))
        >>> len(kb.chunks) > 0
        True
        >>> kb.document_names
        ['clinical_protocol.md']
    """
    kb = KnowledgeBase()

    if not knowledge_dir.exists():
        logger.info("Knowledge directory %s does not exist — RAG disabled.", knowledge_dir)
        return kb

    md_files = sorted(knowledge_dir.glob("*.md"))
    if not md_files:
        logger.info("No .md files in %s — RAG disabled.", knowledge_dir)
        return kb

    for md_file in md_files:
        try:
            content = md_file.read_text(encoding="utf-8")
            chunks = _chunk_markdown(content, md_file.name)
            kb.chunks.extend(chunks)
            kb.document_names.append(md_file.name)
            logger.info(
                "Loaded %d chunks from %s", len(chunks), md_file.name
            )
        except Exception:
            logger.exception("Failed to load %s — skipping.", md_file.name)

    return kb


# ---------------------------------------------------------------------------
# TF-IDF search
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer, lowercased.

    Adequate for PoC keyword matching. Production would use a proper
    tokenizer with stemming/lemmatization.

    Example:
        >>> _tokenize("Values > 14.0 mm are clinically significant")
        ['values', '14', '0', 'mm', 'are', 'clinically', 'significant']
    """
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_index(kb: KnowledgeBase) -> None:
    """Build TF-IDF vectors for all chunks in the knowledge base.

    Mutates kb in place (sets _idf, _tf_vectors, _indexed).
    Called lazily on first search.
    """
    if kb._indexed or not kb.chunks:
        return

    n_docs = len(kb.chunks)

    # Document frequency: how many chunks contain each term
    df: Counter[str] = Counter()
    chunk_tokens: list[list[str]] = []

    for chunk in kb.chunks:
        tokens = _tokenize(chunk.text)
        chunk_tokens.append(tokens)
        unique_terms = set(tokens)
        for term in unique_terms:
            df[term] += 1

    # IDF: log(N / df) — standard formulation
    kb._idf = {
        term: math.log(n_docs / count) for term, count in df.items()
    }

    # TF vectors: term frequency per chunk
    for tokens in chunk_tokens:
        tf: Counter[str] = Counter(tokens)
        n_tokens = len(tokens) if tokens else 1
        tf_normalized = {term: count / n_tokens for term, count in tf.items()}
        kb._tf_vectors.append(tf_normalized)

    kb._indexed = True


def _score_chunk(
    query_tokens: list[str],
    chunk_tf: dict[str, float],
    idf: dict[str, float],
) -> float:
    """Compute TF-IDF similarity between a query and a chunk.

    Uses sum of TF-IDF weights for matching terms (bag-of-words model).

    Args:
        query_tokens: Tokenized search query.
        chunk_tf: Normalized term frequency vector for one chunk.
        idf: Global IDF weights.

    Returns:
        Similarity score (higher = more relevant). Not bounded to [0, 1].
    """
    score = 0.0
    for token in query_tokens:
        if token in chunk_tf:
            score += chunk_tf[token] * idf.get(token, 0.0)
    return score


def search_chunks(
    kb: KnowledgeBase,
    query: str,
    top_k: int = 5,
    min_score: float = 0.0,
) -> list[tuple[_RawChunk, float]]:
    """Search the knowledge base for chunks relevant to a query.

    Uses TF-IDF keyword matching. Returns chunks sorted by relevance
    score (highest first), filtered by min_score.

    Args:
        kb: Loaded KnowledgeBase from load_knowledge_base().
        query: Natural language search query.
        top_k: Maximum number of chunks to return.
        min_score: Minimum relevance score to include (default: include all).

    Returns:
        List of (chunk, score) tuples, sorted by descending score.
        Empty list if knowledge base is empty or no matches found.

    Example:
        >>> kb = load_knowledge_base(Path("knowledge/"))
        >>> results = search_chunks(kb, "biopsy referral", top_k=3)
        >>> len(results) <= 3
        True
        >>> results[0][1] >= results[-1][1]  # sorted by score
        True
    """
    if not kb.chunks:
        return []

    _build_index(kb)

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    scored = [
        (chunk, _score_chunk(query_tokens, tf_vec, kb._idf))
        for chunk, tf_vec in zip(kb.chunks, kb._tf_vectors)
    ]

    # Filter and sort
    scored = [(c, s) for c, s in scored if s > min_score]
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:top_k]
