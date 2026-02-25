# ADR-009: RAG Augmentation with Glass Floor Provenance

## Status: Accepted
## Date: 2025-02-25

## Context

The XAI toolkit produces deterministic, auditable explanations of model
predictions (ADR-002). However, stakeholders need more than "why did the
model predict X" — they need "what should I do about it?" This requires
domain-specific business context: regulatory thresholds, operational
procedures, clinical protocols, etc.

This business knowledge lives in documents outside the codebase (markdown
files for the PoC, potentially PDFs/Word docs in production). The challenge
is enriching agent responses with this context without compromising the
deterministic guarantees that are a core selling point (ADR-002).

## Decision

We introduce a RAG (Retrieval-Augmented Generation) layer that:

1. **Adds a new pure Python module** (`knowledge.py`) that loads, chunks,
   and searches business documents. No MCP imports (ADR-001).

2. **Adds a new MCP tool** (`retrieve_business_context`) that exposes
   the knowledge base to the LLM via the existing adapter pattern.

3. **Enforces "Glass Floor" provenance** — the deterministic explanation
   (Layer 1) is always presented first and unmodified. Business context
   interpretation (Layer 2) is additive, clearly labeled as AI-interpreted,
   and always cites source documents.

4. **Uses keyword-based retrieval (TF-IDF)** for the PoC. The architecture
   supports swapping in embedding-based retrieval without interface changes.

5. **Business documents are project-specific** and live in a `knowledge/`
   directory, keeping the generic toolkit clean.

## Consequences

- ✅ Business users get actionable guidance, not just technical explanations
- ✅ Deterministic outputs remain unchanged and auditable
- ✅ Clear visual separation between computed facts and AI interpretation
- ✅ Generic toolkit stays domain-agnostic (knowledge/ is project-specific)
- ✅ Knowledge module is testable with pytest (no MCP dependency)
- ⚠️ Layer 2 outputs are non-deterministic (LLM synthesizes retrieved context)
- ⚠️ Retrieval quality depends on document structure and query formulation
- ⚠️ Context window usage increases when business context is retrieved

## Alternatives Considered

- **Embedding business rules directly in narrator functions** (rejected:
  couples domain knowledge to generic toolkit, violates ADR-001)
- **Full vector database with embeddings** (rejected for PoC: adds
  infrastructure complexity without architectural value; TF-IDF
  demonstrates the pattern identically; upgrade path is clear)
- **LLM-generated interpretations without retrieval** (rejected:
  hallucination risk too high without grounding in actual documents)
- **No RAG — keep toolkit purely deterministic** (rejected: the value
  gap between "what the model says" and "what to do about it" is the
  primary business pain point)

## Upgrade Path

| PoC (now)           | Production (future)              |
|---------------------|----------------------------------|
| Markdown documents  | PDF/Word ingestion pipeline      |
| TF-IDF search       | Embedding-based vector search    |
| Local knowledge/    | Databricks Vector Search index   |
| Single directory    | Multi-project knowledge scoping  |

The pure function interface (`load_knowledge_base()`, `search_chunks()`)
remains identical regardless of the retrieval backend.
