# RKIS Architecture

RKIS is a Research Knowledge Intelligence System — a retrieval-augmented knowledge
platform for collecting, validating, storing, and reasoning about technical research.

The system is not a generic RAG chatbot. It is a curated knowledge base that tracks
how concepts evolve across verified sources over time.

---

## Core Design Philosophy

- Only proven, cited knowledge enters the system. Hypotheses never enter, not even in staging.
- Every document in the database is approved by definition. The gate happens before insert.
- Concept evolution is a first-class citizen — the system tracks improvements, replacements, and current consensus.
- Components are swappable by design. Embedding model, vector store, and LLM are all replaceable without touching the core knowledge model.

---

## Source Admission Policy

The ingestion pipeline rejects at the source gate. Rejected documents are dropped — not stored.

| Source type | Decision |
| --- | --- |
| IEEE, ACM, ArXiv, peer-reviewed conference paper | Accepted — T1 |
| Research blog with strong citations and conclusive evidence | Accepted — T2 |
| Community discussion with no conclusive proof | Rejected — never stored |
| Hypothesis or speculation | Rejected — never stored |

---

## Source Trust Tiers

| Tier | Sources |
| --- | --- |
| T1 | IEEE, ACM, peer-reviewed conference papers, high-quality ArXiv |
| T2 | Conclusive research blogs, Anthropic/OpenAI research publications, cited conference summaries |
| T3 | Does not exist in this system |

---

## Concept Evolution Model

Research is stored as an improvement chain, not a flat list of documents.

```
Concept X
  -> Paper A (2022) introduced it
  -> Paper B (2023) improved it by Y
  -> Paper C (2024) replaced it with Z
  -> Current state: Z is standard
```

Queries this enables:
- What introduced this concept?
- Which paper improved or superseded the prior approach?
- What is currently the standard approach?
- Show me the full evolution of concept X from 2020 to today.

---

## Database Schema

Three tables. No staging. No status columns — every row is approved by definition.

### documents
```
id               TEXT PRIMARY KEY    — UUID
source           TEXT                — arxiv / ieee / blog
title            TEXT
url              TEXT UNIQUE
authors          TEXT                — JSON array
categories       TEXT                — JSON array
published_at     TIMESTAMP
tier             INTEGER             — 1 or 2
content          TEXT
created_at       TIMESTAMP
```

### chunks
```
id               TEXT PRIMARY KEY    — UUID
document_id      TEXT                — FK → documents.id
chunk_index      INTEGER
content          TEXT
created_at       TIMESTAMP
```

### concept_links
```
id               TEXT PRIMARY KEY    — UUID
document_id      TEXT                — FK → documents.id
concept_tag      TEXT                — SLM extracted
supersedes_id    TEXT                — FK → documents.id, nullable
verified         BOOLEAN             — cross-encoder validated
created_at       TIMESTAMP
```

Embeddings/vectors are stored in the vector store (Qdrant/Pinecone/Weaviate),
not in SQLite. SQLite owns structured metadata. Vector store owns semantic search.

---

## SLM Role — Deterministic Triage Layer

SLM acts as a smart gate before expensive LLM calls.
It handles deterministic decisions — classification, routing, validation.
LLM handles only final answer generation where language quality matters.

| Stage | SLM job |
| --- | --- |
| Source gate | Approve or reject incoming document |
| Tag extraction | Extract 3-5 clean concept tags per document |
| Supersedes extraction | Identify if this paper improves or replaces a prior work |
| Chunk validation | Is this chunk relevant enough to pass to generation? |
| Answer validation | Does the answer contradict the context? |

SLM is currently a placeholder (returns safe defaults).
Target model: Ollama phi3 or gemma2:2b — local, no API cost.

---

## Cross-Encoder Role

Cross-encoder is used in two places:

**1. Retrieval reranking**
After vector search returns top-K candidates, cross-encoder rescores them
by reading query + chunk together. More precise than cosine similarity alone.

**2. Supersedes validation**
```
[Paper A abstract + Paper B abstract] → cross-encoder → relationship score
```
SLM extracts the supersedes claim. Cross-encoder validates it by comparing
both documents directly. Verified flag is set only after this step passes.

---

## Quality Hierarchy

Every layer depends on the one below it.

```
Goal / Intent Management       → routing/ — SLM classifies intent
Context Management             → generation/ — compression, prompt window
Retrieval Planning             → graphs/ — LangGraph selects strategy
Retrieval Quality              → retrieval/ + reranker — hybrid + cross-encoder
Memory Management              → storage/ — repository pattern
Prompt Engineering             → generation/answer_engine.py
Model Quality                  → strategy pattern — swap OpenAI / Ollama
Knowledge Representation       → embeddings/ + vectorstores/
Corpus Quality                 → ingestion/ — source gate, SLM triage
```

---

## Component Map

| Folder | Responsibility |
| --- | --- |
| `config/` | Settings, env vars, constants — single source of truth |
| `core/` | Dataclass models — Document, Chunk, ConceptLink |
| `storage/` | Abstract repository + SQLite implementation |
| `ingestion/` | ArXiv source, LlamaIndex parsing, source gate |
| `chunking/` | Pluggable chunking strategies via strategy pattern |
| `embeddings/` | Abstract embedder + OpenAI + Ollama implementations |
| `vectorstores/` | Abstract store + Qdrant + Pinecone + Weaviate |
| `retrieval/` | Hybrid search + cross-encoder reranker |
| `routing/` | SLM gate — intent classification, query routing |
| `validation/` | SLM gate — chunk relevance, answer hallucination check |
| `generation/` | Prompt builder, context compression, LLM inference |
| `graphs/` | LangGraph pipeline — stateful orchestration of all above |
| `evaluation/` | RAGAS + DeepEval metrics + LangSmith tracing |

---

## Ingestion Flow

```
Document arrives
    ↓
Source gate — SLM approves or rejects
    ↓ (rejected = dropped, never stored)
Extract concept tags — SLM
    ↓
Extract supersedes claim — SLM
    ↓
Validate supersedes claim — cross-encoder
    ↓
Chunk document — LlamaIndex
    ↓
Embed chunks — OpenAI / Ollama
    ↓
Store metadata → SQLite
Store vectors  → Qdrant / Pinecone / Weaviate
Store concept links → SQLite
```

---

## Retrieval and Generation Flow

```
User query
    ↓
Intent classification — SLM gate (routing/)
    ↓
Retrieval strategy selected — LangGraph
    ↓
Hybrid search — dense + keyword
    ↓
Rerank — cross-encoder
    ↓
Chunk validation — SLM gate (validation/)
    ↓
Context compression
    ↓
Prompt construction
    ↓
LLM inference — GPT-4o-mini
    ↓
Answer validation — SLM gate (validation/)
    ↓
Grounded answer with source citations
```

---

## Design Patterns Used

| Pattern | Where | Why |
| --- | --- | --- |
| Repository | storage/ | Abstracts DB — swap SQLite for Postgres without touching pipeline |
| Strategy | embeddings/, vectorstores/, chunking/ | Swap implementations via config |
| Factory | main.py | Instantiates correct provider from config |
| Chain of Responsibility | routing/, validation/ | SLM gates in sequence, each decides pass/fail |
| Pipeline | graphs/ | LangGraph nodes — each stage is isolated and replaceable |

---

## Evaluation Stack

| Tool | Measures |
| --- | --- |
| RAGAS | Faithfulness, context recall, answer relevancy |
| DeepEval | Answer correctness, hallucination score |
| LangSmith | Full pipeline tracing, latency, token usage |

---

## Tech Stack

| Layer | Tool |
| --- | --- |
| Ingestion | LlamaIndex, ArXiv API |
| Orchestration | LangGraph |
| Embeddings | OpenAI text-embedding-3-small, Ollama (local) |
| Vector stores | Qdrant → Pinecone → Weaviate (try each) |
| Reranker | sentence-transformers cross-encoder |
| LLM | GPT-4o-mini |
| SLM (placeholder) | Ollama phi3 / gemma2:2b |
| Tracing | LangSmith |
| Evaluation | RAGAS, DeepEval |
| Storage | SQLite (metadata), vector store (embeddings) |
| Config | python-dotenv, settings.py |
