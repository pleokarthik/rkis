# Research Insight Companion

Local-first RAG pipeline for research paper ingestion, trust-tiered source triage, hybrid retrieval, and chronological topic synthesis.

Runs entirely on local models via Ollama — no paid APIs required.

---

## What it does

Two query modes:

- **Factual** — answer a research question from ingested papers
- **Evolution** — trace how a research topic has developed chronologically across papers

An intent classifier routes automatically between modes based on the query.

---

## Architecture

```
run_query.py / run_progression.py / run_eval.py
        ↓
routing/          ← intent classification (rule-based + phi3 fallback)
        ↓
graphs/           ← LangGraph orchestration (route → retrieve → rerank → compress → generate → validate)
        ↓
retrieval/        ← hybrid search: dense (Qdrant) + BM25, fused with RRF
query/            ← CrossEncoderReranker → context assembly → generation
generation/       ← context compression (phi3)
validation/       ← chunk validation + answer validation (phi3)
        ↑
ingestion/        ← fetch → SourceGate → chunk → concept-tag → supersedes → embed → store
storage/          ← SQLite (metadata) + Qdrant (vectors)
embeddings/       ← Ollama nomic-embed-text (768-dim)
```

---

## Ingestion pipeline

Each paper goes through these passes in order:

1. **Fetch** — ArXiv client, returns title, authors, abstract, categories
2. **SourceGate triage** — phi3 classifies source as `T1` (peer-reviewed), `T2` (cited blog), or `REJECTED`. Rejected documents are dropped before storage.
3. **Save document** — SQLite metadata store
4. **Supersedes extraction** — phi3 detects if the paper claims to improve on prior work, extracts prior work title and ArXiv ID
5. **Supersedes validation** — cross-encoder scores semantic similarity between abstracts. Claims below threshold (0.3) marked unvalidated.
6. **Sentence-window chunking** — splits document into overlapping sentence windows
7. **Chunk validation** — heuristic checks (length, alpha ratio) + phi3 quality check for borderline chunks. Rejected chunks are not embedded.
8. **Concept-tag extraction** — phi3 extracts 3–5 concept tags per chunk, stored in SQLite and as `ConceptLink` rows for cross-document queries
9. **Embed** — Ollama nomic-embed-text (768-dim)
10. **Upsert to Qdrant** — local persistent vector store

---

## Retrieval pipeline

Each query goes through:

1. **Intent classification** — regex patterns for evolution vs factual keywords; phi3 fallback for ambiguous queries
2. **Hybrid retrieval** — dense search (Qdrant cosine) + BM25 keyword search run in parallel
3. **RRF fusion** — Reciprocal Rank Fusion (`1 / (k + rank)`, k=60) merges ranked lists by chunk ID
4. **CrossEncoderReranker** — sentence-transformers cross-encoder rescores fused results. Falls back to passthrough if model unavailable.
5. **Context compression** — phi3 strips redundancy while preserving technical claims, findings, and named entities. Skips compression for short contexts (<200 chars).
6. **Generation** — phi3 generates answer from compressed context
7. **Answer validation** — phi3 scores answer groundedness against original (uncompressed) source context (0.0–1.0). Answers below 0.6 flagged as `[LOW CONFIDENCE]`.

---

## LangGraph orchestration

`graphs/rag_graph.py` implements the full pipeline as a `StateGraph`:

```
route → retrieve → rerank → compress → generate → validate
```

State carries: query, intent, retrieved chunks, compressed context, answer, confidence score.

Both runners execute via the graph — not direct pipeline calls.

---

## Evaluation

`run_eval.py` runs RAGAS metrics against a golden QA set:

- Faithfulness
- Answer relevancy
- Context precision
- Context recall

Uses Ollama as the LLM backend. Output: `evaluation/results/ragas_report.json`.

```bash
python src/rkis/run_eval.py
```

---

## Stack

| Component | Implementation |
|---|---|
| LLM | Ollama phi3 (local) |
| Embeddings | Ollama nomic-embed-text, 768-dim |
| Vector store | Qdrant (local persistent) |
| Metadata store | SQLite |
| Retrieval | Dense + BM25 + RRF fusion |
| Reranker | CrossEncoderReranker (sentence-transformers) |
| Orchestration | LangGraph `StateGraph` |
| Tracing | LangSmith (optional, free tier) |
| Evaluation | RAGAS |
| Source | ArXiv |

---

## Setup

**Requirements:** Ollama running locally with `phi3` and `nomic-embed-text` pulled.

```bash
ollama pull phi3
ollama pull nomic-embed-text
```

**Install:**

```bash
pip install -r requirements.txt
```

**Configure:**

Copy `.env.example` to `.env` and set paths. Defaults work out of the box for local use.

---

## Run

```bash
# Factual query
python src/rkis/run_query.py --query "what is the role of attention in transformers" --top-k 5

# Topic evolution
python src/rkis/run_progression.py --topic "retrieval augmented generation" --papers 10

# Evaluation
python src/rkis/run_eval.py
```

Intent is classified automatically — either runner handles either query type.

---

## Project structure

```
src/rkis/
  config/         # Settings, environment
  core/           # Domain models
  ingestion/      # Fetch, SourceGate, chunking, concept tagging, supersedes
  embeddings/     # Ollama embedder
  vectorstores/   # Qdrant store
  storage/        # SQLite repository
  retrieval/      # BM25 retriever, RRF fusion
  query/          # Query pipeline, reranker, context assembly
  routing/        # Intent classifier, query router
  graphs/         # LangGraph RAG graph
  generation/     # Context compressor
  validation/     # Chunk validator, answer validator
  evaluation/     # RAGAS eval harness
  llm/            # Ollama LLM client
docs/
  FLOW.md         # Pipeline flow diagrams
  PROJECT.md      # Project scope and decisions
```

---

## Status

Active development. Core ingestion and retrieval pipelines are functional end-to-end. LlamaIndex parsing is not yet wired — ArXiv ingestion uses the `arxiv` Python client directly. Pinecone and Weaviate vector store adapters are not implemented.
