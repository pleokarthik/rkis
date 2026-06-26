# RKIS — Research Knowledge Ingestion System

A local RAG (Retrieval-Augmented Generation) pipeline that ingests academic papers from ArXiv, stores them in a hybrid SQLite + Qdrant store, and answers research questions using a locally-running LLM via Ollama.

---

## What It Does

1. Fetches a paper from ArXiv by ID.
2. Runs it through a trust classifier (SourceGate) to reject low-quality sources.
3. Chunks the abstract into sentence windows.
4. Embeds each chunk using a local Ollama embedding model (`nomic-embed-text`).
5. Stores metadata in SQLite and vectors in Qdrant.
6. At query time, embeds the question, retrieves the top-K closest chunks, reranks them, and sends the context to a local LLM (`phi3`) to generate an answer.

---

## Project Layout

```
src/rkis/
├── config/
│   └── settings.py          # Central config (paths, model names, knobs)
├── core/
│   └── models.py            # Dataclasses: Document, Chunk, SearchResult, etc.
├── ingestion/
│   ├── fetcher.py           # arxiv API → Document
│   ├── source_gate.py       # LLM-based trust classifier (T1/T2/REJECTED)
│   └── ingestor.py          # Orchestrates fetch → gate → chunk → embed → store
├── chunking/
│   ├── base.py              # ChunkingStrategy ABC
│   ├── fixed.py             # FixedSizeChunker (character windows)
│   ├── sentence.py          # SentenceWindowChunker (default)
│   └── factory.py           # get_chunker("sentence" | "fixed")
├── embeddings/
│   ├── base.py              # BaseEmbedder ABC
│   ├── ollama_embedder.py   # OllamaEmbedder — nomic-embed-text, 768-dim
│   ├── openai_embedder.py   # OpenAIEmbedder — text-embedding-3-small
│   └── factory.py           # get_embedder("ollama" | "openai")
├── storage/
│   ├── repository.py        # Abstract repository interfaces
│   └── sqlite_repo.py       # SQLite implementations for documents/chunks/concept_links
├── vectorstores/
│   ├── base.py              # BaseVectorStore ABC
│   ├── qdrant_store.py      # QdrantVectorStore (local persistent mode)
│   └── factory.py           # get_vector_store("qdrant")
├── llm/
│   └── ollama_llm.py        # OllamaLLM — wraps /api/chat, defaults to phi3
├── query/
│   ├── reranker.py          # CrossEncoderReranker + PassthroughReranker
│   └── pipeline.py          # QueryPipeline — embed → search → rerank → generate
└── run_query.py             # Entry point for running a query
```

---

## Data Model

| Model | Purpose |
|---|---|
| `Document` | One ArXiv paper. Has `id`, `url` (unique), `title`, `content` (abstract), `tier`, `authors`, `categories`. |
| `Chunk` | A sub-section of a document's content. Has `document_id`, `chunk_index`, `content`. |
| `SearchResult` | Returned by Qdrant: `chunk_id`, `score`, `payload` (contains `document_id`, `chunk_index`, `content`). |
| `QueryResult` | Final output: `answer` string + list of `SearchResult` sources. |
| `GateDecision` | Output of SourceGate: `tier` (T1/T2/REJECTED), `allowed`, `reason`, `confidence`. |
| `ConceptLink` | Relationship between a document and a concept tag (future graph feature). |

---

## Storage

### SQLite (`data/rkis.db`)
- **documents** — one row per paper; `url` is UNIQUE to prevent duplicates.
- **chunks** — one row per chunk; FK to `documents.id`.
- **concept_links** — concept-to-document edges (not yet used in query path).

### Qdrant (`data/qdrant/`)
- Single collection: `rkis_chunks`.
- Each point: `id = chunk_id` (UUID), `vector` = 768-dim float, `payload = {document_id, chunk_index, content}`.
- Runs in local persistent mode — no server process needed.

---

## Configuration (`config/settings.py`)

| Key | Default | Purpose |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `DB_PATH` | `data/rkis.db` | SQLite file |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Used by OpenAI embedder only |
| `CHUNK_SIZE` | 512 | Used by FixedSizeChunker |
| `CHUNK_OVERLAP` | 50 | Used by FixedSizeChunker |
| `TOP_K` | 5 | Number of results returned to LLM |
| `MAX_CONTEXT_CHARS` | 4000 | Context window cap for LLM prompt |
| `ARXIV_CATEGORIES` | cs.AI, cs.LG, cs.CL | Target paper categories |

---

## Setup

```bash
# 1. Create and activate venv
python -m venv .venv
.venv\Scripts\activate

# 2. Install dependencies (torch pinned to avoid AVX-512 issue on older CPUs)
pip install -r requirements.txt

# 3. Start Ollama and pull required models
ollama pull nomic-embed-text
ollama pull phi3

# 4. Ingest a paper
cd src/rkis
python -c "from ingestion.ingestor import ingest_arxiv; print(ingest_arxiv('1706.03762'))"

# 5. Run a query
python run_query.py
```

---

## Active Limitations

- Only ArXiv is supported as an ingestion source.
- Only the abstract is stored and embedded — not the full paper body.
- `CrossEncoderReranker` is bypassed (`PassthroughReranker` active) — was blocked by a torch DLL issue (now resolved by pinning `torch==2.5.1+cpu`).
- `retrieval/`, `generation/`, and `graphs/` packages are empty placeholders.
- No REST API or UI — entry point is `run_query.py`.
