# System Flow — RKIS

---

## Ingestion Flow

Triggered by calling `ingest_arxiv(arxiv_id)` in `ingestion/ingestor.py`.

```
arxiv_id (string)
       │
       ▼
┌─────────────┐
│  fetcher.py │  fetch_arxiv()
│             │  Uses arxiv Python client → returns Document dataclass
│             │  Fields: title, url (entry_id), content (abstract),
│             │          authors, categories, published_at, tier=1
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  Dedup check        │  _doc_repo.get_by_url(document.url)
│  (sqlite_repo.py)   │  If URL already exists → return existing doc_id early
│                     │  Prevents duplicate chunks under different IDs
└──────┬──────────────┘
       │ (new paper only)
       ▼
┌──────────────────┐
│  source_gate.py  │  SourceGate.evaluate(url, title, abstract)
│                  │  Sends structured prompt to OllamaLLM (phi3)
│                  │  LLM returns JSON: {tier, reason, confidence}
│                  │  Tiers:
│                  │    T1 — peer-reviewed (ArXiv, IEEE, ACM) ✓
│                  │    T2 — research blog with citations ✓
│                  │    REJECTED — opinion/news/tutorial ✗ → raises ValueError
└──────┬───────────┘
       │ (allowed)
       ▼
┌───────────────────────┐
│  SQLiteDocumentRepo   │  save(document)
│  (sqlite_repo.py)     │  INSERT OR IGNORE (url is UNIQUE)
│                       │  Returns new doc_id (UUID)
└──────┬────────────────┘
       │
       ▼
┌──────────────────────────┐
│  SentenceWindowChunker   │  chunking/sentence.py
│  (chunking/sentence.py)  │  Splits abstract on "." into sentences
│                          │  Groups into windows of 5 sentences, overlap 1
│                          │  Each window → Chunk(document_id, chunk_index, content)
└──────┬───────────────────┘
       │ [list of Chunk]
       ▼
┌──────────────────────────────────────────────────────┐
│  For each chunk:                                     │
│                                                      │
│  1. SQLiteChunkRepo.save(chunk)                      │
│     INSERT into chunks table → returns chunk_id      │
│                                                      │
│  2. OllamaEmbedder.embed_text(chunk.content)         │
│     POST /api/embed with model=nomic-embed-text       │
│     Returns List[float] of 768 dimensions            │
│                                                      │
│  3. QdrantVectorStore.upsert(chunk_id, vector, payload)
│     payload = {document_id, chunk_index, content}    │
│     Stored as a Point in collection "rkis_chunks"    │
└──────────────────────────────────────────────────────┘
       │
       ▼
  Returns doc_id (str)
```

---

## Query Flow

Triggered by running `run_query.py` or calling `QueryPipeline.run(query)`.

```
query (string)
       │
       ▼
┌─────────────────────────────┐
│  OllamaEmbedder             │  embed_text(query)
│  (embeddings/ollama_embedder)│  Same model as ingestion: nomic-embed-text
│                             │  Returns 768-dim query vector
└──────┬──────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│  QdrantVectorStore.search()      │
│  (vectorstores/qdrant_store.py)  │
│  Cosine similarity search        │
│  Retrieves top_k * 2 candidates  │  (default: 10)
│  Each result: chunk_id, score,   │
│  payload (document_id, chunk_index, content)
└──────┬───────────────────────────┘
       │ [list of SearchResult]
       ▼
┌──────────────────────────┐
│  PassthroughReranker     │  query/reranker.py
│  (reranker.py)           │  Sorts by score descending — no re-scoring
│                          │  NOTE: CrossEncoderReranker available but bypassed
│                          │  (torch 2.5.1+cpu now working — can be re-enabled)
└──────┬───────────────────┘
       │ top_k results (default: 5)
       ▼
┌──────────────────────────────────────────────────────┐
│  _build_context()                                    │
│  Concatenates chunk content strings                  │
│  Stops if cumulative length > MAX_CONTEXT_CHARS (4000)
│  Separator: "\n\n---\n\n"                            │
└──────┬───────────────────────────────────────────────┘
       │ context string
       ▼
┌──────────────────────────────────────────────────────┐
│  _build_prompt()                                     │
│  System role: "research assistant"                   │
│  Instruction: answer using only the provided context │
│  Format: Context block → Question → "Answer:"        │
└──────┬───────────────────────────────────────────────┘
       │ prompt string
       ▼
┌─────────────────────┐
│  OllamaLLM.complete │  llm/ollama_llm.py
│  (model: phi3)      │  POST /api/chat, stream=False
│                     │  Returns response text
└──────┬──────────────┘
       │
       ▼
  QueryResult(answer=str, sources=[SearchResult, ...])
```

---

## Deletion Flow

Triggered by `delete_document(doc_id)` in `ingestion/ingestor.py`.

```
doc_id (string)
       │
       ├──▶ QdrantVectorStore.delete_by_document(doc_id)
       │    Filters collection by payload.document_id
       │    Deletes all matching points from "rkis_chunks"
       │
       ├──▶ SQLiteChunkRepo.delete_by_document(doc_id)
       │    DELETE FROM chunks WHERE document_id = ?
       │
       └──▶ SQLiteDocumentRepo.delete(doc_id)
            DELETE FROM documents WHERE id = ?
```

Deletion order matters: Qdrant first, then chunks, then the document row (respects FK integrity).

---

## Trust Tier Classification

`SourceGate` is the gatekeeper. Every new URL is evaluated before any data is written.

```
Prompt sent to phi3:
  "Classify into exactly one tier:
   - T1: peer-reviewed paper (ArXiv, IEEE, ACM)
   - T2: research blog with citations and verifiable claims
   - REJECTED: opinion, tutorial, news, uncited content"

Response parsed as JSON:
  { "tier": "t1" | "t2" | "rejected", "reason": "...", "confidence": 0.0–1.0 }
```

- `T1` and `T2` are allowed through.
- `REJECTED` raises `ValueError` immediately — nothing is written to any store.

---

## Chunking Strategies

Two implementations, selected via `get_chunker(name)`.

| Strategy | Class | Default params | Behaviour |
|---|---|---|---|
| `"sentence"` | `SentenceWindowChunker` | window=5, overlap=1 | Splits on `.`, groups 5 sentences, 1-sentence overlap |
| `"fixed"` | `FixedSizeChunker` | size=512, overlap=64 | Character windows, snaps back to last space to avoid mid-word cuts |

`"sentence"` is the active default in `ingestor.py`. It preserves semantic boundaries, making retrieval more coherent than character-level splitting.

---

## Embedding Models

| Provider | Class | Model | Dimensions | Requires |
|---|---|---|---|---|
| `"ollama"` | `OllamaEmbedder` | `nomic-embed-text` | 768 | Ollama running locally |
| `"openai"` | `OpenAIEmbedder` | `text-embedding-3-small` | configured | `OPENAI_API_KEY` |

Active: `"ollama"`. The Qdrant collection is sized to 768 dimensions to match.

---

## Key Design Decisions

**Why SQLite + Qdrant together?**
Qdrant stores vectors and payloads for fast similarity search, but is not relational. SQLite holds structured metadata (authors, categories, published date) and is the source of truth for document identity. The `url` UNIQUE constraint in SQLite is what prevents duplicate ingestion.

**Why is deduplication checked before the gate?**
The gate calls a local LLM which is slow (~seconds). Checking the URL first avoids that cost on already-ingested papers.

**Why sentence windows over fixed-size chunks?**
Academic abstracts are dense and sentence boundaries carry semantic weight. Cutting mid-sentence with a fixed window would split a claim across two chunks, hurting retrieval precision.

**Why PassthroughReranker instead of CrossEncoderReranker?**
`sentence-transformers` requires PyTorch. The installed torch version (2.12.0+cpu) failed on DLL init due to AVX-512 requirement on an i7-8550U (AVX2 only). Downgraded to `torch==2.5.1+cpu` — CrossEncoderReranker is now available to re-enable in `query/pipeline.py`.
