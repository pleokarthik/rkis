# Interview Questions — RKIS Project

Questions are grouped by topic. Each answer is grounded in the actual implementation.

---

## RAG (Retrieval-Augmented Generation)

**Q1. What is RAG and why did you use it here instead of fine-tuning an LLM?**

RAG keeps the LLM's weights frozen and retrieves relevant text at query time to augment the prompt. For a research assistant, this is preferable to fine-tuning because: (1) new papers can be added without retraining, (2) the LLM can cite specific passages rather than hallucinating from parametric memory, and (3) fine-tuning a model on academic abstracts would still not guarantee factual grounding on new papers it hasn't seen.

---

**Q2. Walk me through the full RAG pipeline in this project.**

1. **Ingestion**: A paper is fetched from ArXiv, gated by trust tier, chunked into sentence windows, embedded with `nomic-embed-text` (768-dim), and stored — vectors in Qdrant, metadata in SQLite.
2. **Query**: The user's question is embedded with the same model. Qdrant returns the top-2K (10) most similar chunks by cosine similarity. A reranker narrows them to top-K (5). The chunk content is concatenated into a context block (capped at 4000 chars). A prompt combining context and question is sent to `phi3` via Ollama. The answer and source references are returned.

---

**Q3. What is the difference between dense retrieval and keyword search, and which does RKIS use?**

Keyword search (BM25, TF-IDF) matches exact terms — it fails on synonyms or paraphrasing. Dense retrieval embeds both the query and documents into a shared vector space and finds matches by cosine similarity — it captures semantic meaning. RKIS uses dense retrieval via Qdrant. A hybrid approach (combining both) would improve recall but isn't implemented yet.

---

**Q4. What is the role of the reranker, and what are the two options in this codebase?**

The vector search retrieves candidates fast but imprecisely — cosine similarity of embeddings is a proxy for relevance, not a direct relevance score. A reranker scores each candidate against the query more precisely as a pair, trading speed for accuracy.

- `CrossEncoderReranker` uses `sentence-transformers`' `CrossEncoder` with `ms-marco-MiniLM-L-6-v2`. It scores `(query, chunk)` pairs directly.
- `PassthroughReranker` skips scoring and just sorts by the original vector similarity score.

The pipeline fetches `top_k * 2` candidates, reranks, then keeps `top_k` — so the reranker only needs to score 10 items, not the whole collection.

---

## Vector Databases

**Q5. Why Qdrant, and how is it configured here?**

Qdrant is an open-source vector database with a Python client and local persistent mode — no server daemon needed. It's configured with `QdrantClient(path=...)` pointing to `data/qdrant/`, which persists the collection to disk. The collection `rkis_chunks` uses cosine distance on 768-dimensional vectors (matching `nomic-embed-text`'s output).

---

**Q6. What is stored in a Qdrant point's payload, and why?**

Each point stores `{document_id, chunk_index, content}`. `document_id` and `chunk_index` are back-references to SQLite for joining structured metadata (authors, title, date). `content` is duplicated in the payload so the query pipeline can build the LLM context without an extra SQLite round-trip per chunk.

---

**Q7. How does delete work in Qdrant here, and why use a filter-based delete rather than deleting by chunk ID?**

`QdrantVectorStore.delete_by_document(doc_id)` uses a `FilterSelector` on `payload.document_id`. Deleting by chunk ID would require first querying SQLite for all chunk IDs belonging to a document — two operations. The filter-based approach deletes all matching points in one Qdrant call, which is both simpler and atomic from Qdrant's perspective.

---

## Chunking

**Q8. What is chunking and why does it matter for RAG?**

LLMs and embedding models have token limits, and embedding a full document as one unit loses local context — the embedding averages everything, making it hard to retrieve a specific claim. Chunking splits the document into smaller, semantically coherent units. Each chunk is embedded independently, so retrieval can surface the specific passage relevant to a query rather than the whole paper.

---

**Q9. Compare the two chunking strategies in this codebase.**

`FixedSizeChunker` splits on character count (default 512) with 64-char overlap. It snaps back to the last space to avoid mid-word cuts. Fast and deterministic, but can split a sentence mid-thought.

`SentenceWindowChunker` splits on `.` first to get individual sentences, then groups them into windows of 5 with 1-sentence overlap. It preserves sentence boundaries, which is better for retrieval because a claim in an abstract usually lives within 1-2 sentences.

The active chunker in `ingestor.py` is `SentenceWindowChunker`.

---

**Q10. What is chunk overlap and why is it used?**

When you split text into fixed windows without overlap, a concept that spans the boundary of two windows will be split — half in one chunk, half in the next. Overlap repeats the last N characters/sentences of one chunk as the first of the next, ensuring no boundary concept is ever fully isolated in one chunk with no context.

---

## Embeddings

**Q11. What is an embedding and what model does RKIS use?**

An embedding is a fixed-length vector of floats that represents the semantic meaning of text. Texts with similar meanings have vectors close together in cosine distance. RKIS uses `nomic-embed-text` via Ollama, which produces 768-dimensional vectors. At query time, the same model embeds the question, so the query vector and chunk vectors live in the same space.

---

**Q12. Why must the query embedding use the same model as the ingestion embedding?**

Each embedding model defines its own vector space. A vector produced by `nomic-embed-text` and a vector produced by `text-embedding-3-small` are not comparable — cosine similarity between them is meaningless. If you mix models, search returns garbage. The factory and settings are designed so both ingestion and query call `get_embedder("ollama")`.

---

## Storage and Data Integrity

**Q13. Why use both SQLite and Qdrant instead of just one?**

They serve different roles. Qdrant is optimized for approximate nearest-neighbor vector search — that's all it does. SQLite is a relational store optimized for structured queries, foreign keys, and joins. Authors, categories, trust tier, and publication date are structured metadata that shouldn't live in a vector payload. SQLite is the authoritative store for document identity; Qdrant is the retrieval index.

---

**Q14. How does the deduplication guard work and where is the bug it fixed?**

`SQLiteDocumentRepository.save()` uses `INSERT OR IGNORE`, which silently skips the insert when the `url` UNIQUE constraint fires. But it always returns a freshly generated UUID — not the existing row's ID. Chunks were then saved under this phantom ID that had no corresponding document row.

The fix is in `ingest_arxiv()`: call `_doc_repo.get_by_url(document.url)` before any gate evaluation or database writes. If the document exists, return its real stored ID immediately. Nothing else runs.

---

**Q15. Why is the dedup check placed before the SourceGate call?**

The SourceGate calls a local LLM (phi3) which takes several seconds. For a paper that's already been ingested, that cost is pure waste. Checking the URL first short-circuits on the cheap SQLite lookup.

---

## Trust Classification

**Q16. What is SourceGate and why use an LLM for it?**

SourceGate classifies each source URL + title + abstract into T1 (peer-reviewed), T2 (research blog with citations), or REJECTED (opinion, tutorial, news). Using an LLM for this avoids hand-crafted rules that would need updating as new source types emerge. The LLM is prompted to return structured JSON, which is parsed deterministically. The trade-off is latency and occasional hallucinated classification.

---

**Q17. What happens if the LLM returns malformed JSON from SourceGate?**

`_parse()` in `source_gate.py` strips markdown fences (```` ```json ``` ````) and calls `json.loads()`. If parsing fails, a `json.JSONDecodeError` propagates uncaught. There is no retry or fallback — a malformed response crashes ingestion. A production version would add a retry loop with a temperature-0 prompt.

---

## System Design

**Q18. Why is the LLM local (Ollama/phi3) rather than an API like GPT-4?**

For a research ingestion tool, processing papers locally avoids sending potentially sensitive or unpublished preprint content to third-party APIs. It also means no per-token cost at ingestion time, which matters when batch-ingesting hundreds of papers. The trade-off is answer quality — phi3 is weaker than GPT-4.

---

**Q19. What are the current limitations you would fix next?**

1. Only the abstract is ingested — not the full paper. PDFs should be parsed.
2. No REST API — the system is only usable as a Python module or via `run_query.py`.
3. `CrossEncoderReranker` is currently disabled (bypassed with `PassthroughReranker`) — this is now fixable since the torch version was corrected.
4. `ConceptLink` and the `graphs/` package are stubs — the intended knowledge graph layer doesn't exist yet.
5. No batch ingestion — each paper requires a separate call to `ingest_arxiv()`.
6. SourceGate has no retry logic for malformed LLM output.

---

**Q20. How would you scale this system to handle thousands of papers?**

- Replace the single-process Ollama embedding with a batched async embedder (Ollama supports `/api/embed` with batched input).
- Add a job queue (Celery, RQ) so ingestion is async and non-blocking.
- Move Qdrant from local file mode to a Qdrant server instance with horizontal sharding.
- Add a proper ingestion API endpoint (FastAPI) instead of calling Python functions directly.
- Add a caching layer for embeddings of frequently-queried texts.
- Add HNSW index tuning on Qdrant for recall/speed tradeoffs at large collection sizes.
