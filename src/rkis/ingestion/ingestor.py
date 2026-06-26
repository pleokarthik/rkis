from ingestion.fetcher import fetch_arxiv
from ingestion.source_gate import SourceGate
from storage.sqlite_repo import SQLiteDocumentRepository, SQLiteChunkRepository
from chunking import get_chunker
from embeddings import get_embedder
from vectorstores import get_vector_store

_gate = SourceGate()
_doc_repo = SQLiteDocumentRepository()
_chunk_repo = SQLiteChunkRepository()
_chunker = get_chunker("sentence")
# _embedder = get_embedder("openai")
_embedder = get_embedder("ollama")
_vector_store = get_vector_store("qdrant")


def ingest_arxiv(arxiv_id: str) -> str:
    document = fetch_arxiv(arxiv_id)

    existing = _doc_repo.get_by_url(document.url)
    if existing:
        assert existing.id is not None
        return existing.id

    decision = _gate.evaluate(
        url=document.url,
        title=document.title,
        abstract=document.content,
    )

    if not decision.allowed:
        raise ValueError(f"Source rejected [{decision.tier.value}]: {decision.reason}")

    doc_id = _doc_repo.save(document)
    document.id = doc_id

    chunks = _chunker.chunk(document)
    for chunk in chunks:
        chunk.document_id = doc_id
        chunk_id = _chunk_repo.save(chunk)
        vector = _embedder.embed_text(chunk.content)
        _vector_store.upsert(
            chunk_id=chunk_id,
            vector=vector,
            payload={
                "document_id": doc_id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
            },
        )

    return doc_id


def delete_document(doc_id: str) -> None:
    _vector_store.delete_by_document(doc_id)
    _chunk_repo.delete_by_document(doc_id)
    _doc_repo.delete(doc_id)


if __name__ == "__main__":
    doc_id = ingest_arxiv("1706.03762")
    print(f"Ingested: {doc_id}") 