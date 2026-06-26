import logging
from ingestion.fetcher import fetch_arxiv
from ingestion.source_gate import SourceGate
from ingestion.concept_tagger import extract_concept_tags
from ingestion.supersedes_extractor import extract_supersedes
from ingestion.supersedes_validator import validate_supersedes
from validation.chunk_validator import validate_chunk
from storage.sqlite_repo import (
    SQLiteDocumentRepository,
    SQLiteChunkRepository,
    SQLiteConceptLinkRepository,
)
from core.models import ConceptLink
from chunking import get_chunker
from embeddings import get_embedder
from vectorstores import get_vector_store

logger = logging.getLogger(__name__)

_gate = SourceGate()
_doc_repo = SQLiteDocumentRepository()
_chunk_repo = SQLiteChunkRepository()
_link_repo = SQLiteConceptLinkRepository()
_chunker = get_chunker("sentence")
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

    # --- Supersedes extraction & validation ---
    claim = extract_supersedes(document.title, document.content)
    if claim.supersedes and claim.prior_title:
        prior_doc = None
        if claim.prior_arxiv_id:
            prior_url = f"http://arxiv.org/abs/{claim.prior_arxiv_id}"
            prior_doc = _doc_repo.get_by_url(prior_url)

        verified = False
        if prior_doc and prior_doc.id:
            verified = validate_supersedes(document.content, prior_doc_id=prior_doc.id)

        _link_repo.save(ConceptLink(
            document_id=doc_id,
            concept_tag=f"supersedes:{claim.prior_title}",
            supersedes_id=prior_doc.id if prior_doc else None,
            verified=verified,
        ))
        logger.info(
            "Supersedes claim: '%s' -> '%s' (verified=%s)",
            document.title, claim.prior_title, verified,
        )

    # --- Chunk, validate, tag, embed ---
    chunks = _chunker.chunk(document)
    for chunk in chunks:
        chunk.document_id = doc_id

        if not validate_chunk(chunk.content):
            logger.info("Chunk %d rejected for doc %s", chunk.chunk_index, doc_id)
            continue

        tags = extract_concept_tags(chunk.content)
        chunk_id = _chunk_repo.save(chunk, concept_tags=tags)

        for tag in tags:
            _link_repo.save(ConceptLink(
                document_id=doc_id,
                concept_tag=tag,
            ))

        vector = _embedder.embed_text(chunk.content)
        _vector_store.upsert(
            chunk_id=chunk_id,
            vector=vector,
            payload={
                "document_id": doc_id,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "concept_tags": tags,
            },
        )

    return doc_id


def delete_document(doc_id: str) -> None:
    _vector_store.delete_by_document(doc_id)
    _chunk_repo.delete_by_document(doc_id)
    _doc_repo.delete(doc_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    doc_id = ingest_arxiv("1706.03762")
    print(f"Ingested: {doc_id}")
