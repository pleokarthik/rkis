from core.models import DocumentContribution, ProgressionResult, SearchResult
from storage.sqlite_repo import SQLiteDocumentRepository
from vectorstores import get_vector_store
from embeddings import get_embedder
from retrieval.bm25_retriever import BM25Retriever
from retrieval.rrf_fusion import reciprocal_rank_fusion
from query.reranker import get_reranker
from generation.context_compressor import compress_context
from validation.answer_validator import validate_answer, CONFIDENCE_THRESHOLD
from llm.ollama_llm import OllamaLLM
from config.settings import settings


class ProgressionPipeline:
    def __init__(self):
        self.embedder = get_embedder("ollama")
        self.vector_store = get_vector_store("qdrant")
        self.bm25 = BM25Retriever()
        self.reranker = get_reranker()
        self.doc_repo = SQLiteDocumentRepository()
        self.llm = OllamaLLM(model="phi3")

    def run(self, topic: str) -> ProgressionResult:
        fetch_k = settings.TOP_K * 6

        query_vector = self.embedder.embed_text(topic)
        dense_results = self.vector_store.search(query_vector, top_k=fetch_k)
        bm25_results = self.bm25.search(topic, top_k=fetch_k)

        fused = reciprocal_rank_fusion(dense_results, bm25_results)
        candidates = self.reranker.rerank(topic, fused)

        doc_chunks: dict[str, list[SearchResult]] = {}
        for result in candidates:
            doc_id = result.payload.get("document_id", "")
            if doc_id not in doc_chunks:
                doc_chunks[doc_id] = []
            doc_chunks[doc_id].append(result)

        documents = []
        for doc_id in doc_chunks:
            doc = self.doc_repo.get_by_id(doc_id)
            if doc:
                documents.append(doc)
        documents.sort(key=lambda d: d.published_at)

        all_raw_context_parts = []
        timeline = []
        for doc in documents:
            chunks = doc_chunks[doc.id][:3]
            raw_context = "\n\n---\n\n".join(
                r.payload.get("content", "") for r in chunks
            )
            all_raw_context_parts.append(raw_context)
            context = compress_context(raw_context)
            contribution = self._extract_contribution(topic, doc.title, context)
            timeline.append(DocumentContribution(
                doc_id=doc.id,
                title=doc.title,
                published_at=doc.published_at,
                contribution=contribution,
                chunks_used=chunks,
            ))

        narrative = self._synthesize(topic, timeline)

        full_context = "\n\n".join(all_raw_context_parts)
        confidence = validate_answer(narrative, full_context)
        if confidence < CONFIDENCE_THRESHOLD:
            narrative = f"[LOW CONFIDENCE ({confidence:.2f})]\n{narrative}"

        return ProgressionResult(
            topic=topic, narrative=narrative,
            timeline=timeline, confidence=confidence,
        )

    def _extract_contribution(self, topic: str, title: str, context: str) -> str:
        prompt = (
            f"Paper: {title}\n\n"
            f"Context:\n{context}\n\n"
            f"In 2-3 sentences, what specific contribution does this paper "
            f"make to the topic of '{topic}'?\n\n"
            "Contribution:"
        )
        return self.llm.complete(prompt)

    def _synthesize(self, topic: str, timeline: list[DocumentContribution]) -> str:
        entries = "\n\n".join(
            f"[{c.published_at}] {c.title}:\n{c.contribution}"
            for c in timeline
        )
        prompt = (
            f"Below are research papers on '{topic}', listed chronologically "
            f"with their contributions.\n\n"
            f"{entries}\n\n"
            f"Write a concise narrative (4-6 sentences) describing how '{topic}' "
            f"evolved across these papers. Focus on what changed, what problems "
            f"were solved, and what directions emerged.\n\n"
            "Narrative:"
        )
        return self.llm.complete(prompt)
