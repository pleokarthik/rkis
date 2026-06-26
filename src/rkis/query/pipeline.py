from core.models import QueryResult, SearchResult
from vectorstores.qdrant_store import QdrantVectorStore
from embeddings import get_embedder
from llm.ollama_llm import OllamaLLM
from retrieval.bm25_retriever import BM25Retriever
from retrieval.rrf_fusion import reciprocal_rank_fusion
from query.reranker import get_reranker
from config.settings import settings


class QueryPipeline:
    def __init__(self):
        self.embedder = get_embedder("ollama")
        self.vector_store = QdrantVectorStore()
        self.bm25 = BM25Retriever()
        self.reranker = get_reranker()
        self.llm = OllamaLLM(model="phi3")

    def run(self, query: str) -> QueryResult:
        query_vector = self.embedder.embed_text(query)
        dense_results = self.vector_store.search(query_vector, top_k=settings.TOP_K * 2)
        bm25_results = self.bm25.search(query, top_k=settings.TOP_K * 2)

        fused = reciprocal_rank_fusion(dense_results, bm25_results)
        ranked = self.reranker.rerank(query, fused)[:settings.TOP_K]

        context = self._build_context(ranked)
        prompt = self._build_prompt(query, context)
        answer = self.llm.complete(prompt)

        return QueryResult(answer=answer, sources=ranked)

    def _build_context(self, results: list[SearchResult]) -> str:
        parts = []
        total = 0
        for r in results:
            text = r.payload.get("content", "")
            if total + len(text) > settings.MAX_CONTEXT_CHARS:
                break
            parts.append(text)
            total += len(text)
        return "\n\n---\n\n".join(parts)

    def _build_prompt(self, query: str, context: str) -> str:
        return (
            "You are a research assistant. Answer the question using only the provided context.\n"
            "If the context does not contain enough information, say so clearly.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
