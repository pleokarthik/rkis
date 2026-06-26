from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from core.models import SearchResult, QueryResult, ProgressionResult, DocumentContribution
from routing.query_router import QueryRouter, QueryIntent
from embeddings import get_embedder
from vectorstores.qdrant_store import QdrantVectorStore
from retrieval.bm25_retriever import BM25Retriever
from retrieval.rrf_fusion import reciprocal_rank_fusion
from query.reranker import get_reranker
from generation.context_compressor import compress_context
from validation.answer_validator import validate_answer, CONFIDENCE_THRESHOLD
from storage.sqlite_repo import SQLiteDocumentRepository
from llm.ollama_llm import OllamaLLM
from config.settings import settings


class RAGState(TypedDict, total=False):
    query: str
    intent: str
    chunks: list[SearchResult]
    raw_context: str
    compressed_context: str
    answer: str
    confidence_score: float
    sources: list[SearchResult]
    # evolution-specific
    timeline: list[DocumentContribution]


_router = QueryRouter()
_embedder = get_embedder("ollama")
_vector_store = QdrantVectorStore()
_bm25 = BM25Retriever()
_reranker = get_reranker()
_doc_repo = SQLiteDocumentRepository()
_llm = OllamaLLM(model="phi3")


def route_node(state: RAGState) -> RAGState:
    intent = _router.classify(state["query"])
    return {"intent": intent.value}


def retrieve_node(state: RAGState) -> RAGState:
    query = state["query"]
    is_evolution = state["intent"] == QueryIntent.EVOLUTION.value
    fetch_k = settings.TOP_K * (6 if is_evolution else 2)

    query_vector = _embedder.embed_text(query)
    dense_results = _vector_store.search(query_vector, top_k=fetch_k)
    bm25_results = _bm25.search(query, top_k=fetch_k)

    fused = reciprocal_rank_fusion(dense_results, bm25_results)
    return {"chunks": fused}


def rerank_node(state: RAGState) -> RAGState:
    query = state["query"]
    is_evolution = state["intent"] == QueryIntent.EVOLUTION.value
    chunks = state["chunks"]

    ranked = _reranker.rerank(query, chunks)
    if not is_evolution:
        ranked = ranked[:settings.TOP_K]

    parts = []
    total = 0
    for r in ranked:
        text = r.payload.get("content", "")
        if total + len(text) > settings.MAX_CONTEXT_CHARS:
            break
        parts.append(text)
        total += len(text)

    return {
        "sources": ranked,
        "raw_context": "\n\n---\n\n".join(parts),
    }


def compress_node(state: RAGState) -> RAGState:
    return {"compressed_context": compress_context(state["raw_context"])}


def generate_factual_node(state: RAGState) -> RAGState:
    context = state["compressed_context"]
    query = state["query"]
    prompt = (
        "You are a research assistant. Answer the question using only the provided context.\n"
        "If the context does not contain enough information, say so clearly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    return {"answer": _llm.complete(prompt)}


def generate_evolution_node(state: RAGState) -> RAGState:
    query = state["query"]
    sources = state["sources"]

    doc_chunks: dict[str, list[SearchResult]] = {}
    for result in sources:
        doc_id = result.payload.get("document_id", "")
        if doc_id not in doc_chunks:
            doc_chunks[doc_id] = []
        doc_chunks[doc_id].append(result)

    documents = []
    for doc_id in doc_chunks:
        doc = _doc_repo.get_by_id(doc_id)
        if doc:
            documents.append(doc)
    documents.sort(key=lambda d: d.published_at)

    timeline = []
    for doc in documents:
        chunks = doc_chunks[doc.id][:3]
        chunk_context = "\n\n---\n\n".join(
            r.payload.get("content", "") for r in chunks
        )
        compressed = compress_context(chunk_context)
        contribution_prompt = (
            f"Paper: {doc.title}\n\n"
            f"Context:\n{compressed}\n\n"
            f"In 2-3 sentences, what specific contribution does this paper "
            f"make to the topic of '{query}'?\n\n"
            "Contribution:"
        )
        contribution = _llm.complete(contribution_prompt)
        timeline.append(DocumentContribution(
            doc_id=doc.id,
            title=doc.title,
            published_at=doc.published_at,
            contribution=contribution,
            chunks_used=chunks,
        ))

    entries = "\n\n".join(
        f"[{c.published_at}] {c.title}:\n{c.contribution}"
        for c in timeline
    )
    narrative_prompt = (
        f"Below are research papers on '{query}', listed chronologically "
        f"with their contributions.\n\n"
        f"{entries}\n\n"
        f"Write a concise narrative (4-6 sentences) describing how '{query}' "
        f"evolved across these papers. Focus on what changed, what problems "
        f"were solved, and what directions emerged.\n\n"
        "Narrative:"
    )
    narrative = _llm.complete(narrative_prompt)
    return {"answer": narrative, "timeline": timeline}


def validate_node(state: RAGState) -> RAGState:
    confidence = validate_answer(state["answer"], state["raw_context"])
    answer = state["answer"]
    if confidence < CONFIDENCE_THRESHOLD:
        answer = f"[LOW CONFIDENCE ({confidence:.2f})]\n{answer}"
    return {"answer": answer, "confidence_score": confidence}


def _route_after_rerank(state: RAGState) -> str:
    if state["intent"] == QueryIntent.EVOLUTION.value:
        return "generate_evolution"
    return "compress"


def build_rag_graph() -> StateGraph:
    graph = StateGraph(RAGState)

    graph.add_node("route", route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("compress", compress_node)
    graph.add_node("generate_factual", generate_factual_node)
    graph.add_node("generate_evolution", generate_evolution_node)
    graph.add_node("validate", validate_node)

    graph.add_edge(START, "route")
    graph.add_edge("route", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_conditional_edges("rerank", _route_after_rerank, {
        "compress": "compress",
        "generate_evolution": "generate_evolution",
    })
    graph.add_edge("compress", "generate_factual")
    graph.add_edge("generate_factual", "validate")
    graph.add_edge("generate_evolution", "validate")
    graph.add_edge("validate", END)

    return graph.compile()


rag_graph = build_rag_graph()


def run_query(query: str) -> QueryResult:
    result = rag_graph.invoke({"query": query})
    return QueryResult(
        answer=result["answer"],
        sources=result.get("sources", []),
        confidence=result.get("confidence_score", 1.0),
    )


def run_progression(query: str) -> ProgressionResult:
    result = rag_graph.invoke({"query": query})
    return ProgressionResult(
        topic=query,
        narrative=result["answer"],
        timeline=result.get("timeline", []),
        confidence=result.get("confidence_score", 1.0),
    )
