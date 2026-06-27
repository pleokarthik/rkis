"""
rkis_ctx_run.py  —  ctx instrumentation for RKIS
==================================================
Place this file in src/rkis/ and run from the project root:

    python src/rkis/rkis_ctx_run.py

Prerequisites:
    pip install ctx-capture

Four instrumentation points, all inside rerank_node which is the only node
that has both retrieval results AND assembly results in scope:

    Point 1 — run.chunks()   after RRF, annotated with retrieval_path
    Point 2 — run.chunks()   after cross-encoder rerank (scores + truncation)
    Point 3 — run.context()  after char-budget assembly
    Point 4 — run.response() after validate_node (final answer)

Design choice: we do NOT patch retrieve_node separately. Instead we re-run
retrieval inside a single wrapper that has access to both dense_results and
bm25_results before fusion, which is the only way to annotate retrieval_path.
The wrapper replaces both retrieve_node AND rerank_node in one combined node.
This avoids thread-local state passing between nodes.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# ── ctx-capture ───────────────────────────────────────────────────────────────
try:
    import ctxrun
    from ctxrun import ChunkRecord, TokenBudget
except ImportError:
    print("ctx-capture not installed.  Run:  pip install ctx-capture")
    sys.exit(1)

# ── RKIS imports ──────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, START, END
from graphs.rag_graph import (
    RAGState,
    route_node,
    compress_node,
    generate_factual_node,
    generate_evolution_node,
    _embedder,
    _vector_store,
    _bm25,
    _reranker,
)
from retrieval.rrf_fusion import reciprocal_rank_fusion
from routing.query_router import QueryIntent
from config.settings import settings


# ── helpers ───────────────────────────────────────────────────────────────────

def _retrieval_path(chunk_id, dense_ids: set, bm25_ids: set) -> str:
    in_dense = chunk_id in dense_ids
    in_bm25  = chunk_id in bm25_ids
    if in_dense and in_bm25:
        return "both"
    return "ann" if in_dense else "bm25"


def _token_budget(context: str) -> TokenBudget:
    chunk_tokens  = len(context) // 4
    system_tokens = 200
    query_tokens  = 50
    headroom      = 4096 - chunk_tokens - system_tokens - query_tokens
    return TokenBudget(
        total_limit=4096,
        chunks_allocated=chunk_tokens,
        history_allocated=query_tokens,
        system_allocated=system_tokens,
        headroom=max(headroom, 0),
    )


# ── combined retrieve+rerank node ─────────────────────────────────────────────

def ctx_retrieve_rerank_node(state: RAGState, run) -> RAGState:
    """
    Replaces both retrieve_node and rerank_node.
    Runs identical logic to both but with ctx capture at each stage.
    """
    query        = state["query"]
    is_evolution = state["intent"] == QueryIntent.EVOLUTION.value
    fetch_k      = settings.TOP_K * (6 if is_evolution else 2)

    # ── retrieval (same as retrieve_node) ────────────────────────────────────
    query_vector  = _embedder.embed_text(query)
    dense_results = _vector_store.search(query_vector, top_k=fetch_k)
    bm25_results  = _bm25.search(query, top_k=fetch_k)
    fused         = reciprocal_rank_fusion(dense_results, bm25_results)

    dense_ids = {r.chunk_id for r in dense_results}
    bm25_ids  = {r.chunk_id for r in bm25_results}

    # ── Point 1: chunks after retrieval, with retrieval_path annotation ───────
    pre_rerank_chunks = [
        ChunkRecord(
            chunk_id=str(r.chunk_id),
            source_doc_id=r.payload.get("document_id", ""),
            content=r.payload.get("content", ""),
            token_count=len(r.payload.get("content", "")) // 4,
            retrieval_score=float(r.score),
            rerank_score=None,
            retrieval_path=_retrieval_path(r.chunk_id, dense_ids, bm25_ids),
            truncated=False,
        )
        for r in fused
    ]
    run.chunks(pre_rerank_chunks)

    # ── rerank (same as rerank_node) ──────────────────────────────────────────
    all_reranked = _reranker.rerank(query, list(fused))
    ranked       = all_reranked if is_evolution else all_reranked[:settings.TOP_K]

    # char-budget assembly — identical to rerank_node
    parts:         list[str] = []
    total:         int        = 0
    surviving_ids: set        = set()
    for r in ranked:
        text = r.payload.get("content", "")
        if total + len(text) > settings.MAX_CONTEXT_CHARS:
            break
        parts.append(text)
        total += len(text)
        surviving_ids.add(str(r.chunk_id))

    raw_context = "\n\n---\n\n".join(parts)

    rerank_score_by_id = {str(r.chunk_id): float(r.score) for r in all_reranked}

    # ── Point 2: chunks with cross-encoder scores + truncation flags ──────────
    post_rerank_chunks = [
        ChunkRecord(
            chunk_id=c.chunk_id,
            source_doc_id=c.source_doc_id,
            content=c.content,
            token_count=c.token_count,
            retrieval_score=c.retrieval_score,
            rerank_score=rerank_score_by_id.get(c.chunk_id),
            retrieval_path=c.retrieval_path,
            truncated=(c.chunk_id not in surviving_ids),
        )
        for c in pre_rerank_chunks
    ]
    run.chunks(post_rerank_chunks)

    # ── Point 3: assembled context + token budget ─────────────────────────────
    run.context(raw_context, _token_budget(raw_context))

    return {
        "chunks":        fused,
        "sources":       ranked,
        "raw_context":   raw_context,
        "rerank_scores": [r.score for r in all_reranked],
    }


# ── instrumented validate node ────────────────────────────────────────────────

def ctx_validate_node(state: RAGState, run) -> RAGState:
    """Same as validate_node but captures run.response() for ctx."""
    from validation.answer_validator import validate_answer, CONFIDENCE_THRESHOLD

    RERANK_THRESHOLD = -5.0
    rerank_scores = state.get("rerank_scores", [])
    if rerank_scores and max(rerank_scores) < RERANK_THRESHOLD:
        run.response("[NO RELEVANT CONTEXT FOUND]")
        return {"answer": "[NO RELEVANT CONTEXT FOUND]", "confidence_score": 0.0}

    confidence = validate_answer(state["answer"], state["raw_context"])
    answer = state["answer"]
    if confidence < CONFIDENCE_THRESHOLD:
        answer = f"[LOW CONFIDENCE ({confidence:.2f})]\n{answer}"

    # ── Point 4: final answer ─────────────────────────────────────────────────
    run.response(answer)

    return {"answer": answer, "confidence_score": confidence}


def _route_after_rerank(state: RAGState) -> str:
    if state["intent"] == QueryIntent.EVOLUTION.value:
        return "generate_evolution"
    return "compress"


# ── graph builder ─────────────────────────────────────────────────────────────

def build_ctx_graph(run):
    """
    Build a LangGraph StateGraph with ctx instrumentation.
    `run` is bound into the node closures — one graph per query run.
    """
    graph = StateGraph(RAGState)

    graph.add_node("route",    route_node)
    graph.add_node("retrieve_rerank",
                   lambda state: ctx_retrieve_rerank_node(state, run))
    graph.add_node("compress",           compress_node)
    graph.add_node("generate_factual",   generate_factual_node)
    graph.add_node("generate_evolution", generate_evolution_node)
    graph.add_node("validate",
                   lambda state: ctx_validate_node(state, run))

    graph.add_edge(START, "route")
    graph.add_edge("route", "retrieve_rerank")
    graph.add_conditional_edges("retrieve_rerank", _route_after_rerank, {
        "compress":           "compress",
        "generate_evolution": "generate_evolution",
    })
    graph.add_edge("compress",           "generate_factual")
    graph.add_edge("generate_factual",   "validate")
    graph.add_edge("generate_evolution", "validate")
    graph.add_edge("validate", END)

    return graph.compile()


# ── runner ────────────────────────────────────────────────────────────────────

def run_with_ctx(query: str, pipeline: str = "rkis") -> dict:
    """Run one query through the instrumented graph. Returns graph result."""
    run   = ctxrun.start(query=query, pipeline=pipeline)
    graph = build_ctx_graph(run)
    try:
        result = graph.invoke({"query": query})
    except Exception as e:
        try:
            run.response(f"[ERROR: {e}]")
        except Exception:
            run.commit()
        raise
    return result


# ── queries ───────────────────────────────────────────────────────────────────

QUERIES = [
    "what is the role of attention in transformers",
    "retrieval augmented generation evaluation benchmarks",
    "speculative decoding large language models latency",
    "context length extrapolation positional encoding",
]


def main() -> None:
    print("=== RKIS ctx integration run ===\n")
    print(f"Running {len(QUERIES)} queries.\n")

    for i, query in enumerate(QUERIES, 1):
        print(f"[{i}/{len(QUERIES)}] {query}")
        try:
            result = run_with_ctx(query)
            conf   = result.get("confidence_score", "?")
            answer = result.get("answer", "")[:100]
            print(f"  conf={conf:.2f}  → {answer!r}...\n")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print()

    print("Inspect results:")
    print("  ctx list")
    print("  ctx explain --full")
    print("  ctx explain s1r1 --full     # path duplicate + truncation detail")
    print("  ctx diff s1r1 s1r2          # compare queries")


if __name__ == "__main__":
    main()
