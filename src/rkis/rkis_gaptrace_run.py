"""
rkis_gaptrace_run.py  —  gaptrace-capture instrumentation for RKIS
====================================================================
Place this file in src/rkis/ and run from the project root:

    python src/rkis/rkis_gaptrace_run.py

Prerequisites:
    pip install gaptrace-capture      (one-off, not pinned in requirements.txt)

Four instrumentation points, all inside a combined retrieve+rerank node —
this is the only node with both retrieval results AND assembly results in
scope, since LangGraph doesn't support post-hoc node instrumentation without
either wrapping or duplicating node logic. We duplicate, same as the prior
ctx integration did — this is a re-derivation from live rag_graph.py, not a
port of the old script's assumptions.

    Point 1 — cap.chunks()   after RRF, annotated with retrieval_path
    Point 2 — cap.chunks()   after cross-encoder rerank (scores + truncation)
    Point 3 — cap.context()  after char-budget assembly
    Point 4 — cap.response() after validate_node (final answer)

New since the prior integration: requested_count is now threaded through
Point 2's cap.chunks() call, sourced from settings.TOP_K. This is what
feeds candidate_underfill_risk — the prior ctx-capture API didn't expose
this parameter, so it was never exercised against rkis before.

Explicitly NOT instrumented, by design — rkis doesn't do any of these:
    cap.semantic_cache()   — no semantic cache in rkis's pipeline
    cap.metadata_filter()  — no metadata filtering in rkis's pipeline
    cap.cache()            — no generic cache in rkis's pipeline
    cap.tool_call()        — no tool calls in rkis's pipeline
    cap.history()          — rkis is single-turn, no conversation history
Calling these on fabricated data would misrepresent what the pipeline
actually does. Leave them dark; that's the honest result of this dogfood
pass, not a gap to paper over.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# ── gaptrace-capture ─────────────────────────────────────────────────────────
try:
    import gaptrace_capture
    from gaptrace_core import ChunkRecord, TokenBudget
except ImportError:
    print("gaptrace-capture not installed.  Run:  pip install gaptrace-capture")
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
    in_bm25 = chunk_id in bm25_ids
    if in_dense and in_bm25:
        return "both"
    return "ann" if in_dense else "bm25"


def _token_budget(context: str) -> TokenBudget:
    chunk_tokens = len(context) // 4
    system_tokens = 200
    query_tokens = 50
    headroom = 4096 - chunk_tokens - system_tokens - query_tokens
    return TokenBudget(
        total_limit=4096,
        chunks_allocated=chunk_tokens,
        history_allocated=query_tokens,
        system_allocated=system_tokens,
        headroom=max(headroom, 0),
    )


# ── combined retrieve+rerank node ─────────────────────────────────────────────

def gaptrace_retrieve_rerank_node(state: RAGState, cap) -> RAGState:
    """
    Replaces both retrieve_node and rerank_node.
    Runs identical logic to both but with gaptrace capture at each stage.
    """
    query = state["query"]
    is_evolution = state["intent"] == QueryIntent.EVOLUTION.value
    fetch_k = settings.TOP_K * (6 if is_evolution else 2)

    # ── retrieval (same as retrieve_node) ────────────────────────────────────
    query_vector = _embedder.embed_text(query)
    dense_results = _vector_store.search(query_vector, top_k=fetch_k)
    bm25_results = _bm25.search(query, top_k=fetch_k)
    fused = reciprocal_rank_fusion(dense_results, bm25_results)

    dense_ids = {r.chunk_id for r in dense_results}
    bm25_ids = {r.chunk_id for r in bm25_results}

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
    cap.chunks(pre_rerank_chunks)

    # ── rerank (same as rerank_node) ──────────────────────────────────────────
    all_reranked = _reranker.rerank(query, list(fused))
    ranked = all_reranked if is_evolution else all_reranked[:settings.TOP_K]

    # char-budget assembly — identical to rerank_node
    parts: list[str] = []
    total: int = 0
    surviving_ids: set = set()
    for r in ranked:
        text = r.payload.get("content", "")
        if total + len(text) > settings.MAX_CONTEXT_CHARS:
            break
        parts.append(text)
        total += len(text)
        surviving_ids.add(str(r.chunk_id))

    raw_context = "\n\n---\n\n".join(parts)
    print(f"DEBUG: raw_context length = {len(raw_context)} chars, MAX_CONTEXT_CHARS = {settings.MAX_CONTEXT_CHARS}, chunks_used = {len(parts)}, ranked_pool_size = {len(ranked)}")
    rerank_score_by_id = {str(r.chunk_id): float(r.score) for r in all_reranked}

    # ── Point 2: chunks with cross-encoder scores + truncation flags ──────────
    # requested_count is new — feeds candidate_underfill_risk. Sourced from
    # settings.TOP_K, the actual configured target, not fetch_k (the wider
    # pre-fusion fetch width) — TOP_K is what the pipeline actually wants to
    # end up with after rerank+truncation, which is what "underfill" means.
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
    cap.chunks(post_rerank_chunks, requested_count=settings.TOP_K)

    # ── Point 3: assembled context + token budget ─────────────────────────────
    cap.context(raw_context, _token_budget(raw_context))

    return {
        "chunks": fused,
        "sources": ranked,
        "raw_context": raw_context,
        "rerank_scores": [r.score for r in all_reranked],
    }


# ── instrumented validate node ────────────────────────────────────────────────

def gaptrace_validate_node(state: RAGState, cap) -> RAGState:
    """Same as validate_node but captures cap.response() for gaptrace."""
    from validation.answer_validator import validate_answer, CONFIDENCE_THRESHOLD

    RERANK_THRESHOLD = -5.0
    rerank_scores = state.get("rerank_scores", [])
    if rerank_scores and max(rerank_scores) < RERANK_THRESHOLD:
        cap.response("[NO RELEVANT CONTEXT FOUND]")
        return {"answer": "[NO RELEVANT CONTEXT FOUND]", "confidence_score": 0.0}

    confidence = validate_answer(state["answer"], state["raw_context"])
    answer = state["answer"]
    if confidence < CONFIDENCE_THRESHOLD:
        answer = f"[LOW CONFIDENCE ({confidence:.2f})]\n{answer}"

    # ── Point 4: final answer ─────────────────────────────────────────────────
    cap.response(answer)

    return {"answer": answer, "confidence_score": confidence}


def _route_after_rerank(state: RAGState) -> str:
    if state["intent"] == QueryIntent.EVOLUTION.value:
        return "generate_evolution"
    return "compress"


# ── graph builder ─────────────────────────────────────────────────────────────

def build_gaptrace_graph(cap):
    """
    Build a LangGraph StateGraph with gaptrace instrumentation.
    `cap` is bound into the node closures — one graph per query run.
    """
    graph = StateGraph(RAGState)

    graph.add_node("route", route_node)
    graph.add_node("retrieve_rerank",
                    lambda state: gaptrace_retrieve_rerank_node(state, cap))
    graph.add_node("compress", compress_node)
    graph.add_node("generate_factual", generate_factual_node)
    graph.add_node("generate_evolution", generate_evolution_node)
    graph.add_node("validate",
                    lambda state: gaptrace_validate_node(state, cap))

    graph.add_edge(START, "route")
    graph.add_edge("route", "retrieve_rerank")
    graph.add_conditional_edges("retrieve_rerank", _route_after_rerank, {
        "compress": "compress",
        "generate_evolution": "generate_evolution",
    })
    graph.add_edge("compress", "generate_factual")
    graph.add_edge("generate_factual", "validate")
    graph.add_edge("generate_evolution", "validate")
    graph.add_edge("validate", END)

    return graph.compile()


# ── runner ────────────────────────────────────────────────────────────────────

def run_with_gaptrace(query: str, pipeline: str = "rkis") -> dict:
    """Run one query through the instrumented graph. Returns graph result."""
    cap = gaptrace_capture.start(query=query, pipeline=pipeline)
    graph = build_gaptrace_graph(cap)
    try:
        result = graph.invoke({"query": query})
    except Exception as e:
        try:
            cap.response(f"[ERROR: {e}]")
        except Exception:
            cap.commit()
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
    print("=== RKIS gaptrace integration run ===\n")
    print(f"Running {len(QUERIES)} queries.\n")

    for i, query in enumerate(QUERIES, 1):
        print(f"[{i}/{len(QUERIES)}] {query}")
        try:
            result = run_with_gaptrace(query)
            conf = result.get("confidence_score", "?")
            answer = result.get("answer", "")[:100]
            print(f"  conf={conf:.2f}  -> {answer!r}...\n")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print()

    print("Inspect results:")
    print("  gaptrace list")
    print("  gaptrace explain --full")
    print("  gaptrace explain s1r1 --full     # path duplicate + truncation detail")
    print("  gaptrace diff s1r1 s1r2          # compare queries")
    print("  gaptrace budget s1r1             # token waterfall")


if __name__ == "__main__":
    main()
