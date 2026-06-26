from core.models import SearchResult

RRF_K = 60


def reciprocal_rank_fusion(
    *result_lists: list[SearchResult],
    k: int = RRF_K,
) -> list[SearchResult]:
    scores: dict[str, float] = {}
    best_result: dict[str, SearchResult] = {}

    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            cid = result.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            if cid not in best_result or result.score > best_result[cid].score:
                best_result[cid] = result

    fused = []
    for cid, rrf_score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        r = best_result[cid]
        fused.append(SearchResult(chunk_id=cid, score=rrf_score, payload=r.payload))

    return fused
