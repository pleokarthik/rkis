from core.models import SearchResult


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        if not results:
            return []
        pairs = [(query, r.payload.get("content", "")) for r in results]
        scores = self.model.predict(pairs)
        for result, score in zip(results, scores):
            result.score = float(score)
        return sorted(results, key=lambda r: r.score, reverse=True)


class PassthroughReranker:
    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        return sorted(results, key=lambda r: r.score, reverse=True)


def get_reranker():
    try:
        return CrossEncoderReranker()
    except Exception:
        return PassthroughReranker()
