from rank_bm25 import BM25Okapi
from core.models import SearchResult
from storage.sqlite_repo import get_connection


class BM25Retriever:
    def __init__(self):
        self._corpus: list[dict] = []
        self._bm25: BM25Okapi | None = None

    def _build_index(self) -> None:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id, document_id, chunk_index, content FROM chunks ORDER BY rowid")
        rows = cursor.fetchall()
        conn.close()

        self._corpus = [
            {"id": row["id"], "document_id": row["document_id"],
             "chunk_index": row["chunk_index"], "content": row["content"]}
            for row in rows
        ]
        tokenized = [doc["content"].lower().split() for doc in self._corpus]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        self._build_index()
        if not self._bm25 or not self._corpus:
            return []

        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            SearchResult(
                chunk_id=self._corpus[idx]["id"],
                score=float(score),
                payload={
                    "document_id": self._corpus[idx]["document_id"],
                    "chunk_index": self._corpus[idx]["chunk_index"],
                    "content": self._corpus[idx]["content"],
                },
            )
            for idx, score in ranked
            if score > 0
        ]
