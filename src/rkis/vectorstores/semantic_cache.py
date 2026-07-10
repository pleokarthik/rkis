import uuid
from datetime import datetime, timezone
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

COLLECTION_NAME = "rkis_cache"
VECTOR_SIZE = 768  # nomic-embed-text output dimension


class SemanticCache:
    """
    Query-level semantic cache backed by its own Qdrant collection
    ("rkis_cache"), separate from the chunk index ("rkis_chunks").

    Takes an already-constructed QdrantClient rather than opening its own:
    qdrant-client's embedded/local mode takes an exclusive file lock on the
    storage directory, so a second QdrantClient(path=...) pointed at the
    same directory as QdrantVectorStore's would fail to open. Collections
    are cheap and share a client fine; separate embedded instances don't.
    """

    def __init__(self, client: QdrantClient):
        self.client = client
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME not in existing:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )

    def search(
        self, query_vector: list[float], intent: str, threshold: float
    ) -> Optional[tuple[str, float, Optional[str]]]:
        """Return (answer, similarity_score, cached_at) for the best match
        with matching intent, or None if there's no match or it's below
        threshold."""
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=1,
            query_filter=Filter(
                must=[FieldCondition(key="intent", match=MatchValue(value=intent))]
            ),
            with_payload=True,
        ).points

        if not results:
            return None

        top = results[0]
        if top.score < threshold:
            return None

        payload = top.payload or {}
        return payload.get("answer", ""), float(top.score), payload.get("timestamp")

    def store(self, query_vector: list[float], answer: str, intent: str) -> None:
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=query_vector,
                    payload={
                        "answer": answer,
                        "intent": intent,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )
            ],
        )
