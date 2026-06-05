from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from src.rkis.vectorstores.base import BaseVectorStore
from src.rkis.config.settings import settings


COLLECTION_NAME = "rkis_chunks"
VECTOR_SIZE = 1536  # text-embedding-3-small output dimension


class QdrantVectorStore(BaseVectorStore):
    """
    Qdrant implementation of BaseVectorStore.
    Runs locally via in-memory or persistent mode.
    Collection is created on first use if it doesn't exist.
    """

    def __init__(self):
        self.client = QdrantClient(path=str(settings.BASE_DIR / "data" / "qdrant"))
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME not in existing:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )

    def upsert(
        self,
        chunk_id: str,
        vector: List[float],
        payload: Dict[str, Any],
    ) -> None:
        self.client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=chunk_id,
                    vector=vector,
                    payload=payload,
                )
            ],
        )

    def search(
    self,
    query_vector: List[float],
    top_k: int = 10,
    filters: Dict[str, Any] | None = None,
    ) -> List[Dict[str, Any]]:    
        qdrant_filter = None
        if filters:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(
                        key=key,
                        match=MatchValue(value=value),
                    )
                    for key, value in filters.items()
                ]
            )

        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        ).points

        return [
            {
                "chunk_id": hit.id,
                "score": hit.score,
                "payload": hit.payload,
            }
            for hit in results
        ]

    def delete(self, chunk_id: str) -> None:
        self.client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=[chunk_id],
        )