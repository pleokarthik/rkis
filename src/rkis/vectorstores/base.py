from abc import ABC, abstractmethod
from typing import List, Dict, Any
from core.models import SearchResult

class BaseVectorStore(ABC):

    @abstractmethod
    def upsert(
        self,
        chunk_id: str,
        vector: List[float],
        payload: Dict[str, Any],
    ) -> None:
        """Store a vector with its metadata payload."""
        ...

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filters: Dict[str, Any] | None = None,
    ) -> List[SearchResult]:
        """
        Return top_k results by vector similarity.
        Each result is a dict with chunk_id, score, and payload.
        """
        ...

    @abstractmethod
    def delete(self, chunk_id: str) -> None:
        """Remove a vector by chunk_id."""
        ...

    @abstractmethod
    def delete_by_document(self, document_id: str) -> None:
        """Remove all vectors whose payload.document_id matches."""
        ...