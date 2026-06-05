from abc import ABC, abstractmethod
from typing import List, Dict, Any


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
    ) -> List[Dict[str, Any]]:
        """
        Return top_k results by vector similarity.
        Each result is a dict with chunk_id, score, and payload.
        """
        ...

    @abstractmethod
    def delete(self, chunk_id: str) -> None:
        """Remove a vector by chunk_id."""
        ...