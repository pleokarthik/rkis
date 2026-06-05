from abc import ABC, abstractmethod
from typing import List

class BaseEmbedder(ABC):

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single string. Returns a float vector."""
        ...

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of strings. Returns a list of float vectors."""
        ...    