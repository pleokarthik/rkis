from abc import ABC,abstractmethod
from typing import List
from core.models import Document, Chunk

class ChunkingStrategy(ABC):

    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """Split document content into Chunk objects."""
        ...


