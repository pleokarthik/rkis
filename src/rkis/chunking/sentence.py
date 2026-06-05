import uuid
from typing import List
from src.rkis.core.models import Document, Chunk
from src.rkis.chunking.base import ChunkingStrategy

class SentenceWindowChunker(ChunkingStrategy):
    """
    Groups sentences into windows. Preserves sentence boundaries.
    Better semantic coherence than fixed-size for retrieval tasks.
    """

    def __init__(self, window_size: int = 5, overlap: int = 1):
        self.window_size = window_size
        self.overlap = overlap

    def chunk(self, document: Document) -> List[Chunk]:
        assert document.id is not None, "Document must have an id before chunking"
      
        sentences = [s.strip() for s in document.content.split(".") if s.strip()]
        chunks: List[Chunk] = []
        start = 0
        index = 0

        while start < len(sentences):
            window = sentences[start:start + self.window_size]
            content = ". ".join(window).strip() + "."
            if content:
                chunks.append(
                    Chunk(
                        id=str(uuid.uuid4()),
                        document_id=document.id,
                        chunk_index=index,
                        content=content
                    )
                )
                index += 1
            start += self.window_size - self.overlap

        return chunks
