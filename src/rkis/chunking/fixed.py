import uuid
from typing import List
from core.models import Document, Chunk
from chunking.base import ChunkingStrategy

class FixedSizeChunker(ChunkingStrategy):
    """
    Splits document content into fixed-size character windows with overlap.
    No semantic awareness — fast, deterministic, baseline chunker.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> List[Chunk]:
        assert document.id is not None, "Document must have an id before chunking"
        text = document.content.strip()
        chunks: List[Chunk] = []
        start = 0
        index = 0
        while start < len(text):
            end = start + self.chunk_size
            # snap back to last space to avoid mid-word cuts
            if end < len(text) and not text[end].isspace():
                last_space = text.rfind(" ", start, end)
                if last_space > start:
                    end = last_space
            content = text[start:end].strip()
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
            start += self.chunk_size - self.overlap
       
        return chunks