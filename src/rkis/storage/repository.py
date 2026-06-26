from abc import ABC, abstractmethod
from typing import Optional
from core.models import Document, Chunk, ConceptLink


class DocumentRepository(ABC):

    @abstractmethod
    def save(self, document: Document) -> str:
        pass

    @abstractmethod
    def get_by_id(self, doc_id: str) -> Optional[Document]:
        pass

    @abstractmethod
    def get_by_url(self, url: str) -> Optional[Document]:
        pass

    @abstractmethod
    def exists(self, url: str) -> bool:
        pass

    @abstractmethod
    def delete(self, doc_id: str) -> None:
        pass


class ChunkRepository(ABC):

    @abstractmethod
    def save(self, chunk: Chunk, concept_tags: list[str] | None = None) -> str:
        pass

    @abstractmethod
    def get_by_document(self, document_id: str) -> list[Chunk]:
        pass

    @abstractmethod
    def delete_by_document(self, document_id: str) -> None:
        pass


class ConceptLinkRepository(ABC):

    @abstractmethod
    def save(self, link: ConceptLink) -> str:
        pass

    @abstractmethod
    def get_by_document(self, document_id: str) -> list[ConceptLink]:
        pass

    @abstractmethod
    def get_by_tag(self, tag: str) -> list[ConceptLink]:
        pass