from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Document:
    source: str
    title: str
    url: str
    content: str
    published_at: datetime
    tier: int
    authors: list = field(default_factory=list)
    categories: list = field(default_factory=list)
    id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Chunk:
    document_id: str
    chunk_index: int
    content: str
    id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ConceptLink:
    document_id: str
    concept_tag: str
    verified: bool = False
    supersedes_id: Optional[str] = None
    id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)