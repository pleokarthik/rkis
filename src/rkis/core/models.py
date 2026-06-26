from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
from enum import Enum

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
    knowledge_type: str = "research_paper"
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
    related_doc_id: Optional[str] = None        # second endpoint
    relationship_type: str = "related"           # explicit, extensible
    id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class SearchResult:
    chunk_id: str
    score: float
    payload: Dict[str, Any]    

@dataclass
class QueryResult:
    answer: str
    sources: list[SearchResult]

class TrustTier(Enum):
    T1 = "t1"
    T2 = "t2"
    REJECTED = "rejected"

@dataclass
class GateDecision:
    source_url: str
    tier: TrustTier
    allowed: bool
    reason: str
    confidence: float

@dataclass
class DocumentContribution:
    doc_id: str
    title: str
    published_at: datetime
    contribution: str
    chunks_used: list[SearchResult]

@dataclass
class ProgressionResult:
    topic: str
    narrative: str
    timeline: list[DocumentContribution]