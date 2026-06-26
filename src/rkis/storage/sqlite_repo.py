import os
import sqlite3
import uuid
import json
from typing import Optional
from core.models import Document, Chunk, ConceptLink
from storage.repository import (
    DocumentRepository,
    ChunkRepository,
    ConceptLinkRepository
)
from config.settings import settings

def get_connection():
    os.makedirs(os.path.dirname(settings.DB_PATH), exist_ok=True)
    conn = sqlite3.connect(settings.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            title TEXT NOT NULL,
            url TEXT UNIQUE NOT NULL,
            authors TEXT,
            categories TEXT,
            published_at TIMESTAMP,
            tier INTEGER NOT NULL,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            concept_tags TEXT DEFAULT '[]',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
    """)

    try:
        cursor.execute("ALTER TABLE chunks ADD COLUMN concept_tags TEXT DEFAULT '[]'")
    except sqlite3.OperationalError:
        pass

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS concept_links (
            id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            concept_tag TEXT NOT NULL,
            supersedes_id TEXT,
            verified BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents(id),
            FOREIGN KEY (supersedes_id) REFERENCES documents(id)
        )
    """)

    conn.commit()
    conn.close()

class SQLiteDocumentRepository(DocumentRepository):

    def save(self, document: Document) -> str:
        doc_id = document.id or str(uuid.uuid4())
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO documents
            (id, source, title, url, authors, categories, published_at, tier, content)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            doc_id,
            document.source,
            document.title,
            document.url,
            json.dumps(document.authors),
            json.dumps(document.categories),
            document.published_at,
            document.tier,
            document.content
        ))
        conn.commit()
        conn.close()
        return doc_id

    def get_by_id(self, doc_id: str) -> Optional[Document]:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        return Document(
            id=row["id"],
            source=row["source"],
            title=row["title"],
            url=row["url"],
            authors=json.loads(row["authors"] or "[]"),
            categories=json.loads(row["categories"] or "[]"),
            published_at=row["published_at"],
            tier=row["tier"],
            content=row["content"]
        )

    def get_by_url(self, url: str) -> Optional[Document]:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE url = ?", (url,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        return Document(
            id=row["id"],
            source=row["source"],
            title=row["title"],
            url=row["url"],
            authors=json.loads(row["authors"] or "[]"),
            categories=json.loads(row["categories"] or "[]"),
            published_at=row["published_at"],
            tier=row["tier"],
            content=row["content"]
        )

    def exists(self, url: str) -> bool:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM documents WHERE url = ?", (url,))
        row = cursor.fetchone()
        conn.close()
        return row is not None

    def delete(self, doc_id: str) -> None:
        conn = get_connection()
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
        conn.close()

class SQLiteChunkRepository(ChunkRepository):

    def save(self, chunk: Chunk, concept_tags: list[str] | None = None) -> str:
        chunk_id = chunk.id or str(uuid.uuid4())
        tags_json = json.dumps(concept_tags or [])
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO chunks
            (id, document_id, chunk_index, content, concept_tags)
            VALUES (?, ?, ?, ?, ?)
        """, (
            chunk_id,
            chunk.document_id,
            chunk.chunk_index,
            chunk.content,
            tags_json,
        ))
        conn.commit()
        conn.close()
        return chunk_id

    def get_by_document(self, document_id: str) -> list[Chunk]:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM chunks
            WHERE document_id = ?
            ORDER BY chunk_index
        """, (document_id,))
        rows = cursor.fetchall()
        conn.close()
        return [
            Chunk(
                id=row["id"],
                document_id=row["document_id"],
                chunk_index=row["chunk_index"],
                content=row["content"]
            )
            for row in rows
        ]

    def delete_by_document(self, document_id: str) -> None:
        conn = get_connection()
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        conn.commit()
        conn.close()


class SQLiteConceptLinkRepository(ConceptLinkRepository):

    def save(self, link: ConceptLink) -> str:
        link_id = link.id or str(uuid.uuid4())
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR IGNORE INTO concept_links
            (id, document_id, concept_tag, supersedes_id, verified)
            VALUES (?, ?, ?, ?, ?)
        """, (
            link_id,
            link.document_id,
            link.concept_tag,
            link.supersedes_id,
            link.verified
        ))
        conn.commit()
        conn.close()
        return link_id

    def get_by_document(self, document_id: str) -> list[ConceptLink]:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM concept_links
            WHERE document_id = ?
        """, (document_id,))
        rows = cursor.fetchall()
        conn.close()
        return [
            ConceptLink(
                id=row["id"],
                document_id=row["document_id"],
                concept_tag=row["concept_tag"],
                supersedes_id=row["supersedes_id"],
                verified=bool(row["verified"])
            )
            for row in rows
        ]

    def get_by_tag(self, tag: str) -> list[ConceptLink]:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM concept_links
            WHERE concept_tag = ?
        """, (tag,))
        rows = cursor.fetchall()
        conn.close()
        return [
            ConceptLink(
                id=row["id"],
                document_id=row["document_id"],
                concept_tag=row["concept_tag"],
                supersedes_id=row["supersedes_id"],
                verified=bool(row["verified"])
            )
            for row in rows
        ]