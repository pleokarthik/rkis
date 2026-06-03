import sqlite3
import uuid
import json
from datetime import datetime
from typing import Optional
from src.rkis.core.models import Document, Chunk, ConceptLink
from src.rkis.storage.repository import (
    DocumentRepository,
    ChunkRepository,
    ConceptLinkRepository
)
from src.rkis.config.settings import settings

def get_connection():
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
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents(id)
        )
    """)

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