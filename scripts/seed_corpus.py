"""
One-off script: seed the RKIS corpus with a curated set of real arxiv papers
for dogfood testing. Uses the production ingest_arxiv() path exactly as-is —
real SourceGate LLM classification, real chunking, real embedding, real
Qdrant + SQLite writes. Not a permanent module; run once and discard.

    python scripts/seed_corpus.py
"""
import os
import sys
import sqlite3

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "rkis"))

from ingestion.ingestor import ingest_arxiv, _gate, _doc_repo
from ingestion.fetcher import fetch_arxiv
from config.settings import settings

PAPERS = [
    ("1706.03762", "Attention Is All You Need"),
    ("2005.11401", "Retrieval-Augmented Generation (Lewis et al.)"),
    ("2312.10997", "RAG survey"),
    ("2211.17192", "Speculative decoding (Leviathan et al.)"),
    ("2302.01318", "Accelerating LLM decoding with speculative sampling"),
    ("2108.12409", "ALiBi -- train short, test long"),
    ("2104.09864", "RoFormer / RoPE"),
    ("2306.15595", "Position interpolation for context extension"),
]


def main() -> None:
    results = []

    for arxiv_id, label in PAPERS:
        print(f"\n[{arxiv_id}] {label}")

        try:
            document = fetch_arxiv(arxiv_id)
        except Exception as e:
            print(f"  FAILED (fetch): {type(e).__name__}: {e}")
            results.append((arxiv_id, label, "FAILED (fetch)"))
            continue

        existing = _doc_repo.get_by_url(document.url)
        if existing:
            print(f"  SKIPPED (duplicate already in DB): {document.url} -> doc_id={existing.id}")
            results.append((arxiv_id, label, "SKIPPED (duplicate)"))
            continue

        decision = None
        try:
            decision = _gate.evaluate(
                url=document.url, title=document.title, abstract=document.content
            )
            print(
                f"  SourceGate: tier={decision.tier.value} allowed={decision.allowed} "
                f"confidence={decision.confidence:.2f}"
            )
            print(f"  reason: {decision.reason}")
        except Exception as e:
            print(f"  WARNING: SourceGate reporting call failed: {type(e).__name__}: {e}")

        try:
            doc_id = ingest_arxiv(arxiv_id)
            tier_label = decision.tier.value if decision else "?"
            print(f"  INGESTED: doc_id={doc_id}")
            results.append((arxiv_id, label, f"INGESTED (tier={tier_label})"))
        except ValueError as e:
            print(f"  REJECTED: {e}")
            results.append((arxiv_id, label, "REJECTED"))
        except Exception as e:
            print(f"  FAILED (ingest): {type(e).__name__}: {e}")
            results.append((arxiv_id, label, "FAILED (ingest)"))

    print("\n=== Summary ===")
    for arxiv_id, label, status in results:
        print(f"  {arxiv_id:12s} {status:24s} {label}")

    conn = sqlite3.connect(settings.DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM documents")
    doc_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM chunks")
    chunk_count = cur.fetchone()[0]
    conn.close()

    print(f"\nFinal counts: documents={doc_count}  chunks={chunk_count}")


if __name__ == "__main__":
    main()
