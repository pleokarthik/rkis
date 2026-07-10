"""
generate_evolution_node fans out 2 sequential LLM calls (compress_context +
contribution summary) per distinct document in state["sources"]. This test
asserts that fan-out is capped at settings.EVOLUTION_MAX_DOCUMENTS even when
sources span more documents than that.

Run:  python -m unittest tests.test_generate_evolution_cap -v
(from repo root, with the project venv active)
"""
import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "rkis"))

# graphs.rag_graph instantiates real singletons at import time (embedded
# Qdrant client, cross-encoder reranker, BM25 index, query router,
# semantic cache). None of those are exercised by generate_evolution_node,
# so they're stubbed out before import to keep this test hermetic and fast.
# QdrantVectorStore's __init__ needs a fake .client (not just a no-op) since
# SemanticCache(_vector_store.client) reads that attribute at module-import
# time, before SemanticCache's own (also patched) __init__ ever runs.
def _fake_qdrant_vector_store_init(self):
    self.client = MagicMock()


with patch("vectorstores.qdrant_store.QdrantVectorStore.__init__", _fake_qdrant_vector_store_init), \
     patch("vectorstores.semantic_cache.SemanticCache.__init__", return_value=None), \
     patch("query.reranker.get_reranker", return_value=MagicMock()), \
     patch("retrieval.bm25_retriever.BM25Retriever.__init__", return_value=None), \
     patch("routing.query_router.QueryRouter.__init__", return_value=None):
    import graphs.rag_graph  # noqa: F401 -- side effect: populates sys.modules

# graphs/__init__.py does `from graphs.rag_graph import rag_graph`, which
# shadows the `graphs.rag_graph` attribute with the *compiled graph object*
# (name collision: the submodule and its exported compiled graph are both
# called "rag_graph"). Fetch the real module via sys.modules to sidestep that.
rag_graph_module = sys.modules["graphs.rag_graph"]

from core.models import Document, SearchResult
from config.settings import settings


class GenerateEvolutionNodeCapTest(unittest.TestCase):
    def test_caps_per_document_fanout_at_evolution_max_documents(self):
        num_docs = settings.EVOLUTION_MAX_DOCUMENTS + 3
        base_date = datetime(2020, 1, 1)

        docs_by_id = {}
        sources = []
        for i in range(num_docs):
            doc_id = f"doc-{i}"
            docs_by_id[doc_id] = Document(
                source="arxiv",
                title=f"Paper {i}",
                url=f"http://arxiv.org/abs/{i}",
                content="",
                published_at=base_date + timedelta(days=i),
                tier=1,
                id=doc_id,
            )
            sources.append(SearchResult(
                chunk_id=f"chunk-{i}",
                score=1.0,
                payload={"document_id": doc_id, "content": f"content {i}"},
            ))

        state = {"query": "how did positional encoding evolve", "sources": sources}

        with patch.object(rag_graph_module, "_doc_repo") as mock_doc_repo, \
             patch.object(rag_graph_module, "_llm") as mock_llm, \
             patch.object(rag_graph_module, "compress_context") as mock_compress:

            mock_doc_repo.get_by_id.side_effect = lambda doc_id: docs_by_id[doc_id]
            mock_compress.side_effect = lambda ctx: ctx
            mock_llm.complete.return_value = "a contribution"

            result = rag_graph_module.generate_evolution_node(state)

        timeline = result["timeline"]

        self.assertLessEqual(len(timeline), settings.EVOLUTION_MAX_DOCUMENTS)
        self.assertEqual(len(timeline), settings.EVOLUTION_MAX_DOCUMENTS)

        # earliest-N by published_at, not just any N
        expected_ids = [f"doc-{i}" for i in range(settings.EVOLUTION_MAX_DOCUMENTS)]
        self.assertEqual([c.doc_id for c in timeline], expected_ids)

        # 2 LLM-backed calls per surviving document (compress + contribution),
        # plus 1 final narrative call
        self.assertEqual(mock_compress.call_count, settings.EVOLUTION_MAX_DOCUMENTS)
        self.assertEqual(mock_llm.complete.call_count, settings.EVOLUTION_MAX_DOCUMENTS + 1)


if __name__ == "__main__":
    unittest.main()
