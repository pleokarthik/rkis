from src.rkis.vectorstores.base import BaseVectorStore
from src.rkis.vectorstores.qdrant_store import QdrantVectorStore


_STORES = {
    "qdrant": QdrantVectorStore,
}


def get_vector_store(provider: str = "qdrant") -> BaseVectorStore:
    """
    Returns a vector store instance by provider name.
    Pinecone and Weaviate added here when evaluated.
    Fails loud on unknown provider.
    """
    cls = _STORES.get(provider)
    if cls is None:
        raise ValueError(
            f"Unknown vector store provider: '{provider}'. "
            f"Valid options: {list(_STORES.keys())}"
        )
    return cls()