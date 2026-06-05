from src.rkis.embeddings.base import BaseEmbedder
from src.rkis.embeddings.openai_embedder import OpenAIEmbedder
from src.rkis.embeddings.ollama_embedder import OllamaEmbedder


_EMBEDDERS = {
    "openai": OpenAIEmbedder,
    "ollama": OllamaEmbedder,
}


def get_embedder(provider: str = "openai") -> BaseEmbedder:
    """
    Returns an embedder instance by provider name.
    Fails loud on unknown provider.
    """
    cls = _EMBEDDERS.get(provider)
    if cls is None:
        raise ValueError(
            f"Unknown embedder provider: '{provider}'. "
            f"Valid options: {list(_EMBEDDERS.keys())}"
        )
    return cls()