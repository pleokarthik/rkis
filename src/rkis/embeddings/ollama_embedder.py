from typing import List
import httpx
from src.rkis.embeddings.base import BaseEmbedder
from src.rkis.config.settings import settings


class OllamaEmbedder(BaseEmbedder):
    """
    Wraps Ollama local embedding endpoint.
    No API key — assumes Ollama is running locally on default port.
    """

    def __init__(self, model: str = "nomic-embed-text"):
        self.model = model
        # replace the module-level constant with:
        self.url = f"{settings.OLLAMA_BASE_URL}/api/embeddings"

    def embed_text(self, text: str) -> List[float]:
        response = httpx.post(
            self.url,
            json={"model": self.model, "prompt": text},
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(text) for text in texts]