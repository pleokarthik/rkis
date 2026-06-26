from typing import List
import httpx
from embeddings.base import BaseEmbedder
from config.settings import settings


class OllamaEmbedder(BaseEmbedder):
    """
    Wraps Ollama local embedding endpoint.
    No API key — assumes Ollama is running locally on default port.
    """

    def __init__(self, model: str = "nomic-embed-text"):
        self.model = model
        # replace the module-level constant with:
        self.url = f"{settings.OLLAMA_BASE_URL}/api/embed"

    def embed_text(self, text: str) -> List[float]:
        response = httpx.post(
            self.url,
            json={"model": self.model, "input": text},
            timeout=120.0,
        )
        response.raise_for_status()
        """  data = response.json()
        print(f"EMBED RESPONSE: {data}")
        return data["embeddings"][0] """
        return response.json()["embeddings"][0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(text) for text in texts]