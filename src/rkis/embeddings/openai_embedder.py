from typing import List
from openai import OpenAI
from src.rkis.embeddings.base import BaseEmbedder
from src.rkis.config.settings import settings


class OpenAIEmbedder(BaseEmbedder):
    """
    Wraps OpenAI text-embedding-3-small.
    Single source of truth for model name is settings.EMBEDDING_MODEL.
    """

    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.EMBEDDING_MODEL

    def embed_text(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            input=text,
            model=self.model,
        )
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        response = self.client.embeddings.create(
            input=texts,
            model=self.model,
        )
        return [item.embedding for item in response.data]