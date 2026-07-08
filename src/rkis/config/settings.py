import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


def _enable_langsmith_tracing():
    api_key = os.getenv("LANGSMITH_API_KEY", "")
    if api_key:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_API_KEY", api_key)
        os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGSMITH_PROJECT", "rkis"))


_enable_langsmith_tracing()


class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "rkis")
    
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    DB_PATH: str = os.path.join(BASE_DIR, "data", "rkis.db")
    
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "gpt-4o-mini"
    
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 5
    MAX_CONTEXT_CHARS: int = 4000
    EVOLUTION_MAX_DOCUMENTS: int = 5
    
    ARXIV_MAX_RESULTS: int = 50
    ARXIV_CATEGORIES: list = ["cs.AI", "cs.LG", "cs.CL"]

    OLLAMA_BASE_URL: str = "http://localhost:11434"
settings = Settings()