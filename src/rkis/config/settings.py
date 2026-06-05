import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

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
    
    ARXIV_MAX_RESULTS: int = 50
    ARXIV_CATEGORIES: list = ["cs.AI", "cs.LG", "cs.CL"]

    OLLAMA_BASE_URL: str = "http://localhost:11434"
settings = Settings()