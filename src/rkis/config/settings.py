import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "rkis")
    
    DB_PATH: str = "data/rkis.db"
    
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "gpt-4o-mini"
    
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 5
    MAX_CONTEXT_CHARS: int = 4000
    
    ARXIV_MAX_RESULTS: int = 50
    ARXIV_CATEGORIES: list = ["cs.AI", "cs.LG", "cs.CL"]

settings = Settings()