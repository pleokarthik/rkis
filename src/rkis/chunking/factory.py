from src.rkis.chunking.base import ChunkingStrategy
from src.rkis.chunking.fixed import FixedSizeChunker
from src.rkis.chunking.sentence import SentenceWindowChunker

_STRATEGIES = {
    "fixed": FixedSizeChunker,
    "sentence": SentenceWindowChunker,
}

def get_chunker(strategy: str = "sentence", **kwargs) -> ChunkingStrategy:
    """
    Returns a ChunkingStrategy instance by name.
    Fails loud on unknown strategy — no silent fallbacks.
    """
    cls = _STRATEGIES.get(strategy)
    if cls is None:
        raise ValueError(
            f"Unknown chunking strategy: '{strategy}'. "
            f"Valid options: {list(_STRATEGIES.keys())}"
        )
    return cls(**kwargs)