import re
import logging
from llm.ollama_llm import OllamaLLM

logger = logging.getLogger(__name__)

MIN_LENGTH = 50
MAX_LENGTH = 1500

_llm = OllamaLLM(model="phi3")

_PROMPT = """Is this text a coherent, self-contained research concept? Answer ONLY "yes" or "no".

Text:
{text}"""


def _is_purely_numeric(text: str) -> bool:
    stripped = re.sub(r'[\s.,;:\-\+\(\)\[\]]+', '', text)
    return stripped.isdigit()


def validate_chunk(text: str) -> bool:
    if len(text) < MIN_LENGTH:
        logger.info("Chunk rejected: too short (%d chars)", len(text))
        return False

    if len(text) > MAX_LENGTH:
        logger.info("Chunk rejected: too long (%d chars)", len(text))
        return False

    if _is_purely_numeric(text):
        logger.info("Chunk rejected: purely numeric")
        return False

    alpha_ratio = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
    if alpha_ratio < 0.3:
        logger.info("Chunk rejected: low alpha ratio (%.2f)", alpha_ratio)
        return False

    if alpha_ratio < 0.5 or len(text) < 100:
        try:
            raw = _llm.complete(_PROMPT.format(text=text[:500])).strip().lower()
            if "yes" not in raw:
                logger.info("Chunk rejected by LLM quality check")
                return False
        except Exception:
            pass

    return True
