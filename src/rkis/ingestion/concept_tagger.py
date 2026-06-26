import json
import re
from llm.ollama_llm import OllamaLLM

_llm = OllamaLLM(model="phi3")

_PROMPT = """Extract 3-5 concept tags from this research text chunk. Tags should be concise technical terms (1-3 words each).

Text:
{text}

Respond with ONLY a JSON array of strings, no explanation:
["tag1", "tag2", "tag3"]"""


def extract_concept_tags(text: str) -> list[str]:
    prompt = _PROMPT.format(text=text[:1500])
    raw = _llm.complete(prompt).strip()

    cleaned = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    match = re.search(r'\[.*?\]', cleaned, re.DOTALL)
    if not match:
        return []

    try:
        tags = json.loads(match.group())
        return [str(t).strip().lower() for t in tags if isinstance(t, str) and t.strip()][:5]
    except (json.JSONDecodeError, TypeError):
        return []
