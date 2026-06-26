import re
from enum import Enum
from llm.ollama_llm import OllamaLLM


class QueryIntent(Enum):
    FACTUAL = "factual"
    EVOLUTION = "evolution"


_EVOLUTION_PATTERNS = [
    r"\bevol\w*",
    r"\bprogress\w*",
    r"\btimeline\b",
    r"\bchronolog\w*",
    r"\bhow\s+did\b.*\b(change|develop|advance|grow|emerge)",
    r"\bhistory\s+of\b",
    r"\bover\s+time\b",
    r"\btrend\b",
    r"\btrajectory\b",
    r"\bshift\w*\b.*\bfrom\b",
]

_FACTUAL_PATTERNS = [
    r"\bwhat\s+is\b",
    r"\bdefine\b",
    r"\bexplain\b",
    r"\bwhat\s+are\s+the\s+key\b",
    r"\bhow\s+does\b",
    r"\brole\s+of\b",
    r"\bfinding\w*\b",
    r"\bresult\w*\b",
    r"\bmethod\w*\b",
]

_LLM_PROMPT = """Classify this research query into exactly one intent.

Query: {query}

Intents:
- factual: asking about a specific concept, method, finding, or paper
- evolution: asking how a topic changed, progressed, or developed over time

Respond with ONLY one word: factual or evolution"""


def _rule_based_classify(query: str) -> QueryIntent | None:
    q = query.lower()
    evo_score = sum(1 for p in _EVOLUTION_PATTERNS if re.search(p, q))
    fact_score = sum(1 for p in _FACTUAL_PATTERNS if re.search(p, q))

    if evo_score > 0 and evo_score > fact_score:
        return QueryIntent.EVOLUTION
    if fact_score > 0 and fact_score > evo_score:
        return QueryIntent.FACTUAL
    return None


class QueryRouter:
    def __init__(self):
        self._llm = OllamaLLM(model="phi3")

    def classify(self, query: str) -> QueryIntent:
        result = _rule_based_classify(query)
        if result is not None:
            return result
        return self._llm_classify(query)

    def _llm_classify(self, query: str) -> QueryIntent:
        prompt = _LLM_PROMPT.format(query=query)
        raw = self._llm.complete(prompt).strip().lower()
        if "evolution" in raw:
            return QueryIntent.EVOLUTION
        return QueryIntent.FACTUAL
