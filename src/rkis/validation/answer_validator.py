import re
from llm.ollama_llm import OllamaLLM

_llm = OllamaLLM(model="phi3")

CONFIDENCE_THRESHOLD = 0.6

_PROMPT = """Is this answer fully supported by the provided context? Score from 0.0 to 1.0.
- 1.0 = every claim in the answer is directly stated in the context
- 0.5 = some claims are supported, others are inferred or missing
- 0.0 = the answer contradicts or is unrelated to the context

Context:
{context}

Answer:
{answer}

Respond with ONLY a number between 0.0 and 1.0:"""


def validate_answer(answer: str, context: str) -> float:
    if not answer.strip() or not context.strip():
        return 0.0

    raw = _llm.complete(_PROMPT.format(context=context, answer=answer)).strip()

    match = re.search(r'(\d+\.?\d*)', raw)
    if not match:
        return 0.5

    score = float(match.group(1))
    return max(0.0, min(1.0, score))
