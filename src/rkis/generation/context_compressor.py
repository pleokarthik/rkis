from llm.ollama_llm import OllamaLLM

_llm = OllamaLLM(model="phi3")

_PROMPT = """Compress the following research context to its essential facts. Remove redundancy, filler, and repetition. Keep all technical claims, findings, methods, and named entities. Output only the compressed text, nothing else.

Context:
{context}

Compressed:"""


def compress_context(context: str) -> str:
    if not context.strip():
        return context

    if len(context) < 200:
        return context

    compressed = _llm.complete(_PROMPT.format(context=context))

    if len(compressed.strip()) < 30:
        return context

    if len(compressed) > len(context):
        return context

    return compressed.strip()
