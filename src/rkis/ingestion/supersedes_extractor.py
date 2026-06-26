import json
import re
from dataclasses import dataclass
from typing import Optional
from llm.ollama_llm import OllamaLLM

_llm = OllamaLLM(model="phi3")

_PROMPT = """Does this paper claim to supersede, improve upon, or replace prior work?

Title: {title}
Abstract: {abstract}

If yes, respond with ONLY this JSON (no explanation):
{{"supersedes": true, "prior_title": "title of the prior work", "prior_arxiv_id": "arxiv id if mentioned, else null"}}

If no, respond with ONLY:
{{"supersedes": false}}"""


@dataclass
class SupersedesClaim:
    supersedes: bool
    prior_title: Optional[str] = None
    prior_arxiv_id: Optional[str] = None


def extract_supersedes(title: str, abstract: str) -> SupersedesClaim:
    prompt = _PROMPT.format(title=title, abstract=abstract[:2000])
    raw = _llm.complete(prompt).strip()

    cleaned = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    match = re.search(r'\{.*?\}', cleaned, re.DOTALL)
    if not match:
        return SupersedesClaim(supersedes=False)

    try:
        data = json.loads(match.group())
        if not data.get("supersedes"):
            return SupersedesClaim(supersedes=False)
        return SupersedesClaim(
            supersedes=True,
            prior_title=data.get("prior_title"),
            prior_arxiv_id=data.get("prior_arxiv_id"),
        )
    except (json.JSONDecodeError, TypeError):
        return SupersedesClaim(supersedes=False)
