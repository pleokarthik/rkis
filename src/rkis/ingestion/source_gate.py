import json
from core.models import GateDecision, TrustTier
from llm.ollama_llm import OllamaLLM
import re

_llm = OllamaLLM()

_PROMPT = """You are a research source classifier. Given a source, return ONLY a JSON object.

Source URL: {url}
Title: {title}
Abstract: {abstract}

Classify into exactly one tier:
- T1: peer-reviewed paper (ArXiv, IEEE, ACM)
- T2: research blog with citations and verifiable claims
- REJECTED: opinion, tutorial, news, uncited content, or unclear

Respond with ONLY this JSON, no explanation:
{{"tier": "t1" | "t2" | "rejected", "reason": "...", "confidence": 0.0-1.0}}"""


class SourceGate:
    def evaluate(self, url: str, title: str = "", abstract: str = "") -> GateDecision:
        prompt = _PROMPT.format(url=url, title=title, abstract=abstract)
        raw = _llm.complete(prompt)
        return self._parse(url, raw)

    def _parse(self, url: str, raw: str) -> GateDecision:
        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        match = re.search(r'\{.*?\}', cleaned, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON found in SourceGate response: {cleaned}")
        data = json.loads(match.group())
        tier = TrustTier(data["tier"].lower())
        return GateDecision(
            source_url=url,
            tier=tier,
            allowed=tier != TrustTier.REJECTED,
            reason=data["reason"],
            confidence=float(data["confidence"]),
        )

if __name__ == "__main__":
    gate = SourceGate()

    decision = gate.evaluate(
        url="https://arxiv.org/abs/1706.03762",
        title="Attention Is All You Need",
        abstract="We propose a new simple network architecture, the Transformer, based solely on attention mechanisms.",
    )

    print(f"Tier      : {decision.tier.value}")
    print(f"Allowed   : {decision.allowed}")
    print(f"Confidence: {decision.confidence}")
    print(f"Reason    : {decision.reason}")

