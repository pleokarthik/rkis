# Finding: Validator Blind to Reranker Signal — Confident Answer from Rejected Context

**Pipeline stage:** Retrieval → Rerank → Validate
**Query:** `"context length extrapolation positional encoding"` (s1r16)

## What ctx showed

- All 10 chunks: rerank scores -11.47 to -10.98 (cross-encoder unanimous rejection)
- RRF retrieval scores: flat 0.01–0.02 across all chunks (retrieval layer not differentiating)
- Token budget: 51.9% used (2126/4096)
- Window duplicates: 3 (43% of chunks)
- confidence_score returned by validate_node: 1.00

## Why not visible from response

- Response text is coherent and on-topic (positional encoding, context length)
- confidence=1.00 at the API boundary reads as high-quality answer
- No signal in the output that reranker scored all retrieved chunks below -11
- LangSmith shows a normal LLM call with normal latency — nothing anomalous

## Root cause

- `validate_answer()` measures answer-to-context textual overlap/entailment
- When the LLM ignores irrelevant context and generates from parametric memory,
  the answer IS entailed by... nothing in context — but the validator has no
  visibility into reranker scores
- `validate_answer()` operates only on `(answer, raw_context)` — rerank scores
  are not in scope, so a confidence=1.00 means "answer is consistent with
  context" not "context was relevant to the query"

## Fix: gate validate_node on minimum rerank threshold

```python
if max(rerank_scores) < RERANK_THRESHOLD:  # e.g. -5.0
    return confidence=0.0, answer="[NO RELEVANT CONTEXT FOUND]"
```

This requires passing rerank scores through `RAGState` into `validate_node`.
Add to `RAGState`: `rerank_scores: list[float]`
Set in `ctx_retrieve_rerank_node` before returning.

**Before:** confidence=1.00, coherent answer, no warning
**After:** confidence=0.00, explicit signal that retrieval failed for this query

## Before / After

### Before (fix not applied) — s1r16

- rerank scores: -11.47 to -10.98 (all chunks rejected by cross-encoder)
- confidence_score: 1.00
- answer: coherent response about positional encoding (generated from parametric memory)
- ctx signal: rerank range visible, contradiction with confidence=1.00 detectable

```
Query: context length extrapolation positional encoding
Response: Over the course of recent years leading up to April 2023, research
into context length extrapolation within large language models (LLMs) has
significantly evolved with an emphasis on improving 'positional encoding'
techniques ...

┌─────────────────────────────── Chunk Scores ────────────────────────────────┐
│ Retrieval: 0.01-0.02                                                        │
│ Rerank:    -11.47--10.98                                                    │
│ Rerank delta:  -11.3787                                                     │
│ Low-score: 100%                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
┌──────────────────────────────── Truncation ─────────────────────────────────┐
│ Truncated: 9 chunks                                                         │
│ High-score truncations: 0                                                   │
│ Severity: low                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### After (fix applied) — s1r17

- rerank scores: -11.47 to -10.98 (identical — same retrieval, same reranker)
- confidence_score: 0.00
- answer: `[NO RELEVANT CONTEXT FOUND]`
- ctx signal: retrieval failure surfaced explicitly before answer reaches caller

```
Query: context length extrapolation positional encoding
Response: [NO RELEVANT CONTEXT FOUND]

┌──────────────────────────────── Token Usage ────────────────────────────────┐
│ Total: 1626/4096 (39.7%)                                                    │
│   Chunks:   1426                                                            │
│   History:  0                                                               │
│   System:   200                                                             │
│   Headroom: 3003                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
┌─────────────────────────────── Chunk Scores ────────────────────────────────┐
│ Retrieval: 0.01-0.02                                                        │
│ Rerank:    -11.47--10.98                                                    │
│ Rerank delta:  -11.3757                                                     │
│ Low-score: 100%                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
┌───────────────────────────── Duplicate Chunks ──────────────────────────────┐
│ Path dups:     0                                                            │
│ Window dups:   2                                                            │
│ Semantic dups: (deferred)                                                   │
│ Ratio:         40%                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
┌──────────────────────────────── Truncation ─────────────────────────────────┐
│ Truncated: 5 chunks                                                         │
│ High-score truncations: 0                                                   │
│ Severity: low                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Changes made

- `RAGState` — added `rerank_scores: list[float]`
- `ctx_retrieve_rerank_node` — returns `rerank_scores` in state dict
- `ctx_validate_node` — gates on `RERANK_THRESHOLD = -5.0`; short-circuits
  with confidence=0.00 when `max(rerank_scores) < threshold`
