from graphs.rag_graph import rag_graph
from routing.query_router import QueryIntent

query = "What is the role of attention in transformer models?"

print(f"\nQuery: {query}\n")
print("=" * 60)

result = rag_graph.invoke({"query": query})

intent = result.get("intent", "factual")
print(f"Intent: {intent}\n")

if intent == QueryIntent.EVOLUTION.value:
    print("\nTimeline:")
    for entry in result.get("timeline", []):
        print(f"\n  [{entry.published_at}] {entry.title}")
        print(f"  {entry.contribution}")
    print("\n" + "=" * 60)
    print("\nProgression Narrative:")
    print(result["answer"])
else:
    print(f"\nAnswer:\n{result['answer']}")

print(f"\nConfidence: {result.get('confidence_score', 0):.2f}")
print("\nSources:")
for i, source in enumerate(result.get("sources", [])[:5], 1):
    print(f"  [{i}] doc_id={source.payload.get('document_id')} "
          f"chunk={source.payload.get('chunk_index')} "
          f"score={source.score:.4f}")
