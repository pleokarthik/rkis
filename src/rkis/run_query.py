from routing.query_router import QueryRouter, QueryIntent
from query.pipeline import QueryPipeline
from query.progression import ProgressionPipeline

router = QueryRouter()

query = "What is the role of attention in transformer models?"

print(f"\nQuery: {query}\n")
print("=" * 60)

intent = router.classify(query)
print(f"Intent: {intent.value}\n")

if intent == QueryIntent.EVOLUTION:
    pipeline = ProgressionPipeline()
    result = pipeline.run(query)

    print("\nTimeline:")
    for entry in result.timeline:
        print(f"\n  [{entry.published_at}] {entry.title}")
        print(f"  {entry.contribution}")
    print("\n" + "=" * 60)
    print("\nProgression Narrative:")
    print(result.narrative)
else:
    pipeline = QueryPipeline()
    result = pipeline.run(query)

    print(f"\nAnswer:\n{result.answer}")
    print("\nSources:")
    for i, source in enumerate(result.sources, 1):
        print(f"  [{i}] doc_id={source.payload.get('document_id')} "
              f"chunk={source.payload.get('chunk_index')} "
              f"score={source.score:.4f}")
