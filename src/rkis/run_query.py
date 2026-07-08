import sys

import gaptrace_capture

from graphs.rag_graph import rag_graph, _active_capture
from routing.query_router import QueryIntent


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python run_query.py <query>")
        sys.exit(1)
    query = " ".join(sys.argv[1:])

    print(f"\nQuery: {query}\n")
    print("=" * 60)

    cap = gaptrace_capture.start(query=query, pipeline="rkis")
    token = _active_capture.set(cap)
    try:
        result = rag_graph.invoke({"query": query})
    except Exception as e:
        try:
            cap.response(f"[ERROR: {e}]")
        except Exception:
            cap.commit()
        raise
    finally:
        _active_capture.reset(token)

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


if __name__ == "__main__":
    main()
