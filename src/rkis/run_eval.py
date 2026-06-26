from evaluation.ragas_eval import run_evaluation, RESULTS_DIR

print("\nRunning RAGAS evaluation (5 golden QA pairs)...\n")
print("This will query the RAG pipeline for each sample, then score with RAGAS metrics.")
print("LLM backend: Ollama phi3 | Embeddings: nomic-embed-text\n")
print("=" * 60)

results = run_evaluation()

print("\n=== RAGAS Evaluation Summary ===\n")
for metric, score in results["summary"].items():
    print(f"  {metric}: {score:.4f}")

print(f"\nFull report: {RESULTS_DIR / 'ragas_report.json'}")
