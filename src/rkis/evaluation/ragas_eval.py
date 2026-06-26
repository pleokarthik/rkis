import json
import os
from pathlib import Path

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

from graphs.rag_graph import rag_graph

RESULTS_DIR = Path(__file__).parent / "results"

GOLDEN_QA = [
    {
        "query": "What is the Transformer architecture?",
        "reference": "The Transformer is a neural network architecture based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. It was introduced in the paper 'Attention Is All You Need'.",
    },
    {
        "query": "What is the role of self-attention in transformers?",
        "reference": "Self-attention allows the model to attend to all positions in the previous layer of the encoder or decoder, computing a weighted sum of value vectors based on query-key compatibility.",
    },
    {
        "query": "What are the key findings on retrieval-augmented generation?",
        "reference": "Retrieval-augmented generation combines a pre-trained parametric model with a non-parametric retrieval component, allowing the model to access external knowledge and generate more factual, grounded responses.",
    },
    {
        "query": "How do large language models use reinforcement learning from human feedback?",
        "reference": "RLHF fine-tunes language models using human preference data. A reward model is trained on human comparisons, then the language model is optimized via proximal policy optimization to maximize the reward signal.",
    },
    {
        "query": "What problems does multi-head attention solve?",
        "reference": "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions, which a single attention head would inhibit.",
    },
]


def build_dataset() -> EvaluationDataset:
    samples = []
    for qa in GOLDEN_QA:
        result = rag_graph.invoke({"query": qa["query"]})

        contexts = [
            s.payload.get("content", "")
            for s in result.get("sources", [])[:5]
        ]

        samples.append(SingleTurnSample(
            user_input=qa["query"],
            response=result.get("answer", ""),
            retrieved_contexts=contexts,
            reference=qa["reference"],
        ))

    return EvaluationDataset(samples=samples)


def run_evaluation() -> dict:
    ollama_llm = LangchainLLMWrapper(Ollama(model="phi3", base_url="http://localhost:11434"))
    ollama_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
    )

    dataset = build_dataset()

    metrics = [
        Faithfulness(llm=ollama_llm),
        AnswerRelevancy(llm=ollama_llm, embeddings=ollama_embeddings),
        ContextPrecision(llm=ollama_llm),
        ContextRecall(llm=ollama_llm),
    ]

    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        raise_exceptions=False,
    )

    report = results.to_pandas().to_dict(orient="records")
    summary = {
        col: float(results.to_pandas()[col].mean())
        for col in results.to_pandas().select_dtypes(include="number").columns
    }

    output = {"summary": summary, "per_sample": report}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = RESULTS_DIR / "ragas_report.json"
    with open(report_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    return output


if __name__ == "__main__":
    results = run_evaluation()
    print("\n=== RAGAS Evaluation Summary ===\n")
    for metric, score in results["summary"].items():
        print(f"  {metric}: {score:.4f}")
    print(f"\nFull report saved to: {RESULTS_DIR / 'ragas_report.json'}")
