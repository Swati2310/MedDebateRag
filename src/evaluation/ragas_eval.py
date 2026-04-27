"""
RAGAS evaluation for RAG faithfulness.

Measures whether the agents' arguments are grounded in the
retrieved medical literature (faithfulness) and whether relevant
documents were retrieved (context recall).
"""

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, context_recall


def build_ragas_dataset(results: list[dict]) -> Dataset:
    """
    Convert debate results into a RAGAS-compatible HuggingFace Dataset.

    Each item must have:
    - question:   patient case text
    - answer:     debate final diagnosis / argument
    - contexts:   list of retrieved document strings
    - ground_truth: correct pathology (optional, for recall)
    """
    rows = {
        "question":     [r["patient_case"]         for r in results],
        "answer":       [r["debate_diagnosis"]      for r in results],
        "contexts":     [r["retrieved_contexts"]    for r in results],
        "ground_truth": [r["ground_truth"]          for r in results],
    }
    return Dataset.from_dict(rows)


def run_ragas(results: list[dict]) -> dict:
    dataset = build_ragas_dataset(results)
    scores = evaluate(dataset, metrics=[faithfulness, context_recall])
    return scores
