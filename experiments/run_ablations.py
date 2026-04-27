"""
Ablation studies:

A1: No RAG          — debate only (no retrieved docs)
A2: No PDS/HITL     — always give diagnosis, never escalate
A3: 1 round debate
A4: 3 round debate  — standard (main system)
A5: 5 round debate
A6: No debate       — single agent + moderator only

Usage:
    python -m experiments.run_ablations
"""

import os
import sys

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv

load_dotenv()

from src.data.load_ddxplus import format_patient_case, load_ddxplus
from src.debate.orchestrator import run_debate

ABLATION_CASES = 100


def ablation_no_rag(test_df: pd.DataFrame) -> pd.DataFrame:
    """A1: Disable RAG — pass empty retrieved docs."""
    results = []
    for _, row in tqdm(test_df.head(ABLATION_CASES).iterrows(), desc="A1 No-RAG"):
        patient_case = format_patient_case(row)
        # Monkey-patch: override retriever to return empty string
        import src.debate.orchestrator as orch
        original = orch._get_retriever

        class _EmptyRetriever:
            def format_for_prompt(self, _):
                return "No evidence retrieved (ablation A1)."

        orch._retriever = _EmptyRetriever()
        state = run_debate(patient_case, row["pathology"], max_rounds=3)
        orch._retriever = None  # reset

        results.append({
            "ground_truth": row["pathology"],
            "diagnosis":    state["final_diagnosis"],
            "pds":          state["position_drift_score"],
            "ablation":     "no_rag",
        })
    return pd.DataFrame(results)


def ablation_rounds(test_df: pd.DataFrame, rounds: list[int] = [1, 2, 3, 5]) -> dict[str, pd.DataFrame]:
    """A3-A5: Different debate lengths."""
    all_results = {}
    for r in rounds:
        results = []
        for _, row in tqdm(test_df.head(ABLATION_CASES).iterrows(), desc=f"Rounds={r}"):
            state = run_debate(format_patient_case(row), row["pathology"], max_rounds=r)
            results.append({
                "ground_truth": row["pathology"],
                "diagnosis":    state["final_diagnosis"],
                "pds":          state["position_drift_score"],
                "rounds":       r,
            })
        all_results[f"rounds_{r}"] = pd.DataFrame(results)
    return all_results


if __name__ == "__main__":
    os.makedirs("experiments/results", exist_ok=True)
    test_df = load_ddxplus(split="test")

    print("Running A1: No RAG...")
    df_no_rag = ablation_no_rag(test_df)
    df_no_rag.to_csv("experiments/results/ablation_no_rag.csv", index=False)

    print("Running A3-A5: Round ablations...")
    round_results = ablation_rounds(test_df, rounds=[1, 3, 5])
    for name, df in round_results.items():
        df.to_csv(f"experiments/results/ablation_{name}.csv", index=False)

    print("Done. Results in experiments/results/")
