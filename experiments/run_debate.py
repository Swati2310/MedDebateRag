"""
Main experiment: run full debate system + all 4 baselines on N MedQA test cases.

Usage:
    python -m experiments.run_debate [--n 200]
"""

import argparse
import os
import sys

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

from src.data.load_ddxplus import format_patient_case, load_ddxplus
from src.debate.orchestrator import run_debate
from src.evaluation.baselines import (
    baseline_cot,
    baseline_rag_single,
    baseline_self_consistency,
    baseline_single_llm,
)
from src.rag.retriever import load_retriever


def run_all_experiments(n_cases: int = 200) -> pd.DataFrame:
    print(f"Loading dataset ({n_cases} cases)...")
    test_df   = load_ddxplus(split="test")
    retriever = load_retriever()
    results   = []

    for i, row in tqdm(test_df.head(n_cases).iterrows(), total=n_cases):
        patient_case = format_patient_case(row)
        ground_truth = row["pathology"]

        result = {
            "case_id":      i,
            "ground_truth": ground_truth,
            "patient_case": patient_case[:200],
        }

        # ── Baselines ────────────────────────────────────────────────────
        result["b1_single"]  = baseline_single_llm(patient_case)
        result["b2_cot"]     = baseline_cot(patient_case)
        result["b3_selfcon"] = baseline_self_consistency(patient_case, n=3)   # 3 samples
        result["b4_rag"]     = baseline_rag_single(patient_case, retriever)

        # ── Full Debate System ────────────────────────────────────────────
        final_state = run_debate(patient_case, ground_truth, max_rounds=3)
        result["debate_diagnosis"] = final_state["final_diagnosis"]
        result["pds_score"]        = final_state["position_drift_score"]
        result["escalated"]        = final_state["escalate_to_human"]
        result["pds_conf_drift"]   = final_state["pds_components"]["confidence_drift"]
        result["pds_sem_drift"]    = final_state["pds_components"]["semantic_drift"]
        result["pds_disagreement"] = final_state["pds_components"]["final_disagreement"]
        result["a_confidences"]    = str(final_state["doctor_a_confidences"])
        result["b_confidences"]    = str(final_state["doctor_b_confidences"])
        result["a_diagnoses"]      = str(final_state["doctor_a_diagnoses"])
        result["b_diagnoses"]      = str(final_state["doctor_b_diagnoses"])
        result["moderator_verdict"]= final_state["moderator_verdict"]

        results.append(result)

        # Save incrementally every 10 cases
        if len(results) % 10 == 0:
            pd.DataFrame(results).to_csv(
                "experiments/results/main_results_partial.csv", index=False
            )

    df = pd.DataFrame(results)
    os.makedirs("experiments/results", exist_ok=True)
    df.to_csv("experiments/results/main_results.csv", index=False)

    # ── Print accuracy summary ────────────────────────────────────────────
    print("\n=== ACCURACY SUMMARY ===")
    for col, label in [
        ("b1_single",       "Baseline 1 — Single LLM"),
        ("b2_cot",          "Baseline 2 — Chain-of-Thought"),
        ("b3_selfcon",      "Baseline 3 — Self-Consistency"),
        ("b4_rag",          "Baseline 4 — RAG + Single LLM"),
        ("debate_diagnosis","MedDebate-RAG (full system)"),
    ]:
        acc = df.apply(
            lambda r: r["ground_truth"].lower() in str(r[col]).lower(), axis=1
        ).mean() * 100
        print(f"  {label:40s}: {acc:.1f}%")

    esc_rate = df["escalated"].mean() * 100
    print(f"\n  Escalation rate (PDS > 0.5): {esc_rate:.1f}%")
    print(f"\nResults saved → experiments/results/main_results.csv")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="Number of cases to run")
    args = parser.parse_args()
    run_all_experiments(n_cases=args.n)
