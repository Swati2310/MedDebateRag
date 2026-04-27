"""
Run all 4 baselines on 200 DDxPlus test cases.

Usage:
    python -m experiments.run_baselines
"""

import os
import sys

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv

load_dotenv()

from src.data.load_ddxplus import format_patient_case, load_ddxplus
from src.evaluation.baselines import (
    baseline_cot,
    baseline_rag_single,
    baseline_self_consistency,
    baseline_single_llm,
)
from src.rag.retriever import load_retriever


def run_baselines(n_cases: int = 200):
    test_df   = load_ddxplus(split="test")
    retriever = load_retriever()
    results   = []

    for i, row in tqdm(test_df.head(n_cases).iterrows(), total=n_cases):
        patient_case = format_patient_case(row)
        ground_truth = row["pathology"]

        results.append({
            "case_id":      i,
            "ground_truth": ground_truth,
            "b1_single":    baseline_single_llm(patient_case),
            "b2_cot":       baseline_cot(patient_case),
            "b3_selfcon":   baseline_self_consistency(patient_case),
            "b4_rag":       baseline_rag_single(patient_case, retriever),
        })

    df = pd.DataFrame(results)
    os.makedirs("experiments/results", exist_ok=True)
    df.to_csv("experiments/results/baselines.csv", index=False)
    print(f"Saved → experiments/results/baselines.csv")
    return df


if __name__ == "__main__":
    run_baselines()
