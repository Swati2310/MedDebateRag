"""
Resume the 200-case experiment from where it left off.
Reads main_results_partial.csv to find the last completed case_id,
runs the remaining cases, then merges and saves main_results.csv.
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
from src.evaluation.baselines import (
    baseline_cot,
    baseline_rag_single,
    baseline_self_consistency,
    baseline_single_llm,
)
from src.rag.retriever import load_retriever

PARTIAL_PATH = "experiments/results/main_results_partial.csv"
FINAL_PATH   = "experiments/results/main_results.csv"
N_TOTAL      = 200


def resume():
    # Load what we already have
    if os.path.exists(PARTIAL_PATH):
        existing = pd.read_csv(PARTIAL_PATH)
        done_ids = set(existing["case_id"].tolist())
        results  = existing.to_dict("records")
        print(f"Resuming: {len(done_ids)} cases already done.")
    else:
        existing = pd.DataFrame()
        done_ids = set()
        results  = []
        print("Starting fresh.")

    print(f"Loading dataset...")
    test_df   = load_ddxplus(split="test")
    retriever = load_retriever()

    remaining = [(i, row) for i, row in test_df.head(N_TOTAL).iterrows()
                 if i not in done_ids]
    print(f"Cases remaining: {len(remaining)}")

    for i, row in tqdm(remaining, total=len(remaining)):
        patient_case = format_patient_case(row)
        ground_truth = row["pathology"]

        result = {
            "case_id":      i,
            "ground_truth": ground_truth,
            "patient_case": patient_case[:200],
        }

        result["b1_single"]  = baseline_single_llm(patient_case)
        result["b2_cot"]     = baseline_cot(patient_case)
        result["b3_selfcon"] = baseline_self_consistency(patient_case, n=3)
        result["b4_rag"]     = baseline_rag_single(patient_case, retriever)

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

        if len(results) % 10 == 0:
            pd.DataFrame(results).to_csv(PARTIAL_PATH, index=False)
            print(f"  Checkpoint saved ({len(results)} total cases done)")

    df = pd.DataFrame(results).sort_values("case_id").reset_index(drop=True)
    df.to_csv(FINAL_PATH, index=False)
    df.to_csv(PARTIAL_PATH, index=False)

    print(f"\n=== ACCURACY SUMMARY ({len(df)} cases) ===")
    from src.evaluation.metrics import is_correct
    for col, label in [
        ("b1_single",       "Baseline 1 — Single LLM"),
        ("b2_cot",          "Baseline 2 — Chain-of-Thought"),
        ("b3_selfcon",      "Baseline 3 — Self-Consistency"),
        ("b4_rag",          "Baseline 4 — RAG + Single LLM"),
        ("debate_diagnosis","MedDebate-RAG (full system)"),
    ]:
        acc = df.apply(lambda r: is_correct(str(r[col]), r["ground_truth"]), axis=1).mean() * 100
        print(f"  {label:40s}: {acc:.1f}%")

    esc_rate = df["escalated"].mean() * 100
    avg_pds  = df["pds_score"].mean()
    print(f"\n  Escalation rate (PDS > 0.5): {esc_rate:.1f}%")
    print(f"  Average PDS score:           {avg_pds:.3f}")
    print(f"\nResults saved → {FINAL_PATH}")
    return df


if __name__ == "__main__":
    resume()
