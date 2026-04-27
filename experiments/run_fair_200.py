"""
Full 200-case fair evaluation.
All systems (baselines + MedDebate-RAG) must select from MCQ options.
Saves checkpoint every 10 cases. Safe to resume if interrupted.
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
    baseline_cot, baseline_rag_single,
    baseline_self_consistency, baseline_single_llm,
)
from src.rag.retriever import load_retriever

PARTIAL = "experiments/results/fair_200_partial.csv"
FINAL   = "experiments/results/fair_200_final.csv"
N       = 200


def is_correct(pred, truth):
    pred, truth = str(pred).lower().strip(), str(truth).lower().strip()
    if truth in pred: return True
    pw, tw = set(pred.split()), set(truth.split())
    return len(pw & tw) / max(len(tw), 1) >= 0.5


def run():
    # Resume from checkpoint if exists
    if os.path.exists(PARTIAL):
        existing = pd.read_csv(PARTIAL)
        done_ids = set(existing["case_id"].tolist())
        results  = existing.to_dict("records")
        print(f"Resuming from checkpoint: {len(done_ids)} cases done.")
    else:
        done_ids, results = set(), []
        print("Starting fresh.")

    df       = load_ddxplus(split="test")
    retriever = load_retriever()

    remaining = [(i, row) for i, row in df.head(N).iterrows() if i not in done_ids]
    print(f"Cases remaining: {len(remaining)}")

    for i, row in tqdm(remaining, total=len(remaining)):
        case  = format_patient_case(row)
        truth = row["pathology"]

        r = {"case_id": i, "ground_truth": truth}
        r["b1"]      = baseline_single_llm(case)
        r["b2"]      = baseline_cot(case)
        r["b3"]      = baseline_self_consistency(case, n=3)
        r["b4"]      = baseline_rag_single(case, retriever)

        state = run_debate(case, truth, max_rounds=3)
        r["debate"]         = state["final_diagnosis"]
        r["pds"]            = state["position_drift_score"]
        r["escalated"]      = state["escalate_to_human"]
        r["option_ranking"] = str(state.get("option_ranking", []))
        r["a_letters"]      = str(state.get("doctor_a_letters", []))
        r["b_letters"]      = str(state.get("doctor_b_letters", []))

        results.append(r)

        if len(results) % 10 == 0:
            pd.DataFrame(results).to_csv(PARTIAL, index=False)
            print(f"  Checkpoint saved ({len(results)} total done)")

    out = pd.DataFrame(results).sort_values("case_id").reset_index(drop=True)
    out.to_csv(FINAL,   index=False)
    out.to_csv(PARTIAL, index=False)

    print(f"\n{'='*55}")
    print(f"FINAL FAIR EVALUATION — {len(out)} cases")
    print(f"{'='*55}")
    for col, label in [
        ("b1",    "Baseline 1 — Single LLM          "),
        ("b2",    "Baseline 2 — Chain-of-Thought     "),
        ("b3",    "Baseline 3 — Self-Consistency     "),
        ("b4",    "Baseline 4 — RAG + LLM            "),
        ("debate","MedDebate-RAG (with screener)     "),
    ]:
        acc = out.apply(lambda r: is_correct(str(r[col]), r["ground_truth"]), axis=1).mean() * 100
        print(f"  {label}: {acc:.1f}%")

    print(f"\n  Escalation rate (PDS>0.3): {(out['pds']>0.3).mean()*100:.1f}%")
    print(f"  Average PDS:               {out['pds'].mean():.3f}")
    print(f"\nResults saved → {FINAL}")


if __name__ == "__main__":
    run()
