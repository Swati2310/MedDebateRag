"""
Quick CLI tester — run a single patient case through the full debate system.

Usage:
    python test_case.py                        # prompts for input
    python test_case.py "65yo male, chest pain..."
"""

import sys
import os
from dotenv import load_dotenv
load_dotenv()

from src.debate.orchestrator import run_debate

def run(case: str):
    print("\n" + "="*60)
    print("RUNNING DEBATE...")
    print("="*60)

    state = run_debate(case, ground_truth="", max_rounds=3)

    print("\n--- DOCTOR A (Advocate) ---")
    for i, arg in enumerate(state["doctor_a_arguments"]):
        print(f"Round {i+1}: {state['doctor_a_diagnoses'][i]} ({state['doctor_a_confidences'][i]:.0f}% conf)")
        print(f"  {arg[:200]}...")

    print("\n--- DOCTOR B (Devil's Advocate) ---")
    for i, arg in enumerate(state["doctor_b_arguments"]):
        print(f"Round {i+1}: {state['doctor_b_diagnoses'][i]} ({state['doctor_b_confidences'][i]:.0f}% conf)")
        print(f"  {arg[:200]}...")

    print("\n--- MODERATOR VERDICT ---")
    print(state.get("moderator_verdict", ""))

    print("\n--- FINAL RESULT ---")
    print(f"Diagnosis : {state['final_diagnosis']}")
    print(f"PDS Score : {state['position_drift_score']:.3f}  (threshold: 0.3)")
    print(f"  Confidence Drift  : {state['pds_components']['confidence_drift']:.3f}")
    print(f"  Semantic Drift    : {state['pds_components']['semantic_drift']:.3f}")
    print(f"  Final Disagreement: {state['pds_components']['final_disagreement']:.3f}")

    if state["escalate_to_human"]:
        print("\n⚠  ESCALATED TO HUMAN DOCTOR (PDS too high)")
        print(state.get("escalation_reason", ""))
    else:
        print("\n✓  Auto-diagnosis (PDS within safe range)")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        case = " ".join(sys.argv[1:])
    else:
        print("Enter patient case (press Enter twice when done):")
        lines = []
        while True:
            line = input()
            if line == "" and lines:
                break
            lines.append(line)
        case = "\n".join(lines)

    run(case)
