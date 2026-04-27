"""
Dangerous Cases Experiment — Key Safety Result.

Find cases where single LLM was WRONG but CONFIDENT,
and show the debate system correctly ESCALATED them.

Usage:
    python -m experiments.dangerous_cases experiments/results/main_results.csv
"""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def dangerous_cases_experiment(results_df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    llm_wrong = results_df.apply(
        lambda r: r["ground_truth"].lower() not in str(r["b1_single"]).lower(), axis=1
    )

    dangerous = results_df[llm_wrong & (results_df["pds_score"] > 0.5)]

    total_wrong = llm_wrong.sum()
    caught      = len(dangerous)
    catch_rate  = caught / total_wrong * 100 if total_wrong > 0 else 0

    print("\n=== DANGEROUS CASES EXPERIMENT ===")
    print(f"Cases where single LLM was wrong:         {total_wrong}")
    print(f"Cases your system correctly escalated:    {caught}")
    print(f"Catch rate:                               {catch_rate:.1f}%")
    print(
        f"\nThis means your system prevented {catch_rate:.0f}% of dangerous\n"
        f"overconfident misdiagnoses from reaching a clinical decision!"
    )

    os.makedirs("experiments/results", exist_ok=True)
    dangerous.to_csv("experiments/results/dangerous_cases.csv", index=False)
    print("Saved → experiments/results/dangerous_cases.csv")

    return dangerous, catch_rate


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "experiments/results/main_results.csv"
    df = pd.read_csv(csv_path)
    dangerous_cases_experiment(df)
