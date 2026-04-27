"""
PDS Validation Experiment.

Hypothesis: High PDS → Lower Accuracy
            Low PDS  → Higher Accuracy

If true → PDS is a valid uncertainty metric.

Usage:
    python -m experiments.validate_pds experiments/results/main_results.csv
"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def is_correct(pred: str, truth: str) -> bool:
    pred, truth = str(pred).lower().strip(), str(truth).lower().strip()
    if truth in pred:
        return True
    pred_words  = set(pred.split())
    truth_words = set(truth.split())
    overlap = pred_words & truth_words
    return len(overlap) / max(len(truth_words), 1) >= 0.5


def accuracy(df: pd.DataFrame) -> float:
    if len(df) == 0:
        return float("nan")
    return df.apply(
        lambda r: is_correct(str(r["debate_diagnosis"]), r["ground_truth"]), axis=1
    ).mean() * 100


def validate_pds(results_df: pd.DataFrame):
    low  = results_df[results_df["pds_score"] < 0.2]
    mid  = results_df[(results_df["pds_score"] >= 0.2) & (results_df["pds_score"] < 0.5)]
    high = results_df[results_df["pds_score"] >= 0.5]

    buckets = {
        "Low PDS (<0.2)":    (low,  "green"),
        "Mid PDS (0.2-0.5)": (mid,  "orange"),
        "High PDS (>0.5)":   (high, "red"),
    }

    print("PDS Validation Results:")
    print(f"  {'Bucket':<22} {'Cases':>6}  {'Accuracy':>10}")
    print("  " + "-" * 42)
    for label, (subset, _) in buckets.items():
        a = accuracy(subset)
        a_str = f"{a:.1f}%" if not pd.isna(a) else "N/A (no cases)"
        print(f"  {label:<22} {len(subset):>6}  {a_str:>10}")

    nonempty_labels  = [k for k, (v, _) in buckets.items() if len(v) > 0]
    nonempty_accs    = [accuracy(v) for k, (v, _) in buckets.items() if len(v) > 0]
    nonempty_colors  = [c for k, (v, c) in buckets.items() if len(v) > 0]

    correct = results_df.apply(
        lambda r: is_correct(str(r["debate_diagnosis"]), r["ground_truth"]), axis=1
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(nonempty_labels, nonempty_accs, color=nonempty_colors, alpha=0.8)
    axes[0].set_ylabel("Diagnosis Accuracy (%)")
    axes[0].set_title("PDS Score Bucket vs Accuracy\n(Lower PDS = Higher Accuracy)")
    axes[0].set_ylim(0, 100)

    axes[1].scatter(results_df["pds_score"], correct.astype(int), alpha=0.3)
    axes[1].set_xlabel("PDS Score")
    axes[1].set_ylabel("Correct Diagnosis (1=Yes, 0=No)")
    axes[1].set_title("PDS vs Correctness (Raw Scatter)")

    plt.tight_layout()
    os.makedirs("experiments/results", exist_ok=True)
    plt.savefig("experiments/results/pds_validation.png", dpi=150)
    print("\nChart saved → experiments/results/pds_validation.png")

    return {k: accuracy(v) for k, (v, _) in buckets.items()}


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "experiments/results/main_results.csv"
    df = pd.read_csv(csv_path)
    validate_pds(df)
