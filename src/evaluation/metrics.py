import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def diagnosis_accuracy(df: pd.DataFrame, pred_col: str, truth_col: str = "ground_truth") -> float:
    """
    Flexible match: checks both directions —
    - ground truth substring appears in prediction, OR
    - prediction substring appears in ground truth (for MCQ-style short answers)
    Also checks key word overlap (≥50% of GT words in prediction).
    """
    def _match(r):
        gt   = str(r[truth_col]).lower().strip()
        pred = str(r[pred_col]).lower().strip()
        if gt in pred or pred in gt:
            return True
        # keyword overlap: majority of GT words must appear in prediction
        gt_words = [w for w in gt.split() if len(w) > 3]
        if gt_words and sum(w in pred for w in gt_words) / len(gt_words) >= 0.5:
            return True
        return False

    return df.apply(_match, axis=1).mean() * 100.0


def escalation_rate(df: pd.DataFrame, escalated_col: str = "escalated") -> float:
    return df[escalated_col].mean() * 100.0


def pds_auroc(df: pd.DataFrame, pds_col: str = "pds_score",
              truth_col: str = "ground_truth", pred_col: str = "debate_diagnosis") -> float:
    """
    AUROC: treat PDS as a predictor of incorrectness.
    High PDS should correlate with wrong diagnosis.
    """
    incorrect = df.apply(
        lambda r: 0 if r[truth_col].lower() in r[pred_col].lower() else 1, axis=1
    )
    return roc_auc_score(incorrect, df[pds_col])


def calibration_summary(df: pd.DataFrame) -> dict:
    """Group by PDS bucket and report accuracy in each."""
    low  = df[df["pds_score"] < 0.2]
    mid  = df[(df["pds_score"] >= 0.2) & (df["pds_score"] < 0.5)]
    high = df[df["pds_score"] >= 0.5]

    return {
        "low_pds_accuracy":  diagnosis_accuracy(low,  "debate_diagnosis") if len(low) else None,
        "mid_pds_accuracy":  diagnosis_accuracy(mid,  "debate_diagnosis") if len(mid) else None,
        "high_pds_accuracy": diagnosis_accuracy(high, "debate_diagnosis") if len(high) else None,
        "low_n":  len(low),
        "mid_n":  len(mid),
        "high_n": len(high),
    }
