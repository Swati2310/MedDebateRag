"""
Primary dataset: MedQA-USMLE-4-options (GBaker/MedQA-USMLE-4-options)

Originally the plan used DDxPlus, but that dataset is no longer on HuggingFace Hub.
MedQA-USMLE provides 10,178 clinical vignette questions with ground-truth answers,
which serves the same purpose: testing differential diagnosis reasoning.
"""

import pandas as pd
from datasets import load_dataset


def load_ddxplus(split: str = "train") -> pd.DataFrame:
    """
    Load MedQA-USMLE as a drop-in replacement for DDxPlus.
    Returns a DataFrame with columns compatible with the rest of the pipeline.
    """
    hf_split = "test" if split == "test" else "train"
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split=hf_split)

    cases = []
    for item in dataset:
        cases.append({
            "id":           item.get("meta_info", ""),
            "age":          "",          # not always explicit — extracted by format_patient_case
            "sex":          "",
            "symptoms":     item["question"],
            "antecedents":  "",
            "pathology":    item["answer"],   # ground truth
            "differential": list(item["options"].values()) if isinstance(item["options"], dict) else [],
        })

    return pd.DataFrame(cases)


def format_patient_case(row) -> str:
    """Format a row into readable patient case text for the agents."""
    question = row["symptoms"] if isinstance(row["symptoms"], str) else str(row["symptoms"])

    # Include MCQ options so agents can reason over them
    options_text = ""
    if isinstance(row.get("differential"), list) and row["differential"]:
        opts = row["differential"]
        options_text = "\n\nAnswer options:\n" + "\n".join(
            f"  {chr(65+i)}. {opt}" for i, opt in enumerate(opts)
        )

    return f"CLINICAL VIGNETTE:\n{question}{options_text}".strip()


if __name__ == "__main__":
    df = load_ddxplus(split="test")
    print(f"Loaded {len(df)} cases")
    print(f"Unique diagnoses: {df['pathology'].nunique()}")
    print("\nSample case:")
    print(format_patient_case(df.iloc[0]))
    print(f"\nGround truth: {df.iloc[0]['pathology']}")
