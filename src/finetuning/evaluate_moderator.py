"""
Compare fine-tuned Llama moderator vs GPT-4o moderator verdict quality.

Metrics:
- Format compliance rate (does output follow the required template?)
- Winner accuracy vs GPT-4o labels
- BLEU / ROUGE on REASONING field
"""

import json
import re

import pandas as pd
from tqdm import tqdm

from src.agents.doctor_a import parse_agent_response
from src.agents.moderator import run_moderator


def _extract_winner(text: str) -> str:
    m = re.search(r"WINNER:\s*(.+?)(?=\n|\Z)", text, re.IGNORECASE)
    return m.group(1).strip() if m else ""


def evaluate_moderators(test_transcripts: list[dict], n: int = 100) -> pd.DataFrame:
    """
    Run both GPT-4o and fine-tuned moderator on the same transcripts
    and compare winner labels.
    """
    records = []

    for item in tqdm(test_transcripts[:n]):
        # GPT-4o verdict (gold)
        gpt_result = run_moderator(
            patient_case=item["patient_case"],
            full_transcript=item["debate_transcript"],
            state=item["state"],
            use_finetuned=False,
        )

        # Fine-tuned verdict
        ft_result = run_moderator(
            patient_case=item["patient_case"],
            full_transcript=item["debate_transcript"],
            state=item["state"],
            use_finetuned=True,
        )

        records.append({
            "gpt4o_winner":   gpt_result.get("winner", ""),
            "llama_winner":   ft_result.get("winner", ""),
            "match":          gpt_result.get("winner", "") == ft_result.get("winner", ""),
        })

    df = pd.DataFrame(records)
    print(f"Winner agreement rate: {df['match'].mean() * 100:.1f}%")
    return df


if __name__ == "__main__":
    with open("data/evaluation_transcripts.json") as f:
        transcripts = json.load(f)

    df = evaluate_moderators(transcripts)
    df.to_csv("experiments/results/moderator_comparison.csv", index=False)
    print("Saved to experiments/results/moderator_comparison.csv")
