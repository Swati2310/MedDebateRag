"""
Generate moderator training data using Gemini as the teacher model.

Strategy:
1. Run Gemini (strong model) as moderator on debate transcripts
2. Collect high-quality (transcript → verdict) pairs
3. These can be used to fine-tune a smaller model on Colab if needed
"""

import json

from tqdm import tqdm

from src.agents.moderator import MODERATOR_SYSTEM
from src.llm_client import generate


def generate_moderator_training_sample(debate_transcript: str, patient_case: str) -> str:
    """Use Gemini to generate gold-standard moderator verdicts."""
    prompt = f"Patient: {patient_case}\n\nTranscript:\n{debate_transcript}"
    return generate(prompt, system=MODERATOR_SYSTEM, temperature=0.3)


def build_training_dataset(debates_df, output_path: str, n_samples: int = 1000):
    """
    Generate n_samples training examples for moderator fine-tuning.
    Format: instruction-following style.
    """
    training_data = []

    for i, row in tqdm(debates_df.head(n_samples).iterrows(), total=n_samples):
        verdict = generate_moderator_training_sample(
            row["debate_transcript"],
            row["patient_case"],
        )

        training_data.append({
            "instruction": MODERATOR_SYSTEM,
            "input":  f"Patient: {row['patient_case']}\n\nDebate:\n{row['debate_transcript']}",
            "output": verdict,
        })

    with open(output_path, "w") as f:
        json.dump(training_data, f, indent=2)

    print(f"Generated {len(training_data)} training samples → {output_path}")
    return training_data
