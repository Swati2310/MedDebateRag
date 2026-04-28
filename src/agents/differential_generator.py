"""
Differential Diagnosis Generator — Clinical Mode

For free-text patient cases (no MCQ options provided), this module
uses RAG-retrieved PubMed abstracts to generate the top 4 candidate
diagnoses, which are then formatted as A/B/C/D options and fed into
the standard debate pipeline.
"""

import re
from src.llm_client import generate

DIFF_SYSTEM = """You are a senior physician generating a differential diagnosis.
Given a patient case and relevant medical literature, identify the top 4 most likely diagnoses.

Rules:
1. Base your differentials on BOTH the patient presentation AND the retrieved literature
2. Rank from most to least likely
3. Each diagnosis should be specific and clinically actionable
4. Do NOT repeat the same diagnosis in different wording

ALWAYS respond in exactly this format:
DIFFERENTIAL_1: [specific diagnosis name]
DIFFERENTIAL_2: [specific diagnosis name]
DIFFERENTIAL_3: [specific diagnosis name]
DIFFERENTIAL_4: [specific diagnosis name]
""".strip()

DIFF_PROMPT = """
## Patient Case
{patient_case}

## Relevant Medical Literature (use this as evidence)
{retrieved_docs}

Based on the patient presentation and the medical literature above,
list the top 4 most likely diagnoses ranked from most to least likely.
""".strip()


def generate_differentials(patient_case: str, retrieved_docs: str) -> list[str]:
    """
    Generate top 4 differential diagnoses from free-text case + RAG docs.
    Returns list of 4 diagnosis strings.
    """
    prompt = DIFF_PROMPT.format(
        patient_case=patient_case,
        retrieved_docs=retrieved_docs[:3000],
    )
    text = generate(prompt, system=DIFF_SYSTEM, temperature=0.3)
    if not text:
        return ["Diagnosis A", "Diagnosis B", "Diagnosis C", "Diagnosis D"]

    matches = re.findall(r"DIFFERENTIAL_\d:\s*(.+)", text)
    differentials = [m.strip() for m in matches if m.strip()]

    # Pad to 4 if parsing missed some
    fallbacks = ["Further evaluation needed", "Alternative diagnosis",
                 "Rule out secondary cause", "Consult specialist"]
    while len(differentials) < 4:
        differentials.append(fallbacks[len(differentials) - 1])

    return differentials[:4]


def build_clinical_case(patient_case: str, differentials: list[str]) -> str:
    """
    Append generated differentials to the patient case as MCQ options A/B/C/D.
    The result feeds directly into the standard debate pipeline.
    """
    options = "\n".join(
        f"  {letter}. {diagnosis}"
        for letter, diagnosis in zip(["A", "B", "C", "D"], differentials)
    )
    return f"{patient_case.strip()}\n\nAnswer options:\n{options}"
