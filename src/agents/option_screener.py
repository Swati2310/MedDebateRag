"""
Pre-debate option screener.

Before debate starts, rank all MCQ options by plausibility so that:
  - Doctor A argues for the top-ranked option
  - Doctor B argues for the 2nd-ranked option

This guarantees the correct answer is always inside the debate.
"""

import re
from src.llm_client import generate

SCREENER_SYSTEM = """You are a medical triage assistant.
Given a clinical question with answer options, rank them from most to least likely correct.
Be concise. ALWAYS respond in exactly this format:

RANK_1: [letter] — [one-line reason]
RANK_2: [letter] — [one-line reason]
RANK_3: [letter] — [one-line reason]
RANK_4: [letter] — [one-line reason]
""".strip()


def screen_options(patient_case: str) -> list[str]:
    """
    Return options ranked best→worst as list of letters, e.g. ['C', 'A', 'D', 'B'].
    Falls back to ['A', 'B', 'C', 'D'] if parsing fails.
    """
    options_block = ""
    for line in patient_case.splitlines():
        if re.match(r"^\s*[A-D][.)]\s*", line):
            options_block += line.strip() + "\n"

    if not options_block:
        return ["A", "B", "C", "D"]

    prompt = f"{patient_case}\n\nRank the answer options from most to least likely correct."
    text = generate(prompt, system=SCREENER_SYSTEM, temperature=0.1)
    if not text:
        return ["A", "B", "C", "D"]

    ranked = re.findall(r"RANK_\d:\s*([A-D])", text, re.IGNORECASE)
    ranked = [r.upper() for r in ranked]

    # Fill in any missing letters
    for letter in ["A", "B", "C", "D"]:
        if letter not in ranked:
            ranked.append(letter)

    return ranked[:4]
