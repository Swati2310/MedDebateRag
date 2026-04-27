"""
Final Answer Extractor — runs after the debate.

The debate agents reason freely about the case. This module then takes
the moderator's conclusion + the original MCQ options and maps the
debate outcome to the EXACT correct option text.

This separates clinical reasoning (debate) from answer selection (extractor),
which significantly improves accuracy on MCQ-style questions.
"""

import re

from src.llm_client import generate

EXTRACTOR_SYSTEM = """
You are a medical exam answer selector.
You are given a clinical question, the answer options (A/B/C/D),
and a summary of a clinical debate.

Your ONLY job: pick which option letter (A, B, C, or D) best answers
the question, based on the debate reasoning.

Rules:
1. You MUST pick exactly one letter: A, B, C, or D
2. Do not explain — just output the letter and the option text
3. Use the debate as context, but always anchor to the provided options

ALWAYS respond in EXACTLY this format:
ANSWER_LETTER: [A|B|C|D]
ANSWER_TEXT: [exact option text, copied verbatim]
""".strip()

EXTRACTOR_PROMPT = """
## Clinical Question
{question}

## Answer Options
{options_formatted}

## Debate Summary
- Doctor A final diagnosis: {a_diagnosis} (confidence: {a_conf:.0f}%)
- Doctor B final diagnosis: {b_diagnosis} (confidence: {b_conf:.0f}%)
- Moderator verdict: {verdict}
- Moderator final diagnosis: {mod_diagnosis}

Based on the debate reasoning above, which option letter best answers
the clinical question? Pick the option that most closely matches
the moderator's conclusion.
""".strip()


def extract_final_answer(state: dict, patient_case: str) -> dict:
    """
    Map debate conclusion → exact MCQ option text.

    Returns dict with keys: answer_letter, answer_text (= final_diagnosis override)
    Falls back to moderator's original diagnosis if no options found.
    """
    # Parse options from the patient case text
    options = _parse_options(patient_case)
    if not options:
        return {"answer_letter": "", "answer_text": state.get("final_diagnosis", "")}

    # If moderator already picked a letter directly, trust it (skip extra LLM call)
    if state.get("moderator_answer_letter") and state["moderator_answer_letter"] in options:
        letter = state["moderator_answer_letter"]
        return {"answer_letter": letter, "answer_text": options[letter]}

    options_formatted = "\n".join(
        f"  {letter}. {text}" for letter, text in options.items()
    )

    # Extract the question (everything before "Answer options:")
    question = patient_case.split("Answer options:")[0].strip()

    prompt = EXTRACTOR_PROMPT.format(
        question=question[:800],
        options_formatted=options_formatted,
        a_diagnosis=state["doctor_a_diagnoses"][-1],
        a_conf=state["doctor_a_confidences"][-1],
        b_diagnosis=state["doctor_b_diagnoses"][-1],
        b_conf=state["doctor_b_confidences"][-1],
        verdict=state.get("moderator_verdict", ""),
        mod_diagnosis=state.get("final_diagnosis", ""),
    )

    text = generate(prompt, system=EXTRACTOR_SYSTEM, temperature=0.1)

    letter = _parse_letter(text)
    answer_text = options.get(letter, state.get("final_diagnosis", ""))

    return {"answer_letter": letter, "answer_text": answer_text}


def _parse_options(patient_case: str) -> dict:
    """Extract {A: text, B: text, ...} from patient case string."""
    options = {}
    # Match lines like "  A. some text" or "  A) some text"
    for m in re.finditer(r"^\s*([A-D])[.)]\s*(.+)$", patient_case, re.MULTILINE):
        options[m.group(1)] = m.group(2).strip()
    return options


def _parse_letter(text: str) -> str:
    """Extract the answer letter from extractor response."""
    m = re.search(r"ANSWER_LETTER:\s*([A-D])", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Fallback: find any lone A/B/C/D
    m = re.search(r"\b([A-D])\b", text)
    return m.group(1).upper() if m else "A"
