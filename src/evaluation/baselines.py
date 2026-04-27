import re
from collections import Counter

from src.llm_client import generate

_MCQ_SYSTEM = """You are a medical exam assistant.
If the question has answer options (A/B/C/D), you MUST pick exactly one option letter and output its exact text.
ALWAYS respond in this format:
ANSWER_LETTER: [A|B|C|D]
ANSWER_TEXT: [exact option text copied verbatim]
If there are no options, just output the diagnosis."""


def _parse_mcq(text, patient_case: str) -> str:
    """Extract the chosen option text from a response, falling back to raw text."""
    if not text:
        return ""
    # Try structured format first
    m = re.search(r"ANSWER_TEXT:\s*(.+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Try to extract letter and map to option
    m_letter = re.search(r"ANSWER_LETTER:\s*([A-D])", text, re.IGNORECASE)
    if m_letter:
        letter = m_letter.group(1).upper()
        for line in patient_case.splitlines():
            if re.match(rf"^\s*{letter}[.)]\s*", line):
                return re.sub(rf"^\s*{letter}[.)]\s*", "", line).strip()
    return text.strip()


def baseline_single_llm(patient_case: str) -> str:
    """No debate, no RAG — just ask Gemini directly."""
    text = generate(
        f"Answer this clinical question by selecting the best option.\n\n{patient_case}",
        system=_MCQ_SYSTEM,
        temperature=0.3,
    )
    return _parse_mcq(text, patient_case)


def baseline_cot(patient_case: str) -> str:
    """Single LLM with step-by-step reasoning, then pick an option."""
    text = generate(
        f"Think step by step, then select the best answer option.\n\n{patient_case}",
        system=_MCQ_SYSTEM,
        temperature=0.3,
    )
    return _parse_mcq(text, patient_case)


def baseline_self_consistency(patient_case: str, n: int = 5) -> str:
    """Sample n answers, return majority vote."""
    answers = []
    for _ in range(n):
        text = generate(
            f"Select the single best answer option.\n\n{patient_case}",
            system=_MCQ_SYSTEM,
            temperature=0.9,
        )
        answers.append(_parse_mcq(text, patient_case))
    return Counter(answers).most_common(1)[0][0]


def baseline_rag_single(patient_case: str, retriever) -> str:
    """RAG retrieval + single LLM — no debate."""
    docs = retriever.format_for_prompt(patient_case[:200])
    text = generate(
        f"Using the medical literature below, select the best answer option.\n\nLiterature:\n{docs}\n\nPatient:\n{patient_case}",
        system=_MCQ_SYSTEM,
        temperature=0.3,
    )
    return _parse_mcq(text, patient_case)
