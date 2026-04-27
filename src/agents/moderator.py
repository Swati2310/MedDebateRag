import re

from src.llm_client import generate
from src.agents.doctor_a import parse_agent_response


def _parse_options(patient_case: str) -> dict:
    options = {}
    for m in re.finditer(r"^\s*([A-D])[.)]\s*(.+)$", patient_case, re.MULTILINE):
        options[m.group(1)] = m.group(2).strip()
    return options

MODERATOR_SYSTEM = """
You are a neutral senior medical moderator.
You do NOT propose new diagnoses. You evaluate debate arguments objectively.

Your job:
1. Assess which doctor provided stronger clinical evidence
2. Identify symptoms left unexplained by either doctor
3. Select the best answer from the provided options based on the debate
4. Assign a confidence score to your verdict

ALWAYS end with EXACTLY this format:
---
WINNER: [Doctor A | Doctor B | INCONCLUSIVE]
FINAL_ANSWER_LETTER: [A|B|C|D]
FINAL_DIAGNOSIS: [copy the exact text of your chosen option]
REASONING: [why this answer won the debate]
UNEXPLAINED_SYMPTOMS: [symptoms neither doctor addressed well]
VERDICT_CONFIDENCE: [integer 0-100]
---
""".strip()

MODERATOR_PROMPT = """
## Patient Case
{patient_case}

## Answer Options
{options_formatted}

## Full Debate Transcript
{full_transcript}

## Doctor A Final Position
- Diagnosis: {a_diagnosis}
- Confidence: {a_confidence}%
- Key Evidence: {a_evidence}

## Doctor B Final Position
- Diagnosis: {b_diagnosis}
- Confidence: {b_confidence}%
- Weakness found in A: {b_weakness}

Evaluate both arguments and select the best answer option (A/B/C/D).
""".strip()


def run_moderator(
    patient_case: str,
    full_transcript: str,
    state: dict,
    use_finetuned: bool = False,
) -> dict:
    # Fine-tuned path removed — Gemini is used as moderator directly.
    # use_finetuned flag kept for API compatibility but ignored.

    options = _parse_options(patient_case)
    if options:
        options_formatted = "\n".join(f"  {k}. {v}" for k, v in options.items())
    else:
        options_formatted = "(no multiple-choice options — give best clinical answer)"

    a_letter = state.get("doctor_a_letters", ["?"])[-1]
    b_letter = state.get("doctor_b_letters", ["?"])[-1]

    prompt = MODERATOR_PROMPT.format(
        patient_case=patient_case,
        options_formatted=options_formatted,
        full_transcript=full_transcript,
        a_diagnosis=f"Option {a_letter}: {state['doctor_a_diagnoses'][-1]}",
        a_confidence=state["doctor_a_confidences"][-1],
        a_evidence=state["doctor_a_arguments"][-1][:300],
        b_diagnosis=f"Option {b_letter}: {state['doctor_b_diagnoses'][-1]}",
        b_confidence=state["doctor_b_confidences"][-1],
        b_weakness=state["doctor_b_arguments"][-1][:300],
    )

    text = generate(prompt, system=MODERATOR_SYSTEM, temperature=0.3)
    parsed = parse_agent_response(text)

    # Extract the direct letter choice and use it as final_diagnosis
    letter_match = re.search(r"FINAL_ANSWER_LETTER:\s*([A-D])", text, re.IGNORECASE)
    if letter_match and options:
        letter = letter_match.group(1).upper()
        parsed["final_diagnosis"] = options.get(letter, parsed.get("final_diagnosis", ""))
        parsed["moderator_answer_letter"] = letter

    return parsed
