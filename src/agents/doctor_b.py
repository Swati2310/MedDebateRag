from src.llm_client import generate
from src.agents.doctor_a import parse_agent_response

DOCTOR_B_SYSTEM = """
You are Doctor B, a senior physician playing devil's advocate.
Your role: Challenge Doctor A by arguing for a DIFFERENT answer option.

CRITICAL RULES:
1. You MUST pick a DIFFERENT option letter than Doctor A — never the same letter
2. State your option letter immediately at the start
3. Argue why your option is better supported by clinical evidence and literature
4. Directly attack the weaknesses in Doctor A's chosen option
5. Use retrieved medical literature as evidence
6. State your confidence honestly (0-100%)

ALWAYS end with EXACTLY this format:
---
ARGUMENT: [your full clinical argument for your chosen option]
OPTION_LETTER: [A|B|C|D — MUST be different from Doctor A's letter]
DIAGNOSIS: [exact text of your chosen option]
CONFIDENCE: [integer 0-100]
WEAKNESS_IN_A: [specific weakness in Doctor A's chosen option]
SUPPORTING_EVIDENCE: [evidence from literature supporting your option]
---
""".strip()

DOCTOR_B_PROMPT = """
## Patient Case
{patient_case}

## Retrieved Medical Literature
{retrieved_docs}

## Debate History So Far
{debate_history}

## Doctor A's Last Argument
{doctor_a_argument}
Doctor A chose: Option {doctor_a_letter}

{instruction}

You MUST pick a DIFFERENT letter than Doctor A (not {doctor_a_letter}).
End with the required format including OPTION_LETTER.
""".strip()


def run_doctor_b(
    patient_case: str,
    retrieved_docs: str,
    debate_history: str,
    doctor_a_argument: str,
    round_num: int,
    doctor_a_letter: str = "?",
    suggested_letter: str = None,
) -> dict:
    if round_num == 1:
        hint = f" Start by evaluating Option {suggested_letter} as your counter-position." if suggested_letter else ""
        instruction = f"This is Round 1. Doctor A chose Option {doctor_a_letter}. Pick a DIFFERENT option and argue why yours is better.{hint}"
    else:
        instruction = f"This is Round {round_num}. Continue challenging Doctor A's Option {doctor_a_letter}. Strengthen your alternative."

    prompt = DOCTOR_B_PROMPT.format(
        patient_case=patient_case,
        retrieved_docs=retrieved_docs,
        debate_history=debate_history,
        doctor_a_argument=doctor_a_argument,
        doctor_a_letter=doctor_a_letter,
        instruction=instruction,
    )

    text = generate(prompt, system=DOCTOR_B_SYSTEM, temperature=0.7)
    return parse_agent_response(text or "")
