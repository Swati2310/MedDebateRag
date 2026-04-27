import re

from src.llm_client import generate

DOCTOR_A_SYSTEM = """
You are Doctor A, a senior physician in a clinical debate panel.
Your role: Advocate for the BEST answer option to the clinical question.

CRITICAL RULES:
1. If answer options (A/B/C/D) are provided, you MUST commit to ONE specific letter in every round
2. Start Round 1 by picking the letter you believe is correct — state it immediately
3. Use the retrieved medical literature as your primary evidence
4. Respond directly to Doctor B's counterarguments each round
5. You may change your letter choice if Doctor B's evidence is genuinely stronger
6. State your confidence honestly (0-100%)

ALWAYS end with EXACTLY this format:
---
ARGUMENT: [your full clinical argument]
OPTION_LETTER: [A|B|C|D — your chosen option letter]
DIAGNOSIS: [exact text of your chosen option]
CONFIDENCE: [integer 0-100]
KEY_EVIDENCE: [top 3 pieces of evidence from literature]
---
""".strip()

DOCTOR_A_PROMPT = """
## Patient Case
{patient_case}

## Retrieved Medical Literature (your evidence base)
{retrieved_docs}

## Debate History So Far
{debate_history}

## Doctor B's Last Argument
{doctor_b_argument}

{instruction}

End with the required format. You MUST include OPTION_LETTER: [A|B|C|D].
""".strip()


def run_doctor_a(
    patient_case: str,
    retrieved_docs: str,
    debate_history: str,
    doctor_b_argument: str,
    round_num: int,
    suggested_letter: str = None,
) -> dict:
    if round_num == 1:
        hint = f" Start by evaluating Option {suggested_letter} — argue for it if the evidence supports it." if suggested_letter else ""
        instruction = f"This is Round 1. Propose your initial answer with evidence.{hint}"
    else:
        instruction = f"This is Round {round_num}. Respond to Doctor B's argument. Defend or refine your position."

    prompt = DOCTOR_A_PROMPT.format(
        patient_case=patient_case,
        retrieved_docs=retrieved_docs,
        debate_history=debate_history,
        doctor_b_argument=doctor_b_argument if doctor_b_argument else "None yet.",
        instruction=instruction,
    )

    text = generate(prompt, system=DOCTOR_A_SYSTEM, temperature=0.7)
    return parse_agent_response(text)


def parse_agent_response(text: str) -> dict:
    """Extract structured fields from agent response."""
    result = {"raw": text}

    patterns = {
        "argument":             r"ARGUMENT:\s*(.+?)(?=\nOPTION_LETTER:|\nDIAGNOSIS:|\Z)",
        "option_letter":        r"OPTION_LETTER:\s*([A-D])",
        "diagnosis":            r"DIAGNOSIS:\s*(.+?)(?=\nCONFIDENCE:|\Z)",
        "confidence":           r"CONFIDENCE:\s*(\d+)",
        "key_evidence":         r"KEY_EVIDENCE:\s*(.+?)(?=\n---|$)",
        # moderator-specific
        "winner":               r"WINNER:\s*(.+?)(?=\n|\Z)",
        "final_diagnosis":      r"FINAL_DIAGNOSIS:\s*(.+?)(?=\n|\Z)",
        "reasoning":            r"REASONING:\s*(.+?)(?=\nUNEXPLAINED|\Z)",
        "unexplained_symptoms": r"UNEXPLAINED_SYMPTOMS:\s*(.+?)(?=\nVERDICT|\Z)",
        "verdict_confidence":   r"VERDICT_CONFIDENCE:\s*(\d+)",
        # doctor_b-specific
        "weakness_in_a":        r"WEAKNESS_IN_A:\s*(.+?)(?=\nSUPPORTING|\Z)",
        "supporting_evidence":  r"SUPPORTING_EVIDENCE:\s*(.+?)(?=\n---|$)",
    }

    for field, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        result[field] = match.group(1).strip() if match else ""

    for field in ("confidence", "verdict_confidence"):
        try:
            result[field] = float(result[field])
        except (ValueError, TypeError):
            result[field] = 50.0

    return result
