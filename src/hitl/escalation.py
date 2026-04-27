def decide_escalation(pds_score: float, threshold: float = 0.5) -> tuple[bool, str]:
    """
    Human-in-the-Loop escalation decision based on PDS.

    Returns: (escalate: bool, reason: str)
    """
    if pds_score >= threshold:
        reason = (
            f"Position Drift Score ({pds_score:.3f}) exceeds threshold ({threshold}). "
            f"Agents showed significant position instability during debate. "
            f"Human physician review strongly recommended before any clinical decision."
        )
        return True, reason
    else:
        reason = (
            f"Position Drift Score ({pds_score:.3f}) is below threshold ({threshold}). "
            f"Agents maintained stable positions — system is confident in diagnosis."
        )
        return False, reason


def format_escalation_summary(state: dict) -> str:
    """Format a structured summary for the human doctor when escalation is triggered."""
    return f"""
CLINICAL AI ESCALATION NOTICE
{'=' * 40}

PATIENT CASE:
{state['patient_case']}

AI DEBATE SUMMARY:
  Doctor A argued: {state['doctor_a_diagnoses'][-1]}
  (Final confidence: {state['doctor_a_confidences'][-1]:.0f}%)

  Doctor B argued: {state['doctor_b_diagnoses'][-1]}
  (Final confidence: {state['doctor_b_confidences'][-1]:.0f}%)

UNCERTAINTY ANALYSIS (PDS):
  Overall PDS Score:   {state['position_drift_score']:.3f}
  Confidence Drift:    {state['pds_components']['confidence_drift']:.3f}
  Semantic Drift:      {state['pds_components']['semantic_drift']:.3f}
  Final Disagreement:  {state['pds_components']['final_disagreement']:.3f}

REASON FOR ESCALATION:
{state['escalation_reason']}

[!] Human physician review required before any clinical decision.
""".strip()
