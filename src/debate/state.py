from typing import Any, Dict, List, Optional, TypedDict


class DebateState(TypedDict):
    # ── Input ──────────────────────────────────────
    patient_case:            str
    ground_truth:            str

    # ── RAG ────────────────────────────────────────
    retrieved_docs_a:        str
    retrieved_docs_b:        str

    # ── Debate Control ─────────────────────────────
    current_round:           int
    max_rounds:              int

    # ── Doctor A Outputs (per round) ───────────────
    doctor_a_arguments:      List[str]
    doctor_a_confidences:    List[float]
    doctor_a_diagnoses:      List[str]
    doctor_a_letters:        List[str]

    # ── Doctor B Outputs (per round) ───────────────
    doctor_b_arguments:      List[str]
    doctor_b_confidences:    List[float]
    doctor_b_diagnoses:      List[str]
    doctor_b_letters:        List[str]

    # ── Moderator Output ───────────────────────────
    moderator_verdict:       Optional[str]
    final_diagnosis:         Optional[str]
    verdict_confidence:      Optional[float]

    # ── PDS (Novel Metric) ─────────────────────────
    position_drift_score:    Optional[float]
    pds_components:          Optional[Dict[str, Any]]

    # ── HITL ───────────────────────────────────────
    escalate_to_human:       bool
    escalation_reason:       Optional[str]

    # ── Option Screener ────────────────────────────
    option_ranking:          List[str]   # e.g. ['C', 'A', 'D', 'B']

    # ── Meta ───────────────────────────────────────
    use_finetuned_moderator: bool
    skip_rag:                bool   # when True, debate runs without PubMed retrieval
