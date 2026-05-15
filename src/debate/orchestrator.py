import re

from langgraph.graph import StateGraph, END

from src.debate.state import DebateState
from src.agents.doctor_a import run_doctor_a
from src.agents.doctor_b import run_doctor_b
from src.agents.moderator import run_moderator
from src.agents.answer_extractor import extract_final_answer
from src.agents.option_screener import screen_options
from src.uncertainty.pds import PositionDriftScore
from src.hitl.escalation import decide_escalation


def _parse_options(patient_case: str) -> dict:
    """Extract {A: diagnosis_text, B: diagnosis_text, ...} from case."""
    options = {}
    for m in re.finditer(r"^\s*([A-D])[.)]\s*(.+)$", patient_case, re.MULTILINE):
        options[m.group(1)] = m.group(2).strip()
    return options

PDS_THRESHOLD = 0.15

# Lazy-loaded global retriever (initialised on first use)
_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is None:
        from src.rag.retriever import load_retriever
        _retriever = load_retriever()
    return _retriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_debate_history(state: DebateState) -> str:
    history = ""
    for i in range(len(state["doctor_a_arguments"])):
        history += f"\n--- Round {i+1} ---\n"
        history += f"Doctor A: {state['doctor_a_arguments'][i][:300]}...\n"
        if i < len(state["doctor_b_arguments"]):
            history += f"Doctor B: {state['doctor_b_arguments'][i][:300]}...\n"
    return history or "No debate history yet."


def format_full_transcript(state: DebateState) -> str:
    transcript = ""
    for i in range(len(state["doctor_a_arguments"])):
        transcript += f"\n=== ROUND {i+1} ===\n"
        transcript += f"DOCTOR A (Confidence: {state['doctor_a_confidences'][i]}%):\n"
        transcript += f"{state['doctor_a_arguments'][i]}\n\n"
        if i < len(state["doctor_b_arguments"]):
            transcript += f"DOCTOR B (Confidence: {state['doctor_b_confidences'][i]}%):\n"
            transcript += f"{state['doctor_b_arguments'][i]}\n\n"
    return transcript


# ---------------------------------------------------------------------------
# Node Functions
# ---------------------------------------------------------------------------

def retrieve_node(state: DebateState) -> DebateState:
    """RAG retrieval + option screening before debate starts.

    Retrieval is targeted: after screening we know which diagnosis each
    doctor will argue, so we search PubMed specifically for that diagnosis
    rather than using a generic query for both doctors.

    When state['skip_rag'] is True, only runs the option screener —
    no retrieval — so doctors debate from case text alone.
    """
    patient_case = state["patient_case"]

    # Always run option screener regardless of RAG flag
    state["option_ranking"] = screen_options(patient_case)

    if state.get("skip_rag", False):
        # Debate-only mode: no PubMed retrieval
        state["retrieved_docs_a"] = "No literature retrieved — debate from case text only."
        state["retrieved_docs_b"] = "No literature retrieved — debate from case text only."
        return state

    retriever = _get_retriever()
    question_part = patient_case.split("Answer options:")[0].strip()
    base_query = question_part[-600:] if len(question_part) > 600 else question_part

    ranking = state["option_ranking"]
    options = _parse_options(patient_case)

    if options and len(ranking) >= 2:
        a_diagnosis = options.get(ranking[0], "")
        b_diagnosis = options.get(ranking[1], "")
        query_a = f"{base_query} {a_diagnosis}".strip()
        query_b = f"{base_query} {b_diagnosis}".strip()
    else:
        query_a = base_query
        query_b = base_query + " alternative diagnosis differential"

    state["retrieved_docs_a"] = retriever.format_for_prompt(query_a)
    state["retrieved_docs_b"] = retriever.format_for_prompt(query_b)
    return state


def doctor_a_node(state: DebateState) -> DebateState:
    """Run Doctor A — starts with the top-ranked option from the screener."""
    ranking = state.get("option_ranking", ["A", "B", "C", "D"])
    suggested = ranking[0] if ranking else "A"
    result = run_doctor_a(
        patient_case=state["patient_case"],
        retrieved_docs=state["retrieved_docs_a"],
        debate_history=format_debate_history(state),
        doctor_b_argument=state["doctor_b_arguments"][-1] if state["doctor_b_arguments"] else "",
        round_num=state["current_round"] + 1,
        suggested_letter=suggested if state["current_round"] == 0 else None,
    )

    state["doctor_a_arguments"].append(result.get("argument", result.get("raw", "")))
    state["doctor_a_confidences"].append(result.get("confidence", 50.0))
    state["doctor_a_diagnoses"].append(result.get("diagnosis", "Unknown"))
    state["doctor_a_letters"].append(result.get("option_letter", "?"))
    return state


def doctor_b_node(state: DebateState) -> DebateState:
    """Run Doctor B — starts with the 2nd-ranked option from the screener."""
    a_letter = state.get("doctor_a_letters", ["?"])[-1]
    ranking = state.get("option_ranking", ["A", "B", "C", "D"])
    # Pick 2nd ranked option that differs from Doctor A's current letter
    suggested = next((l for l in ranking[1:] if l != a_letter), None)
    result = run_doctor_b(
        patient_case=state["patient_case"],
        retrieved_docs=state["retrieved_docs_b"],
        debate_history=format_debate_history(state),
        doctor_a_argument=state["doctor_a_arguments"][-1],
        round_num=state["current_round"] + 1,
        doctor_a_letter=a_letter,
        suggested_letter=suggested if state["current_round"] == 0 else None,
    )

    state["doctor_b_arguments"].append(result.get("argument", result.get("raw", "")))
    state["doctor_b_confidences"].append(result.get("confidence", 50.0))
    state["doctor_b_diagnoses"].append(result.get("diagnosis", "Unknown"))
    state["doctor_b_letters"].append(result.get("option_letter", "?"))
    state["current_round"] += 1
    return state


def moderator_node(state: DebateState) -> DebateState:
    """Run Moderator after all debate rounds."""
    transcript = format_full_transcript(state)

    result = run_moderator(
        patient_case=state["patient_case"],
        full_transcript=transcript,
        state=state,
        use_finetuned=state.get("use_finetuned_moderator", False),
    )

    state["moderator_verdict"] = result.get("winner", "INCONCLUSIVE")
    state["final_diagnosis"]   = result.get("final_diagnosis", result.get("diagnosis", ""))
    state["verdict_confidence"] = result.get("verdict_confidence", result.get("confidence", 50.0))
    return state


def answer_extractor_node(state: DebateState) -> DebateState:
    """Map debate conclusion → exact MCQ option text (if options present)."""
    result = extract_final_answer(state, state["patient_case"])
    if result["answer_text"]:
        state["final_diagnosis"] = result["answer_text"]
    return state


def pds_node(state: DebateState) -> DebateState:
    """Compute Position Drift Score."""
    calculator = PositionDriftScore()
    pds, components = calculator.compute(
        doctor_a_confidences=state["doctor_a_confidences"],
        doctor_b_confidences=state["doctor_b_confidences"],
        doctor_a_arguments=state["doctor_a_arguments"],
        doctor_b_arguments=state["doctor_b_arguments"],
    )

    state["position_drift_score"] = pds
    state["pds_components"]       = components
    return state


def escalation_node(state: DebateState) -> DebateState:
    """HITL decision based on PDS."""
    escalate, reason = decide_escalation(
        pds_score=state["position_drift_score"],
        threshold=PDS_THRESHOLD,
    )
    state["escalate_to_human"] = escalate
    state["escalation_reason"] = reason
    return state


# ---------------------------------------------------------------------------
# Conditional Edge
# ---------------------------------------------------------------------------

def should_continue(state: DebateState) -> str:
    if state["current_round"] < state["max_rounds"]:
        return "continue"
    return "moderate"


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

def build_debate_graph():
    graph = StateGraph(DebateState)

    graph.add_node("retrieve",         retrieve_node)
    graph.add_node("doctor_a",         doctor_a_node)
    graph.add_node("doctor_b",         doctor_b_node)
    graph.add_node("moderator",        moderator_node)
    graph.add_node("answer_extractor", answer_extractor_node)
    graph.add_node("compute_pds",      pds_node)
    graph.add_node("escalation",       escalation_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve",         "doctor_a")
    graph.add_edge("doctor_a",         "doctor_b")
    graph.add_conditional_edges(
        "doctor_b",
        should_continue,
        {"continue": "doctor_a", "moderate": "moderator"},
    )
    graph.add_edge("moderator",        "answer_extractor")
    graph.add_edge("answer_extractor", "compute_pds")
    graph.add_edge("compute_pds",      "escalation")
    graph.add_edge("escalation",       END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_debate(
    patient_case: str,
    ground_truth: str,
    max_rounds: int = 3,
    use_finetuned: bool = False,
    use_rag: bool = True,
) -> dict:
    graph = build_debate_graph()

    initial_state: DebateState = {
        "patient_case":            patient_case,
        "ground_truth":            ground_truth,
        # When use_rag=False, pre-fill empty docs so retrieve_node is skipped
        "retrieved_docs_a":        "" if use_rag else "No literature retrieved.",
        "retrieved_docs_b":        "" if use_rag else "No literature retrieved.",
        "current_round":           0,
        "max_rounds":              max_rounds,
        "doctor_a_arguments":      [],
        "doctor_a_confidences":    [],
        "doctor_a_diagnoses":      [],
        "doctor_a_letters":        [],
        "doctor_b_arguments":      [],
        "doctor_b_confidences":    [],
        "doctor_b_diagnoses":      [],
        "doctor_b_letters":        [],
        "option_ranking":          [],
        "moderator_verdict":       None,
        "final_diagnosis":         None,
        "verdict_confidence":      None,
        "position_drift_score":    None,
        "pds_components":          None,
        "escalate_to_human":       False,
        "escalation_reason":       None,
        "use_finetuned_moderator": use_finetuned,
        "skip_rag":                not use_rag,
    }

    return graph.invoke(initial_state)


def run_debate_clinical(
    patient_case: str,
    max_rounds: int = 3,
) -> dict:
    """
    Clinical mode — for free-text cases with no MCQ options.
    Uses RAG to generate top 4 differential diagnoses, formats them
    as A/B/C/D options, then runs the standard debate pipeline.
    Returns the final state plus the generated differentials.
    """
    from src.agents.differential_generator import generate_differentials, build_clinical_case

    retriever = _get_retriever()
    query = patient_case[-600:] if len(patient_case) > 600 else patient_case
    retrieved_docs = retriever.format_for_prompt(query)

    differentials = generate_differentials(patient_case, retrieved_docs)
    enriched_case = build_clinical_case(patient_case, differentials)

    result = run_debate(enriched_case, ground_truth="", max_rounds=max_rounds)
    result["generated_differentials"] = differentials
    result["enriched_case"] = enriched_case
    return result
