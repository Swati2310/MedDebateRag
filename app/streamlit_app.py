import os
import sys

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
load_dotenv()

from src.data.load_ddxplus import format_patient_case, load_ddxplus
from src.debate.orchestrator import (
    retrieve_node, doctor_a_node, doctor_b_node,
    moderator_node, answer_extractor_node, pds_node, escalation_node,
)
from src.agents.differential_generator import generate_differentials, build_clinical_case
from src.hitl.escalation import decide_escalation, format_escalation_summary
from src.debate.state import DebateState
from app.report_generator import generate_pdf_report

st.set_page_config(page_title="MedDebate-RAG", layout="wide", page_icon="🏥")

st.title("MedDebate-RAG")
st.caption("Uncertainty-Aware Multi-Agent Clinical Debate | Frontiers of LLMs Project")

st.info(
    "**How it works:** Two AI doctors debate a diagnosis using real medical literature. "
    "A moderator computes the **Position Drift Score (PDS)** to measure uncertainty. "
    "If PDS is too high → escalates to human doctor."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    max_rounds    = st.slider("Debate Rounds", 1, 5, 3)
    pds_threshold = st.slider("Escalation Threshold", 0.05, 0.5, 0.15,
                               help="PDS above this → escalate to human")
    use_finetuned = False  # Fine-tuned moderator — coming soon

    st.markdown("---")
    st.markdown("**PDS Thresholds:**")
    st.markdown("🟢 < 0.10 — High Confidence")
    st.markdown("🟡 0.10–0.15 — Medium Confidence")
    st.markdown("🔴 > 0.15 — Escalate to Human")

# ── Patient Case Input ────────────────────────────────────────────────────────
st.subheader("Patient Case")
st.caption("Enter any free-text clinical case — no options needed. The system generates differential diagnoses from PubMed evidence.")

col1, col2 = st.columns([3, 1])

with col2:
    st.markdown("**Load a sample case:**")
    if st.button("Load Random Case", use_container_width=True):
        df = load_ddxplus()
        sample = df.sample(1).iloc[0]
        raw = format_patient_case(sample)
        # Strip MCQ options — clinical mode uses free text only
        clinical_text = raw.split("Answer options:")[0].strip()
        st.session_state["patient_case"]  = clinical_text
        st.session_state["ground_truth"]  = sample["pathology"]
        st.rerun()

    if "ground_truth" in st.session_state:
        st.info(f"Ground truth:\n**{st.session_state['ground_truth']}**")

with col1:
    patient_case = st.text_area(
        "Patient symptoms, history, lab results:",
        value=st.session_state.get("patient_case", ""),
        placeholder=(
            "Example: 45-year-old male presenting with chest pain radiating to the left arm, "
            "shortness of breath, diaphoresis. Elevated troponin (2.5 ng/mL). "
            "History of hypertension and smoking for 20 years."
        ),
        height=160,
    )

# ── Run Debate ────────────────────────────────────────────────────────────────
if st.button("Start Clinical Debate", type="primary", disabled=not patient_case):

    # Initialise state
    state: DebateState = {
        "patient_case":            patient_case,
        "ground_truth":            "",
        "retrieved_docs_a":        "",
        "retrieved_docs_b":        "",
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
    }

    # ── Step 1: RAG Retrieval + Differential Generation ──────────────────
    with st.status("Retrieving medical literature from PubMed...", expanded=True) as s:
        # Get RAG docs first
        from src.rag.retriever import load_retriever
        retriever = load_retriever()
        query = patient_case[-600:] if len(patient_case) > 600 else patient_case
        retrieved_docs = retriever.format_for_prompt(query)
        st.write("Retrieved relevant PubMed abstracts.")
        s.update(label="Medical literature retrieved", state="complete", expanded=False)

    with st.status("Generating differential diagnoses from evidence...", expanded=True) as s:
        differentials = generate_differentials(patient_case, retrieved_docs)
        enriched_case = build_clinical_case(patient_case, differentials)
        state["patient_case"] = enriched_case

        st.markdown("**AI-generated differential diagnoses (based on PubMed evidence):**")
        for letter, dx in zip(["A", "B", "C", "D"], differentials):
            st.markdown(f"- **{letter}.** {dx}")
        s.update(label=f"4 differentials generated", state="complete", expanded=True)

    with st.status("Ranking differentials & preparing debate...", expanded=False) as s:
        state = retrieve_node(state)
        ranking = state.get("option_ranking", [])
        if ranking:
            st.write(f"Debate seeding: **{'  >  '.join(ranking)}**")
        s.update(label="Debate ready", state="complete", expanded=False)

    # ── Steps 2–N: Debate Rounds ──────────────────────────────────────────
    round_results = []   # store per-round data for transcript display later

    for round_num in range(max_rounds):
        # Doctor A
        with st.status(f"Round {round_num + 1} — Doctor A building argument...", expanded=True) as s:
            state = doctor_a_node(state)
            a_letter = state["doctor_a_letters"][-1]
            a_conf   = state["doctor_a_confidences"][-1]
            st.write(f"Chose **Option {a_letter}** with **{a_conf:.0f}%** confidence")
            st.caption(state["doctor_a_arguments"][-1][:200] + "...")
            s.update(label=f"Round {round_num + 1} — Doctor A: Option {a_letter} ({a_conf:.0f}%)",
                     state="complete", expanded=False)

        # Doctor B
        with st.status(f"Round {round_num + 1} — Doctor B responding...", expanded=True) as s:
            state = doctor_b_node(state)
            b_letter = state["doctor_b_letters"][-1]
            b_conf   = state["doctor_b_confidences"][-1]
            st.write(f"Chose **Option {b_letter}** with **{b_conf:.0f}%** confidence")
            st.caption(state["doctor_b_arguments"][-1][:200] + "...")
            s.update(label=f"Round {round_num + 1} — Doctor B: Option {b_letter} ({b_conf:.0f}%)",
                     state="complete", expanded=False)

        round_results.append({
            "a_letter": a_letter, "a_conf": a_conf,
            "b_letter": b_letter, "b_conf": b_conf,
        })

    # ── Moderator ─────────────────────────────────────────────────────────
    with st.status("Moderator evaluating full debate...", expanded=True) as s:
        from src.debate.orchestrator import format_full_transcript
        transcript = format_full_transcript(state)
        state = moderator_node(state)
        st.write(f"Verdict: **{state['moderator_verdict']}** | Confidence: **{state['verdict_confidence']:.0f}%**")
        s.update(label=f"Moderator verdict: {state['moderator_verdict']}", state="complete", expanded=False)

    # ── Answer Extractor + PDS + Escalation ───────────────────────────────
    with st.status("Computing Position Drift Score...", expanded=False) as s:
        state = answer_extractor_node(state)
        state = pds_node(state)
        # Use slider threshold instead of hardcoded one
        escalate, reason = decide_escalation(state["position_drift_score"], pds_threshold)
        state["escalate_to_human"] = escalate
        state["escalation_reason"] = reason
        s.update(label=f"PDS computed: {state['position_drift_score']:.3f}", state="complete")

    final_state = state

    # ── Debate Transcript ─────────────────────────────────────────────────
    st.subheader("Debate Transcript")
    n_rounds = len(final_state["doctor_a_arguments"])

    for round_num in range(n_rounds):
        with st.expander(f"Round {round_num + 1}", expanded=(round_num == n_rounds - 1)):
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("#### 🟢 Doctor A (Advocate)")
                st.write(final_state["doctor_a_arguments"][round_num])
                st.metric("Option",      final_state["doctor_a_letters"][round_num])
                st.metric("Diagnosis",   final_state["doctor_a_diagnoses"][round_num])
                st.metric("Confidence",  f"{final_state['doctor_a_confidences'][round_num]:.0f}%")

            with col_b:
                st.markdown("#### 🟠 Doctor B (Devil's Advocate)")
                st.write(final_state["doctor_b_arguments"][round_num])
                st.metric("Option",      final_state["doctor_b_letters"][round_num])
                st.metric("Diagnosis",   final_state["doctor_b_diagnoses"][round_num])
                st.metric("Confidence",  f"{final_state['doctor_b_confidences'][round_num]:.0f}%")

    # ── PDS Analysis ──────────────────────────────────────────────────────
    st.subheader("Position Drift Score Analysis")

    pds  = final_state["position_drift_score"]
    comp = final_state["pds_components"]

    left_col, right_col = st.columns([1, 1])

    with left_col:
        c1, c2 = st.columns(2)
        c3, c4 = st.columns(2)
        c1.metric("Confidence Drift",   f"{comp['confidence_drift']:.3f}",
                  help="How much numeric confidence changed")
        c2.metric("Semantic Drift",     f"{comp['semantic_drift']:.3f}",
                  help="How much argument meaning changed")
        c3.metric("Final Disagreement", f"{comp['final_disagreement']:.3f}",
                  help="Gap between agents at end")
        c4.metric(
            "PDS Score", f"{pds:.3f}",
            delta="Below threshold" if pds < pds_threshold else "Above threshold — escalating",
            delta_color="normal" if pds < pds_threshold else "inverse",
        )
        st.caption(comp["interpretation"])

    with right_col:
        needle_color = "#e74c3c" if pds >= pds_threshold else "#2ecc71"
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(pds, 3),
            number={"suffix": "", "font": {"size": 36, "color": "white"}},
            title={"text": "Position Drift Score", "font": {"size": 14, "color": "white"}},
            gauge={
                "axis": {
                    "range": [0, 0.5], "tickwidth": 1,
                    "tickcolor": "rgba(255,255,255,0.6)", "nticks": 6,
                    "tickfont": {"color": "rgba(255,255,255,0.8)"},
                },
                "bar":  {"color": needle_color, "thickness": 0.3},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0,    0.10],           "color": "rgba(46,204,113,0.25)"},
                    {"range": [0.10, pds_threshold],  "color": "rgba(241,196,15,0.25)"},
                    {"range": [pds_threshold, 0.5],   "color": "rgba(231,76,60,0.25)"},
                ],
                "threshold": {
                    "line":      {"color": "white", "width": 3},
                    "thickness": 0.75,
                    "value":     pds_threshold,
                },
            },
        ))
        gauge_fig.update_layout(
            height=240,
            margin=dict(l=20, r=20, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
        )
        st.plotly_chart(gauge_fig, use_container_width=True)

    # ── Final Verdict ─────────────────────────────────────────────────────
    st.subheader("Final Verdict")

    if not final_state["escalate_to_human"]:
        st.success(f"Final Diagnosis: **{final_state['final_diagnosis']}**")
        st.info(f"Moderator Confidence: {final_state['verdict_confidence']:.0f}% | PDS: {pds:.3f} (safe)")
    else:
        st.error("ESCALATE TO HUMAN DOCTOR")
        st.warning(final_state["escalation_reason"])

        with st.expander("Summary for Human Doctor"):
            st.text(format_escalation_summary(final_state))

    # ── Download Report ───────────────────────────────────────────────────
    st.divider()
    st.subheader("Export Report")
    pdf_bytes = generate_pdf_report(final_state, patient_case)
    st.download_button(
        label="Download PDF Report",
        data=pdf_bytes,
        file_name="meddebate_report.pdf",
        mime="application/pdf",
        type="primary",
    )
    st.caption("Structured PDF suitable for medical records — includes full transcript, PDS analysis, and final verdict.")
