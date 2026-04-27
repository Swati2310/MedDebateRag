import os
import sys

import streamlit as st
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
load_dotenv()

from src.data.load_ddxplus import format_patient_case, load_ddxplus
from src.debate.orchestrator import run_debate
from src.hitl.escalation import format_escalation_summary

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
    max_rounds     = st.slider("Debate Rounds", 1, 5, 3)
    pds_threshold  = st.slider("Escalation Threshold", 0.1, 0.9, 0.3,
                                help="PDS above this → escalate to human")
    use_finetuned  = st.checkbox("Use Fine-tuned Moderator (Llama)", value=False)

    st.markdown("---")
    st.markdown("**PDS Thresholds:**")
    st.markdown("🟢 < 0.2 — High Confidence")
    st.markdown("🟡 0.2-0.3 — Medium Confidence")
    st.markdown("🔴 > 0.3 — Escalate to Human")

# ── Patient Case Input ────────────────────────────────────────────────────────
st.subheader("Patient Case Input")

col1, col2 = st.columns([2, 1])

with col1:
    patient_case = st.text_area(
        "Enter patient symptoms, age, medical history:",
        value=st.session_state.get("patient_case", ""),
        placeholder=(
            "Example: 45-year-old male presenting with chest pain, shortness of breath, "
            "sweating. Elevated troponin (2.5 ng/mL). History of hypertension and smoking."
        ),
        height=150,
    )

with col2:
    st.markdown("**Or load a sample case:**")
    if st.button("Load Random DDxPlus Case"):
        df = load_ddxplus()
        sample = df.sample(1).iloc[0]
        st.session_state["patient_case"]  = format_patient_case(sample)
        st.session_state["ground_truth"] = sample["pathology"]
        st.rerun()

    if "ground_truth" in st.session_state:
        st.success(f"Ground Truth: {st.session_state['ground_truth']}")

# ── Run Debate ────────────────────────────────────────────────────────────────
if st.button("Start Clinical Debate", type="primary", disabled=not patient_case):

    with st.spinner("Retrieving medical literature from PubMed..."):
        final_state = run_debate(
            patient_case,
            ground_truth="",
            max_rounds=max_rounds,
            use_finetuned=use_finetuned,
        )

    # ── Debate Rounds ─────────────────────────────────────────────────────
    st.subheader("Debate Transcript")
    n_rounds = len(final_state["doctor_a_arguments"])

    for round_num in range(n_rounds):
        with st.expander(f"Round {round_num + 1}", expanded=(round_num == n_rounds - 1)):
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("#### Doctor A (Advocate)")
                st.write(final_state["doctor_a_arguments"][round_num])
                st.metric("Diagnosis",   final_state["doctor_a_diagnoses"][round_num])
                st.metric("Confidence",  f"{final_state['doctor_a_confidences'][round_num]:.0f}%")

            with col_b:
                st.markdown("#### Doctor B (Devil's Advocate)")
                st.write(final_state["doctor_b_arguments"][round_num])
                st.metric("Diagnosis",   final_state["doctor_b_diagnoses"][round_num])
                st.metric("Confidence",  f"{final_state['doctor_b_confidences'][round_num]:.0f}%")

    # ── PDS Analysis ──────────────────────────────────────────────────────
    st.subheader("Position Drift Score Analysis")

    pds  = final_state["position_drift_score"]
    comp = final_state["pds_components"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Confidence Drift",   f"{comp['confidence_drift']:.3f}",   help="How much numeric confidence changed")
    c2.metric("Semantic Drift",     f"{comp['semantic_drift']:.3f}",     help="How much argument meaning changed")
    c3.metric("Final Disagreement", f"{comp['final_disagreement']:.3f}", help="Gap between agents at end")
    c4.metric(
        "PDS Score", f"{pds:.3f}",
        delta="Below threshold" if pds < pds_threshold else "Above threshold — escalating",
        delta_color="normal" if pds < pds_threshold else "inverse",
    )

    st.progress(pds, text=f"PDS: {pds:.3f} | {comp['interpretation']}")

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
