"""
PDF Report Generator for MedDebate-RAG clinical debate results.
Produces a structured, printable report suitable for medical records.
"""

from datetime import datetime
from fpdf import FPDF


class DebateReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(30, 60, 120)
        self.cell(0, 8, "MedDebate-RAG  |  Clinical Debate Report", align="L")
        self.set_font("Helvetica", "", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, datetime.now().strftime("%Y-%m-%d  %H:%M"), align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(180, 180, 200)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(150, 150, 150)
        self.cell(0, 5,
            "AI-assisted second opinion only. Not a substitute for licensed medical judgement.  "
            f"Page {self.page_no()}",
            align="C",
        )


def _safe(text, max_chars=2000):
    """Truncate and strip non-latin characters that fpdf can't render."""
    text = str(text or "")[:max_chars]
    return text.encode("latin-1", errors="replace").decode("latin-1")


def generate_pdf_report(state: dict, patient_case: str) -> bytes:
    pdf = DebateReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(12, 15, 12)

    # ── Section helper ────────────────────────────────────────────────────────
    def section_title(title):
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_fill_color(230, 235, 245)
        pdf.set_text_color(20, 40, 100)
        pdf.cell(0, 7, f"  {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(2)

    def body(text, size=9):
        pdf.set_font("Helvetica", "", size)
        pdf.set_text_color(30, 30, 30)
        pdf.multi_cell(0, 5, _safe(text))
        pdf.ln(2)

    def kv(key, value, bold_val=False):
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(60, 60, 60)
        pdf.cell(0, 5, _safe(f"{key}:"), new_x="LMARGIN", new_y="NEXT")
        if bold_val:
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(20, 80, 20)
        else:
            pdf.set_font("Helvetica", "", 9)
            pdf.set_text_color(30, 30, 30)
        pdf.set_x(22)
        pdf.multi_cell(0, 5, _safe(str(value)))
        pdf.ln(1)

    # ── 1. Patient Case ───────────────────────────────────────────────────────
    section_title("1.  Patient Case")
    body(patient_case, size=8)

    # ── 2. Debate Transcript ──────────────────────────────────────────────────
    section_title("2.  Debate Transcript")
    n_rounds = len(state.get("doctor_a_arguments", []))
    for i in range(n_rounds):
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(0, 60, 120)
        pdf.cell(0, 6, f"Round {i + 1}", new_x="LMARGIN", new_y="NEXT")

        # Doctor A
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(0, 100, 0)
        a_letter = state.get("doctor_a_letters", [])[i] if i < len(state.get("doctor_a_letters", [])) else "?"
        a_conf   = state.get("doctor_a_confidences", [])[i] if i < len(state.get("doctor_a_confidences", [])) else 0
        pdf.cell(0, 5, f"  Doctor A  |  Option {a_letter}  |  Confidence {a_conf:.0f}%", new_x="LMARGIN", new_y="NEXT")
        body(state["doctor_a_arguments"][i][:800])

        # Doctor B
        if i < len(state.get("doctor_b_arguments", [])):
            pdf.set_font("Helvetica", "B", 8)
            pdf.set_text_color(150, 50, 0)
            b_letter = state.get("doctor_b_letters", [])[i] if i < len(state.get("doctor_b_letters", [])) else "?"
            b_conf   = state.get("doctor_b_confidences", [])[i] if i < len(state.get("doctor_b_confidences", [])) else 0
            pdf.cell(0, 5, f"  Doctor B  |  Option {b_letter}  |  Confidence {b_conf:.0f}%", new_x="LMARGIN", new_y="NEXT")
            body(state["doctor_b_arguments"][i][:800])

        pdf.ln(1)

    # ── 3. PDS Analysis ───────────────────────────────────────────────────────
    section_title("3.  Position Drift Score (PDS) Analysis")
    pds  = state.get("position_drift_score", 0)
    comp = state.get("pds_components", {})

    kv("Confidence Drift",   f"{comp.get('confidence_drift', 0):.3f}  (how much numeric confidence changed)")
    kv("Semantic Drift",     f"{comp.get('semantic_drift', 0):.3f}  (how much argument meaning shifted)")
    kv("Final Disagreement", f"{comp.get('final_disagreement', 0):.3f}  (gap between doctors at end)")
    kv("PDS Score",          f"{pds:.3f}  (range 0.0 - 1.0)")
    kv("Interpretation",     comp.get("interpretation", ""))
    pdf.ln(2)

    # PDS bar (visual)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 4, "PDS Scale:  0.0 ---- 0.15 (threshold) ------------ 1.0", new_x="LMARGIN", new_y="NEXT")
    bar_width = 176
    filled = int(min(pds, 1.0) * bar_width)
    pdf.set_fill_color(200, 220, 200)
    pdf.rect(12, pdf.get_y(), bar_width, 4, style="F")
    color = (200, 60, 60) if pds >= 0.15 else (60, 160, 60)
    pdf.set_fill_color(*color)
    pdf.rect(12, pdf.get_y(), filled, 4, style="F")
    pdf.ln(7)

    # ── 4. Final Verdict ──────────────────────────────────────────────────────
    section_title("4.  Final Verdict")
    escalated = state.get("escalate_to_human", False)

    if not escalated:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(20, 120, 20)
        pdf.cell(0, 7, f"  DIAGNOSIS:  {_safe(state.get('final_diagnosis', ''))}", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        kv("Moderator Verdict",    state.get("moderator_verdict", ""))
        kv("Verdict Confidence",   f"{state.get('verdict_confidence', 0):.0f}%")
        kv("PDS",                  f"{pds:.3f}  (below threshold — safe to auto-diagnose)")
    else:
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(180, 30, 30)
        pdf.cell(0, 7, "  ACTION REQUIRED: ESCALATE TO HUMAN DOCTOR", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
        body(state.get("escalation_reason", ""))
        kv("PDS", f"{pds:.3f}  (above 0.15 threshold — uncertain case)")

    pdf.ln(4)

    # ── 5. Disclaimer ─────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(130, 130, 130)
    pdf.multi_cell(0, 4,
        "This report was generated by MedDebate-RAG, an AI-assisted clinical decision support tool. "
        "It is intended as a second opinion aid only and must be reviewed by a licensed physician "
        "before any clinical decision is made. Not for direct patient use."
    )

    return bytes(pdf.output())
