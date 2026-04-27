import numpy as np
from sentence_transformers import SentenceTransformer, util


class PositionDriftScore:
    """
    ════════════════════════════════════════════════════
    POSITION DRIFT SCORE (PDS) — Novel Uncertainty Metric
    ════════════════════════════════════════════════════

    Measures how much agents' positions DRIFT during debate.

    Intuition: If a doctor's position crumbles under
    adversarial pressure → the diagnosis is uncertain.
    If it stays firm → we can trust it.

    Formula:
    PDS = 0.35 × Confidence_Drift
        + 0.40 × Semantic_Drift        ← most important
        + 0.25 × Final_Disagreement

    Range: 0.0 (very certain) → 1.0 (very uncertain)

    Thresholds:
    PDS < 0.20  → HIGH confidence   → give diagnosis
    PDS 0.20-0.50 → MEDIUM confidence → give with caution
    PDS > 0.50  → LOW confidence    → escalate to human
    ════════════════════════════════════════════════════
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.weights = {
            "confidence_drift":   0.35,
            "semantic_drift":     0.40,
            "final_disagreement": 0.25,
        }

    # ── Component 1: Confidence Drift ─────────────────
    def confidence_drift(self, conf_a: list, conf_b: list) -> float:
        """How much did numeric confidence CHANGE across rounds?"""
        std_a = np.std(conf_a) / 100.0
        std_b = np.std(conf_b) / 100.0
        return float((std_a + std_b) / 2.0)

    # ── Component 2: Semantic Drift ───────────────────
    def semantic_drift(self, args_a: list, args_b: list) -> float:
        """
        Did the MEANING of arguments change across rounds?
        Compares Round 1 embedding vs Final Round embedding.

        High cosine distance = agent changed diagnosis position
        Low cosine distance  = agent stayed consistent
        """
        if len(args_a) < 2 or len(args_b) < 2:
            return 0.0

        emb_a_first = self.embedder.encode(args_a[0],  convert_to_tensor=True)
        emb_a_last  = self.embedder.encode(args_a[-1], convert_to_tensor=True)
        emb_b_first = self.embedder.encode(args_b[0],  convert_to_tensor=True)
        emb_b_last  = self.embedder.encode(args_b[-1], convert_to_tensor=True)

        sim_a = util.cos_sim(emb_a_first, emb_a_last).item()
        sim_b = util.cos_sim(emb_b_first, emb_b_last).item()

        drift_a = 1.0 - max(sim_a, 0)
        drift_b = 1.0 - max(sim_b, 0)

        return float((drift_a + drift_b) / 2.0)

    # ── Component 3: Final Disagreement ───────────────
    def final_disagreement(self, conf_a: list, conf_b: list) -> float:
        """How far apart are agents in the FINAL round?"""
        gap = abs(conf_a[-1] - conf_b[-1]) / 100.0
        return float(gap)

    # ── Final PDS Score ───────────────────────────────
    def compute(
        self,
        doctor_a_confidences: list,
        doctor_b_confidences: list,
        doctor_a_arguments: list,
        doctor_b_arguments: list,
    ) -> tuple[float, dict]:
        """
        Compute Position Drift Score.
        Returns: (pds_score: float, components: dict)
        """
        c1 = self.confidence_drift(doctor_a_confidences, doctor_b_confidences)
        c2 = self.semantic_drift(doctor_a_arguments, doctor_b_arguments)
        c3 = self.final_disagreement(doctor_a_confidences, doctor_b_confidences)

        pds = (
            self.weights["confidence_drift"]   * c1 +
            self.weights["semantic_drift"]     * c2 +
            self.weights["final_disagreement"] * c3
        )
        pds = float(np.clip(pds, 0.0, 1.0))

        components = {
            "confidence_drift":   round(c1, 4),
            "semantic_drift":     round(c2, 4),
            "final_disagreement": round(c3, 4),
            "pds_score":          round(pds, 4),
            "interpretation":     self.interpret(pds),
            "weights_used":       self.weights,
        }

        return pds, components

    def interpret(self, pds: float) -> str:
        if pds < 0.20:
            return "HIGH CONFIDENCE — Safe to give diagnosis"
        elif pds < 0.50:
            return "MEDIUM CONFIDENCE — Give with caution"
        else:
            return "LOW CONFIDENCE — Escalate to human doctor"
