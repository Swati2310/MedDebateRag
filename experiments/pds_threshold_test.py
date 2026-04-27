"""
Run hard clinical cases through the debate system to find the optimal PDS threshold.
Cases are designed to be genuinely ambiguous — symptoms overlap multiple diagnoses.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from dotenv import load_dotenv; load_dotenv()

from src.debate.orchestrator import run_debate

HARD_CASES = [
    {
        "label": "Hypoparathyroidism vs Hypothyroidism (hidden labs)",
        "case": """CLINICAL VIGNETTE:
A 45-year-old woman presents with fatigue, cold intolerance, weight gain,
constipation, and muscle cramps with finger tingling for 6 months.
Reflexes show delayed relaxation. Labs: TSH 2.1 mIU/L (normal),
Free T4 normal, Calcium 6.8 mg/dL (LOW), Phosphate 5.2 (HIGH), PTH 12 (LOW).

Answer options:
  A. Levothyroxine therapy
  B. Calcium and Vitamin D supplementation
  C. Cinacalcet (calcimimetic agent)
  D. Teriparatide (PTH analog)""",
        "truth": "Calcium and Vitamin D supplementation"
    },
    {
        "label": "TIA management — anticoagulation debate",
        "case": """CLINICAL VIGNETTE:
A 72-year-old man with atrial fibrillation on warfarin (INR 1.8) has
sudden right-sided weakness and aphasia lasting 40 minutes that fully resolved.
CT head: no hemorrhage. CHADS2 score: 4.

Answer options:
  A. Increase warfarin dose targeting INR 2.5-3.5
  B. Add aspirin 81mg to current warfarin
  C. Switch to dabigatran (direct thrombin inhibitor)
  D. Add clopidogrel to current warfarin therapy""",
        "truth": "Switch to dabigatran (direct thrombin inhibitor)"
    },
    {
        "label": "Intoxication — overlapping presentations",
        "case": """CLINICAL VIGNETTE:
A 22-year-old man is brought unresponsive to the ED from a party.
He has nystagmus, ataxia, combative behavior when aroused, slurred speech.
Vitals: HR 110, BP 130/80, RR 14. Pupils are mid-dilated, sluggishly reactive.
Urine tox screen is pending.

Answer options:
  A. Opioid intoxication — give naloxone
  B. Phencyclidine (PCP) intoxication — supportive care
  C. Benzodiazepine intoxication — give flumazenil
  D. Alcohol intoxication — supportive care and monitoring""",
        "truth": "Phencyclidine (PCP) intoxication — supportive care"
    },
    {
        "label": "Autoimmune overlap — lupus vs mixed connective tissue",
        "case": """CLINICAL VIGNETTE:
A 32-year-old woman presents with Raynaud's phenomenon, puffy swollen hands,
dysphagia, myalgia, and mild arthritis for 8 months. ANA is positive (1:640),
anti-U1 RNP antibody strongly positive, anti-dsDNA negative, anti-Sm negative.
Complement levels normal.

Answer options:
  A. Systemic lupus erythematosus (SLE)
  B. Mixed connective tissue disease (MCTD)
  C. Systemic sclerosis (scleroderma)
  D. Polymyositis""",
        "truth": "Mixed connective tissue disease (MCTD)"
    },
    {
        "label": "Chest pain — ACS vs PE vs aortic dissection",
        "case": """CLINICAL VIGNETTE:
A 58-year-old man with hypertension presents with sudden severe tearing
chest pain radiating to the back. BP: right arm 180/100, left arm 145/85.
ECG: normal sinus rhythm, no ST changes. Troponin: mildly elevated.
CXR shows widened mediastinum.

Answer options:
  A. Activate cath lab for primary PCI (STEMI protocol)
  B. Start IV heparin for suspected PE
  C. Emergency CT angiography for aortic dissection
  D. Administer thrombolytics for massive PE""",
        "truth": "Emergency CT angiography for aortic dissection"
    },
    {
        "label": "Pediatric rash — Kawasaki vs other febrile illness",
        "case": """CLINICAL VIGNETTE:
A 4-year-old boy has 6 days of fever unresponsive to antibiotics,
bilateral non-purulent conjunctivitis, strawberry tongue, cracked red lips,
polymorphous rash on trunk, and cervical lymphadenopathy >1.5cm.
WBC 18,000, CRP elevated, Echo: coronary arteries at upper limit of normal.

Answer options:
  A. IV antibiotics for bacterial sepsis
  B. IVIG 2g/kg + aspirin (Kawasaki disease treatment)
  C. Oral steroids for systemic JIA
  D. Supportive care for viral illness""",
        "truth": "IVIG 2g/kg + aspirin (Kawasaki disease treatment)"
    },
    {
        "label": "Psychiatry — first episode psychosis medication choice",
        "case": """CLINICAL VIGNETTE:
A 24-year-old man with first-episode schizophrenia, BMI 31, fasting glucose
108 mg/dL (pre-diabetic), and family history of diabetes presents for
medication initiation. He has positive symptoms (hallucinations, delusions)
and moderate negative symptoms.

Answer options:
  A. Olanzapine (high metabolic risk but effective for negative symptoms)
  B. Risperidone (moderate metabolic risk, good evidence)
  C. Aripiprazole (low metabolic risk, weight neutral)
  D. Clozapine (reserved for treatment-resistant cases)""",
        "truth": "Aripiprazole (low metabolic risk, weight neutral)"
    },
    {
        "label": "Renal — AKI cause differentiation",
        "case": """CLINICAL VIGNETTE:
A 68-year-old man post-CABG day 2 has rising creatinine (1.2 → 3.4 mg/dL),
urine output 15 mL/hr despite 2L IV fluids. UA: muddy brown casts,
FENa 3.2%. BP 95/60. Recent vancomycin and gentamicin use.
No evidence of obstruction on bedside ultrasound.

Answer options:
  A. Pre-renal AKI — aggressive IV fluid resuscitation
  B. Aminoglycoside-induced ATN — discontinue nephrotoxins, supportive care
  C. Contrast-induced nephropathy — N-acetylcysteine
  D. Post-renal AKI — bladder catheterization""",
        "truth": "Aminoglycoside-induced ATN — discontinue nephrotoxins, supportive care"
    },
]


def run_tests():
    results = []
    print(f"Running {len(HARD_CASES)} hard cases...\n")
    print(f"{'#':<3} {'Label':<45} {'PDS':>6} {'ConfD':>6} {'SemD':>6} {'Disagr':>7} {'Correct'}")
    print("-" * 90)

    for idx, c in enumerate(HARD_CASES):
        state = run_debate(c["case"], c["truth"], max_rounds=3)
        pds   = state["position_drift_score"]
        comp  = state["pds_components"]
        pred  = state["final_diagnosis"] or ""
        truth = c["truth"].lower()
        correct = truth in pred.lower() or any(
            w in pred.lower() for w in truth.split() if len(w) > 4
        )

        print(f"{idx:<3} {c['label'][:44]:<45} {pds:>6.3f} {comp['confidence_drift']:>6.3f} "
              f"{comp['semantic_drift']:>6.3f} {comp['final_disagreement']:>7.3f} "
              f"{'OK' if correct else 'WRONG'}")

        results.append({
            "label": c["label"],
            "pds": pds,
            "conf_drift": comp["confidence_drift"],
            "sem_drift": comp["semantic_drift"],
            "disagreement": comp["final_disagreement"],
            "correct": correct,
            "pred": pred[:80],
            "truth": c["truth"][:80],
            "a_letters": state.get("doctor_a_letters", []),
            "b_letters": state.get("doctor_b_letters", []),
        })

    print("\n" + "="*60)
    print("PDS THRESHOLD ANALYSIS")
    print("="*60)

    import numpy as np
    pds_scores = [r["pds"] for r in results]
    correct_mask = [r["correct"] for r in results]

    print(f"\nPDS distribution across {len(results)} hard cases:")
    print(f"  Min:    {min(pds_scores):.3f}")
    print(f"  Max:    {max(pds_scores):.3f}")
    print(f"  Mean:   {sum(pds_scores)/len(pds_scores):.3f}")
    print(f"  Median: {sorted(pds_scores)[len(pds_scores)//2]:.3f}")

    print(f"\nAccuracy by PDS bucket:")
    for lo, hi in [(0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 1.0)]:
        bucket = [r for r in results if lo <= r["pds"] < hi]
        if bucket:
            acc = sum(r["correct"] for r in bucket) / len(bucket) * 100
            print(f"  PDS [{lo:.1f}-{hi:.1f}): {len(bucket):2d} cases → {acc:.0f}% accuracy")

    print(f"\nThreshold recommendations:")
    for thresh in [0.10, 0.15, 0.20, 0.25, 0.30]:
        escalated = sum(1 for p in pds_scores if p >= thresh)
        below     = [r for r in results if r["pds"] < thresh]
        above     = [r for r in results if r["pds"] >= thresh]
        acc_below = sum(r["correct"] for r in below)/len(below)*100 if below else 0
        acc_above = sum(r["correct"] for r in above)/len(above)*100 if above else 0
        print(f"  Threshold {thresh:.2f}: escalate {escalated}/{len(results)} cases "
              f"| auto-acc={acc_below:.0f}% | escalated-acc={acc_above:.0f}%")

    print(f"\nDetailed results:")
    for r in results:
        flag = "⚠ ESCALATE" if r["pds"] >= 0.20 else "✓ auto"
        print(f"  [{flag}] PDS={r['pds']:.3f} | {'OK' if r['correct'] else 'WRONG'} | {r['label'][:50]}")
        print(f"           A={r['a_letters']} B={r['b_letters']}")
        print(f"           Pred:  {r['pred'][:70]}")
        print(f"           Truth: {r['truth'][:70]}")

    return results


if __name__ == "__main__":
    run_tests()
