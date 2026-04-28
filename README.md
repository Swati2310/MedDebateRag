# MedDebate-RAG
### Uncertainty-Aware Multi-Agent Clinical Debate System

A research project that uses structured AI-to-AI debate combined with retrieval-augmented generation (RAG) to support clinical diagnosis — with built-in uncertainty detection and human escalation. Works in **clinical free-text mode** (real-world) and **benchmark MCQ mode** (evaluation).

---

## What This Project Does

Traditional AI medical systems give a single answer with no way to know how confident they are. MedDebate-RAG takes a different approach:

1. **Accept free-text patient cases** — no multiple-choice options needed in clinical mode
2. **Generate differential diagnoses** from PubMed evidence using RAG
3. **Two AI doctors debate** — Doctor A advocates for one diagnosis, Doctor B challenges it
4. **Each doctor retrieves targeted evidence** — Doctor A searches PubMed for *their specific* diagnosis, Doctor B for *theirs*
5. **A moderator judges** which argument is better supported by real medical literature
6. **A novel uncertainty score (PDS)** measures disagreement — high disagreement → escalate to a human doctor
7. **Export a structured PDF report** suitable for medical records

---

## Architecture

### Clinical Mode (Streamlit — Real-World Use)

```
Free-Text Patient Case
         │
         ▼
┌─────────────────────────────────────────────────────┐
│          RAG Retrieval (Initial)                     │
│   7,583 PubMed abstracts · FAISS vector index        │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│         Differential Diagnosis Generator             │
│   Generates top 4 candidate diagnoses from evidence  │
│   → formats as A / B / C / D options                 │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────┐
│                 Option Screener                      │
│   Ranks A/B/C/D by plausibility · seeds the debate   │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  Targeted RAG Retrieval│
              │  Doctor A → papers for │
              │    their diagnosis     │
              │  Doctor B → papers for │
              │    their diagnosis     │
              └───────────┬───────────┘
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
    ┌───────────────┐           ┌───────────────┐
    │   Doctor A    │           │   Doctor B    │
    │  (Advocate)   │◄─────────►│ (Devil's      │
    │  argues top   │  3 rounds │  Advocate)    │
    │  ranked option│           │  argues #2    │
    └───────┬───────┘           └───────┬───────┘
            └─────────────┬─────────────┘
                          ▼
              ┌───────────────────────┐
              │      Moderator        │
              │  Picks winner from    │
              │  debate transcript    │
              └───────────┬───────────┘
                          │
              ┌───────────────────────┐
              │    Answer Extractor   │
              │  Maps verdict to exact│
              │  diagnosis text       │
              └───────────┬───────────┘
                          │
              ┌───────────────────────┐
              │  Position Drift Score │
              │  PDS = 0.35×ConfDrift │
              │      + 0.40×SemDrift  │
              │      + 0.25×Disagree  │
              └───────────┬───────────┘
                          │
               PDS < 0.15 │  PDS >= 0.15
                  ┌────────┴────────┐
                  ▼                 ▼
          ✅ Auto-Diagnose    ⚠️ Escalate to
          + PDF Export         Human Doctor
                               + PDF Export
```

### Benchmark Mode (Evaluation — MCQ datasets)

Same pipeline but skips the differential generator — MCQ options (A/B/C/D) are taken directly from the dataset. Used for all accuracy comparisons.

---

## Key Components

### 1. Differential Diagnosis Generator (`src/agents/differential_generator.py`)
**New — clinical mode only.** For free-text patient cases (no MCQ options), this module uses RAG-retrieved PubMed abstracts to generate the top 4 candidate diagnoses, which are then formatted as A/B/C/D options and fed into the standard debate pipeline. This makes the system usable in real-world settings where options are never provided upfront.

### 2. Option Screener (`src/agents/option_screener.py`)
Before debate starts, a quick LLM pass ranks all 4 options by plausibility. Doctor A is seeded with the top-ranked option, Doctor B with the 2nd — guaranteeing the debate always covers the most likely candidates.

### 3. Targeted RAG Knowledge Base (`src/rag/`)
- 7,583 PubMed medical abstracts embedded with SentenceTransformers
- FAISS vector index for fast similarity search
- **Targeted retrieval:** after option screening, Doctor A queries PubMed specifically for *their assigned diagnosis*, Doctor B for *theirs* — each agent gets focused, relevant evidence rather than identical generic results

### 4. Debate Agents (`src/agents/`)
- **Doctor A** — Advocates for the best answer option with targeted medical evidence
- **Doctor B** — Must argue for a *different* option (forced disagreement prevents groupthink)
- **Moderator** — Sees all options and full debate transcript, picks the winner
- **Answer Extractor** — Maps the moderator's choice to exact diagnosis text

### 5. Position Drift Score — PDS (`src/uncertainty/pds.py`)
Novel uncertainty metric computed after debate:

```
PDS = 0.35 × Confidence Drift
    + 0.40 × Semantic Drift
    + 0.25 × Final Disagreement
```

- **Confidence Drift** — how much doctors' numeric confidence changed across rounds
- **Semantic Drift** — how much the meaning of arguments shifted (cosine similarity)
- **Final Disagreement** — confidence gap between doctors at the end

**Validated finding:** Cases with PDS > 0.15 correlate with lower accuracy — PDS correctly identifies uncertain cases. Threshold calibrated on a held-out hard-case set.

### 6. HITL Escalation (`src/hitl/escalation.py`)
- PDS < 0.15 → auto-diagnose
- PDS >= 0.15 → escalate to human with a structured summary of the debate

### 7. LangGraph Orchestrator (`src/debate/orchestrator.py`)
Full debate pipeline built as a LangGraph state machine:
```
retrieve → doctor_a → doctor_b → [loop N rounds] → moderator → answer_extractor → pds → escalation
```

### 8. PDF Report Generator (`app/report_generator.py`)
Generates a structured, printable PDF report including full debate transcript, PDS analysis, and final verdict — suitable for medical records. Built with fpdf2.

---

## Interactive UI (`app/streamlit_app.py`)

The Streamlit app runs in **clinical free-text mode** — paste any patient case, no multiple-choice options needed.

**Features:**
- **Live debate animation** — each pipeline step (retrieval, differential generation, debate rounds, PDS) streams results in real time using `st.status()`
- **PDS gauge meter** — interactive Plotly gauge showing uncertainty level with color-coded zones (green / yellow / red)
- **Adjustable thresholds** — sidebar sliders for debate rounds (1–5) and escalation threshold (0.05–0.5)
- **Load sample case** — pulls a random DDxPlus case for instant testing
- **Full transcript viewer** — expandable per-round panels showing each doctor's argument, option, and confidence
- **PDF export** — download a structured report of the full debate

---

## Dataset & Evaluation

**Dataset:** [MedQA-USMLE](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) — real questions from the US Medical Licensing Exam (USMLE) with 4 MCQ options.

**Fair evaluation:** All systems must select from the same MCQ options — no credit for verbose output that happens to contain the right word.

### Results (200-case fair evaluation — full benchmark)

| System | Accuracy | RAG | Debate | Uncertainty |
|--------|----------|-----|--------|-------------|
| Baseline 1 — Single LLM | 83.5% | ❌ | ❌ | ❌ |
| Baseline 2 — Chain-of-Thought | 83.0% | ❌ | ❌ | ❌ |
| Baseline 3 — Self-Consistency | 82.5% | ❌ | ❌ | ❌ |
| Baseline 4 — RAG + LLM | 79.5% | ✅ | ❌ | ❌ |
| **MedDebate-RAG (with screener)** | **85.0%** | ✅ | ✅ | ✅ PDS |

**Key findings:**
- MedDebate-RAG beats every baseline — **#1 across all 5 systems**
- **+1.5%** over Single LLM (85.0% vs 83.5%)
- **+2.0%** over Chain-of-Thought (85.0% vs 83.0%) — debate outperforms CoT at scale
- **+5.5%** over RAG alone (85.0% vs 79.5%) — structured debate recovers what naive RAG loses
- Average PDS = 0.138 | PDS range: 0.041 – 0.332
- **32.5% of cases escalated** to human review (PDS > 0.15) — system is selectively confident
- On the 67.5% of cases answered automatically, the system is most reliable
- All baselines evaluated under identical MCQ-constrained conditions

---

## Project Structure

```
MedDebateRag/
├── app/
│   ├── streamlit_app.py          # Interactive demo UI (clinical free-text mode)
│   └── report_generator.py       # PDF report generation (fpdf2)
├── experiments/
│   ├── run_debate.py             # Main 200-case benchmark experiment
│   ├── validate_pds.py           # PDS validation analysis
│   ├── run_ablations.py          # Component ablation study
│   ├── dangerous_cases.py        # Confident-but-wrong analysis
│   └── results/                  # CSV results + charts
├── src/
│   ├── agents/
│   │   ├── option_screener.py    # Pre-debate option ranking
│   │   ├── differential_generator.py  # Clinical mode: generate A/B/C/D from free text
│   │   ├── doctor_a.py           # Advocate agent
│   │   ├── doctor_b.py           # Devil's advocate agent
│   │   ├── moderator.py          # Judge agent
│   │   └── answer_extractor.py   # Answer mapper
│   ├── debate/
│   │   ├── orchestrator.py       # LangGraph state machine + targeted retrieval
│   │   └── state.py              # Typed debate state
│   ├── rag/
│   │   ├── knowledge_base.py     # FAISS index + embeddings
│   │   └── retriever.py          # Top-k retrieval
│   ├── uncertainty/
│   │   └── pds.py                # Position Drift Score
│   ├── hitl/
│   │   └── escalation.py         # Human escalation logic
│   ├── evaluation/
│   │   └── baselines.py          # 4 baseline systems
│   └── llm_client.py             # Gemini API client (retry logic)
├── setup_knowledge_base.py       # Build FAISS index from PubMed
├── test_case.py                  # CLI test for custom cases
└── requirements.txt
```

---

## Setup & Running

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up environment variables
Create a `.env` file:
```
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_CHAT_MODEL=gemini-2.5-flash-lite
```

### 3. Build the knowledge base
```bash
python setup_knowledge_base.py
```
Downloads 7,583 PubMed abstracts and builds the FAISS index (~5 minutes).

### 4. Run the interactive demo
```bash
streamlit run app/streamlit_app.py
```
Opens at http://localhost:8501 — paste any free-text patient case and watch the debate live.

### 5. Test a custom case from terminal
```bash
python test_case.py "65-year-old male with crushing chest pain, ST elevation in leads II, III, aVF"
```

### 6. Run the full benchmark experiment
```bash
python -m experiments.run_debate --n 200
```

---

## Edge Cases & Limitations

| Type | Example | Why It Fails |
|------|---------|-------------|
| Confident but wrong | Both doctors agree on wrong answer | PDS stays low — no disagreement to detect (AI groupthink) |
| Equivalent drugs | Ibuprofen vs Indomethacin (both treat pericarditis) | Medically valid but USMLE accepts only one |
| Ethics/legal questions | "What should the resident do?" | System trained for clinical diagnosis, not medical ethics |
| Numerical calculations | Lab value computations | Small arithmetic errors with high confidence |
| Rare diseases | Very uncommon presentations | PubMed evidence may be sparse; differential generator may miss the true diagnosis |

---

## Novel Contributions

1. **Position Drift Score (PDS)** — new uncertainty metric based on how much AI agents change their positions during structured debate; threshold calibrated at 0.15
2. **Clinical free-text mode** — differential diagnosis generator turns any free-text patient case into a debatable set of options using PubMed evidence; no MCQ options required
3. **Targeted per-doctor retrieval** — each doctor retrieves PubMed evidence specifically for their assigned diagnosis rather than sharing a generic query
4. **Option-anchored debate** — doctors commit to specific answer letters, preventing free-form hallucination
5. **Pre-debate option screener** — seeds debate with top-ranked candidates, ensuring correct answer is always inside the debate
6. **Forced disagreement** — Doctor B must always pick a different option than Doctor A, eliminating AI groupthink

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Google Gemini 2.5 Flash Lite |
| Debate Orchestration | LangGraph |
| Vector Search | FAISS |
| Embeddings | SentenceTransformers (`all-MiniLM-L6-v2`) |
| Dataset | MedQA-USMLE (GBaker/MedQA-USMLE-4-options) |
| Knowledge Base | PubMed abstracts (7,583) |
| UI | Streamlit |
| Visualization | Plotly (PDS gauge meter) |
| PDF Export | fpdf2 |

---

## Author

Swati — Stony Brook University
Dhruv — Stony Brook University
