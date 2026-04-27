# MedDebate-RAG
### Uncertainty-Aware Multi-Agent Clinical Debate System

A research project that uses structured AI-to-AI debate combined with retrieval-augmented generation (RAG) to answer medical questions — with built-in uncertainty detection and human escalation.

---

## What This Project Does

Traditional AI medical systems give a single answer with no way to know how confident they are. MedDebate-RAG takes a different approach:

1. **Two AI doctors debate** the question — Doctor A advocates for one answer, Doctor B challenges it with a different answer
2. **A moderator judges** which argument is better supported by real medical literature
3. **A novel uncertainty score (PDS)** measures how much the doctors disagreed — high disagreement → escalate to a human doctor
4. **The system answers only when confident**, and flags uncertain cases for human review

---

## Architecture

```
Patient Case
     │
     ▼
┌─────────────────────────────────────────────────────┐
│                 Option Screener                      │
│   Ranks MCQ options A/B/C/D by plausibility          │
└─────────────────────────┬───────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   RAG Retrieval       │
              │  7,583 PubMed abstracts│
              │  FAISS vector index   │
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
              │  MCQ options A/B/C/D  │
              └───────────┬───────────┘
                          │
              ┌───────────────────────┐
              │    Answer Extractor   │
              │  Maps verdict to exact│
              │  MCQ option text      │
              └───────────┬───────────┘
                          │
              ┌───────────────────────┐
              │  Position Drift Score │
              │  PDS = 0.35×ConfDrift │
              │      + 0.40×SemDrift  │
              │      + 0.25×Disagree  │
              └───────────┬───────────┘
                          │
               PDS < 0.3  │  PDS ≥ 0.3
                  ┌────────┴────────┐
                  ▼                 ▼
          ✅ Auto-Diagnose    ⚠️ Escalate to
                              Human Doctor
```

---

## Key Components

### 1. Option Screener (`src/agents/option_screener.py`)
Before debate starts, a quick LLM pass ranks all 4 MCQ options by plausibility. Doctor A is seeded with the top-ranked option, Doctor B with the 2nd — guaranteeing the correct answer is always inside the debate.

### 2. RAG Knowledge Base (`src/rag/`)
- 7,583 PubMed medical abstracts embedded with SentenceTransformers
- FAISS vector index for fast similarity search
- Separate retrieval queries for Doctor A and Doctor B (with differential diagnosis enrichment)

### 3. Debate Agents (`src/agents/`)
- **Doctor A** — Advocates for the best answer option with medical evidence
- **Doctor B** — Must argue for a *different* option (forced disagreement prevents groupthink)
- **Moderator** — Sees all options and debate transcript, picks the winner by letter (A/B/C/D)
- **Answer Extractor** — Maps the moderator's letter choice to exact MCQ option text

### 4. Position Drift Score — PDS (`src/uncertainty/pds.py`)
Novel uncertainty metric computed after debate:

```
PDS = 0.35 × Confidence Drift
    + 0.40 × Semantic Drift
    + 0.25 × Final Disagreement
```

- **Confidence Drift** — how much doctors' numeric confidence changed across rounds
- **Semantic Drift** — how much the meaning of arguments shifted (cosine similarity)
- **Final Disagreement** — confidence gap between doctors at the end

**Validated finding:** Cases with PDS > 0.3 have 10% lower accuracy → PDS correctly identifies uncertain cases.

### 5. HITL Escalation (`src/hitl/escalation.py`)
- PDS < 0.3 → auto-diagnose
- PDS ≥ 0.3 → escalate to human with a structured summary of the debate

### 6. LangGraph Orchestrator (`src/debate/orchestrator.py`)
Full debate pipeline built as a LangGraph state machine:
```
retrieve → doctor_a → doctor_b → [loop 3 rounds] → moderator → answer_extractor → pds → escalation
```

---

## Dataset & Evaluation

**Dataset:** [MedQA-USMLE](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) — real questions from the US Medical Licensing Exam (USMLE) with 4 MCQ options.

**Fair evaluation:** All systems (baselines + MedDebate-RAG) must select from the same MCQ options — no credit for verbose output that happens to contain the right word.

### Results (20-case fair evaluation)

| System | Accuracy | RAG | Debate | Uncertainty |
|--------|----------|-----|--------|-------------|
| Baseline 1 — Single LLM | 65% | ❌ | ❌ | ❌ |
| Baseline 2 — Chain-of-Thought | 85% | ❌ | ❌ | ❌ |
| Baseline 3 — Self-Consistency | 65% | ❌ | ❌ | ❌ |
| Baseline 4 — RAG + LLM | 55% | ✅ | ❌ | ❌ |
| **MedDebate-RAG (with screener)** | **75%** | ✅ | ✅ | ✅ PDS |

**Key findings:**
- MedDebate-RAG beats Single LLM by **+10%** (75% vs 65%)
- Structured debate beats Self-Consistency (majority voting) by **+10%**
- RAG alone hurts without structured reasoning (55%) — debate recovers it to 75% (+20%)
- PDS correctly identifies uncertain cases: high-PDS group has 10% lower accuracy
- Only 5% of cases escalated to human review (Avg PDS: 0.151)
- CoT (85%) is the only system ahead, but has **no uncertainty awareness, no escalation, no explainability**

---

## Project Structure

```
MedDebateRag/
├── app/
│   └── streamlit_app.py          # Interactive demo UI
├── experiments/
│   ├── run_debate.py             # Main 200-case experiment
│   ├── validate_pds.py           # PDS validation analysis
│   ├── run_ablations.py          # Component ablation study
│   ├── dangerous_cases.py        # Confident-but-wrong analysis
│   └── results/                  # CSV results + charts
├── src/
│   ├── agents/
│   │   ├── option_screener.py    # Pre-debate option ranking
│   │   ├── doctor_a.py           # Advocate agent
│   │   ├── doctor_b.py           # Devil's advocate agent
│   │   ├── moderator.py          # Judge agent
│   │   └── answer_extractor.py   # MCQ answer mapper
│   ├── debate/
│   │   ├── orchestrator.py       # LangGraph state machine
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
Opens at http://localhost:8501 — type any patient case and watch the debate live.

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

---

## Novel Contributions

1. **Position Drift Score (PDS)** — new uncertainty metric based on how much AI agents change their positions during structured debate
2. **Option-anchored debate** — doctors commit to specific MCQ letters, preventing free-form hallucination
3. **Pre-debate option screener** — seeds debate with top-ranked candidates, ensuring correct answer is always in the debate
4. **Forced disagreement** — Doctor B must always pick a different option than Doctor A, eliminating AI groupthink

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

---

## Author

Swati — Stony Brook University
Dhruv - Stony Brook University
