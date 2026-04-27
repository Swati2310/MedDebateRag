# MedDebate-RAG: Final Implementation Plan
### Uncertainty-Aware Multi-Agent Clinical Debate with RAG + PDS + HITL

---

## 🎯 Project Identity

| Field | Detail |
|---|---|
| **Project Title** | MedDebate-RAG: Uncertainty-Aware Multi-Agent Clinical Reasoning with Human-in-the-Loop Escalation |
| **Course** | Frontiers of LLMs |
| **Novel Contribution** | Position Drift Score (PDS) — a new uncertainty metric measuring agent position stability under adversarial debate pressure |
| **Pipeline** | Debate + RAG + PDS (UQ) + HITL — unified for clinical differential diagnosis |
| **Fine-Tuning** | QLoRA fine-tuning of Moderator agent on debate transcripts |

---

## 💡 One Line Summary

> *"Two AI doctors debate a diagnosis using real medical knowledge. A fine-tuned moderator measures how much their positions drift (PDS). If drift is too high — it escalates to a human doctor instead of guessing."*

---

## 🏗️ Complete System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   MedDebate-RAG Pipeline                     │
│                                                              │
│  Patient Case (DDxPlus / MedQA)                             │
│         ↓                                                    │
│  ┌──────────────────────┐                                    │
│  │   RAG Retriever      │  ← Component 1: RAG               │
│  │   PubMed + Guidelines│    Grounds agents in real          │
│  └──────────┬───────────┘    medical knowledge               │
│             ↓                                                │
│  ┌──────────────────────────────────┐                        │
│  │         DEBATE ARENA             │  ← Component 2: DEBATE │
│  │                                  │                        │
│  │  🔵 Doctor A  ←→  🔴 Doctor B   │                        │
│  │  (Advocate)      (Devil's Advoc.)│                        │
│  │  GPT-4o mini     GPT-4o mini     │                        │
│  │       ↕               ↕         │                        │
│  │  Round 1, 2, 3 (3 Debate Rounds) │                        │
│  └──────────────┬───────────────────┘                        │
│                 ↓                                            │
│  ┌──────────────────────────────────┐                        │
│  │   ⚖️ MODERATOR AGENT             │  ← Fine-tuned with     │
│  │   (QLoRA Fine-tuned Llama 3.1 8B)│    QLoRA on debate     │
│  │                                  │    transcripts         │
│  │   Reads full debate transcript   │                        │
│  │   Gives structured verdict       │                        │
│  └──────────────┬───────────────────┘                        │
│                 ↓                                            │
│  ┌──────────────────────────────────┐                        │
│  │   📐 PDS CALCULATOR              │  ← Component 3: UQ+PDS │
│  │                                  │    YOUR NOVEL METRIC   │
│  │   C1: Confidence Drift (0.35w)   │                        │
│  │   C2: Semantic Drift   (0.40w)   │                        │
│  │   C3: Final Disagreement(0.25w)  │                        │
│  │                                  │                        │
│  │   PDS = weighted combination     │                        │
│  └──────────────┬───────────────────┘                        │
│                 ↓                                            │
│  ┌──────────────────────────────────┐                        │
│  │   🚨 HITL ESCALATION             │  ← Component 4: HITL   │
│  │                                  │                        │
│  │   PDS < 0.5 → Final Diagnosis ✅ │                        │
│  │   PDS > 0.5 → Escalate Human 🚨  │                        │
│  └──────────────────────────────────┘                        │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔑 The 5 Core Components

| # | Component | Technology | Novel? |
|---|---|---|---|
| 1 | RAG | FAISS + PubMed + SentenceTransformers | ❌ Exists |
| 2 | Multi-Agent Debate | LangGraph + GPT-4o mini | ❌ Exists |
| 3 | PDS (Uncertainty) | NumPy + SentenceTransformers | ✅ YOUR INVENTION |
| 4 | HITL Escalation | Python logic driven by PDS | ❌ Exists |
| 5 | Fine-tuned Moderator | QLoRA on Llama 3.1 8B | ✅ Novel application |

> **The combination of all 5 applied to clinical diagnosis = novel pipeline**

---

## 📁 Project Folder Structure

```
medebate-rag/
│
├── data/
│   ├── raw/                          # DDxPlus, MedQA raw files
│   ├── processed/                    # Cleaned + formatted patient cases
│   └── knowledge_base/               # PubMed abstracts for RAG
│       ├── pubmed_abstracts.json     # Downloaded PubMed docs
│       ├── faiss.index               # Built FAISS index
│       └── documents.pkl             # Serialized doc store
│
├── src/
│   ├── data/
│   │   ├── load_ddxplus.py           # Load + format DDxPlus dataset
│   │   └── load_medqa.py             # Load MedQA dataset
│   │
│   ├── rag/
│   │   ├── knowledge_base.py         # PubMed loader + FAISS builder
│   │   ├── retriever.py              # Top-k semantic retrieval
│   │   └── embeddings.py             # SentenceTransformer setup
│   │
│   ├── agents/
│   │   ├── doctor_a.py               # Advocate agent (GPT-4o mini)
│   │   ├── doctor_b.py               # Devil's Advocate (GPT-4o mini)
│   │   └── moderator.py              # Moderator (fine-tuned Llama)
│   │
│   ├── debate/
│   │   ├── state.py                  # LangGraph DebateState schema
│   │   └── orchestrator.py           # Full debate loop (LangGraph)
│   │
│   ├── uncertainty/
│   │   ├── pds.py                    # Position Drift Score (NOVEL)
│   │   ├── confidence_drift.py       # Component 1
│   │   ├── semantic_drift.py         # Component 2
│   │   └── disagreement.py           # Component 3
│   │
│   ├── finetuning/
│   │   ├── generate_training_data.py # GPT-4o → debate transcripts
│   │   ├── finetune_moderator.py     # QLoRA fine-tuning script
│   │   └── evaluate_moderator.py    # Compare fine-tuned vs GPT-4o
│   │
│   ├── evaluation/
│   │   ├── baselines.py              # All 4 baseline methods
│   │   ├── metrics.py                # Accuracy, calibration, AUROC
│   │   └── ragas_eval.py             # RAG faithfulness scores
│   │
│   └── hitl/
│       └── escalation.py             # PDS → escalation decision
│
├── experiments/
│   ├── run_baselines.py              # Run all 4 baselines
│   ├── run_debate.py                 # Run full debate system
│   ├── run_ablations.py              # All ablation studies
│   ├── validate_pds.py               # PDS validation experiment
│   ├── dangerous_cases.py            # Key safety experiment
│   └── results/                      # All output CSVs + plots
│
├── notebooks/
│   ├── 01_data_exploration.ipynb     # Explore DDxPlus dataset
│   ├── 02_rag_setup.ipynb            # Build + test RAG
│   ├── 03_debate_prototype.ipynb     # Test single debate
│   ├── 04_pds_development.ipynb      # Develop + validate PDS
│   ├── 05_finetuning.ipynb           # QLoRA fine-tuning notebook
│   └── 06_results_analysis.ipynb     # Final results + plots
│
├── app/
│   └── streamlit_app.py              # Demo UI
│
├── requirements.txt
├── .env
└── README.md
```

---

## ⚙️ Environment Setup

### Step 0: Create .env File
```
OPENAI_API_KEY=your_openai_key_here
WANDB_API_KEY=your_wandb_key_here      # free at wandb.ai
HF_TOKEN=your_huggingface_token_here   # free at huggingface.co
```

### Step 0: Install All Dependencies
```bash
# Core LLM & Agents
pip install langchain==0.2.0
pip install langgraph==0.1.0
pip install openai==1.30.0

# RAG
pip install faiss-cpu==1.8.0
pip install sentence-transformers==3.0.0

# Fine-tuning (QLoRA)
pip install transformers==4.41.0
pip install peft==0.11.0
pip install bitsandbytes==0.43.0
pip install trl==0.9.0
pip install accelerate==0.30.0

# Data
pip install datasets==2.19.0
pip install pandas==2.2.0
pip install numpy==1.26.0

# Math / Stats
pip install scipy==1.13.0
pip install scikit-learn==1.4.0

# Evaluation
pip install ragas==0.1.9

# Experiment Tracking
pip install wandb==0.17.0

# Visualization
pip install matplotlib==3.9.0
pip install seaborn==0.13.0
pip install plotly==5.22.0

# App
pip install streamlit==1.35.0

# Utilities
pip install python-dotenv==1.0.0
pip install tqdm==4.66.0
pip install pydantic==2.7.0
```

### requirements.txt
```
langchain==0.2.0
langgraph==0.1.0
openai==1.30.0
faiss-cpu==1.8.0
sentence-transformers==3.0.0
transformers==4.41.0
peft==0.11.0
bitsandbytes==0.43.0
trl==0.9.0
accelerate==0.30.0
datasets==2.19.0
pandas==2.2.0
numpy==1.26.0
scipy==1.13.0
scikit-learn==1.4.0
ragas==0.1.9
wandb==0.17.0
matplotlib==3.9.0
seaborn==0.13.0
plotly==5.22.0
streamlit==1.35.0
python-dotenv==1.0.0
tqdm==4.66.0
pydantic==2.7.0
```

---

## 📊 STEP 1: Data Setup

### 1.1 Load DDxPlus — Primary Dataset
```python
# src/data/load_ddxplus.py

import pandas as pd
from datasets import load_dataset

def load_ddxplus(split="train"):
    """
    DDxPlus: 20,000+ differential diagnosis cases
    Source: https://github.com/mila-iqia/ddxplus
    Each case has: symptoms, antecedents, ground truth pathology
    """
    dataset = load_dataset("appier-ai-research/ddxplus", split=split)
    
    cases = []
    for item in dataset:
        cases.append({
            "id":           item["index"],
            "age":          item["AGE"],
            "sex":          item["SEX"],
            "symptoms":     item["SYMPTOMS"],
            "antecedents":  item["ANTECEDENTS"],
            "pathology":    item["PATHOLOGY"],        # ground truth ✅
            "differential": item["DIFFERENTIAL_DIAGNOSIS"]
        })
    
    return pd.DataFrame(cases)


def format_patient_case(row):
    """Format DDxPlus row into readable patient case text for agents"""
    symptoms = ', '.join(row['symptoms']) if isinstance(row['symptoms'], list) else row['symptoms']
    antecedents = ', '.join(row['antecedents']) if row['antecedents'] else 'None reported'
    
    return f"""
PATIENT CASE:
- Age: {row['age']} years old
- Sex: {row['sex']}

Presenting Symptoms:
{symptoms}

Medical History / Antecedents:
{antecedents}
""".strip()


# Quick test
if __name__ == "__main__":
    df = load_ddxplus()
    print(f"Loaded {len(df)} cases")
    print(f"Unique diagnoses: {df['pathology'].nunique()}")
    print("\nSample case:")
    print(format_patient_case(df.iloc[0]))
    print(f"\nGround truth: {df.iloc[0]['pathology']}")
```

### 1.2 Load MedQA — Secondary Dataset
```python
# src/data/load_medqa.py

from datasets import load_dataset

def load_medqa(split="test"):
    """
    MedQA: USMLE-style medical board exam questions
    Used as secondary evaluation dataset
    """
    dataset = load_dataset("bigbio/med_qa", "med_qa_en_bigbio_qa")
    return dataset[split]


def format_medqa_case(item):
    """Format MedQA item into patient case text"""
    return f"""
CLINICAL QUESTION:
{item['question']}
""".strip()
```

---

## 📚 STEP 2: RAG Knowledge Base

### 2.1 Fetch PubMed Abstracts
```python
# src/rag/knowledge_base.py

import requests
import json
import time
from tqdm import tqdm

def fetch_pubmed_abstracts(search_terms, max_per_term=200):
    """
    Fetch PubMed abstracts for medical topics
    Free API — no key needed for basic use
    """
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    all_docs = []
    
    for term in tqdm(search_terms, desc="Fetching PubMed"):
        # Search for PMIDs
        search_url = f"{base_url}esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": term,
            "retmax": max_per_term,
            "retmode": "json"
        }
        
        r = requests.get(search_url, params=params)
        pmids = r.json()["esearchresult"]["idlist"]
        
        if not pmids:
            continue
        
        # Fetch abstracts for those PMIDs
        fetch_url = f"{base_url}efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "text"
        }
        
        r = requests.get(fetch_url, params=fetch_params)
        abstracts = r.text.split("\n\n")
        
        for abstract in abstracts:
            if len(abstract) > 100:
                all_docs.append({
                    "term": term,
                    "abstract": abstract.strip()
                })
        
        time.sleep(0.5)  # Be respectful to PubMed API
    
    return all_docs


# Medical topics to fetch (covers DDxPlus diagnoses)
SEARCH_TERMS = [
    "differential diagnosis chest pain",
    "myocardial infarction diagnosis treatment",
    "pulmonary embolism clinical features",
    "pneumonia diagnosis management",
    "appendicitis clinical presentation",
    "acute abdomen differential diagnosis",
    "headache differential diagnosis",
    "fever diagnosis clinical guidelines",
    "shortness of breath diagnosis",
    "abdominal pain differential diagnosis"
]

# Run once to build knowledge base
if __name__ == "__main__":
    docs = fetch_pubmed_abstracts(SEARCH_TERMS, max_per_term=200)
    print(f"Fetched {len(docs)} documents")
    
    with open("data/knowledge_base/pubmed_abstracts.json", "w") as f:
        json.dump(docs, f)
    print("Saved to data/knowledge_base/pubmed_abstracts.json")
```

### 2.2 Build FAISS Index
```python
# src/rag/knowledge_base.py (continued)

import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

class MedicalKnowledgeBase:
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
    
    def load_documents(self, filepath):
        with open(filepath) as f:
            self.documents = json.load(f)
        print(f"Loaded {len(self.documents)} documents")
        return self
    
    def build_index(self):
        texts = [doc['abstract'] for doc in self.documents]
        
        print("Embedding documents...")
        embeddings = self.embedder.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Build FAISS flat index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        
        print(f"FAISS index built: {self.index.ntotal} vectors")
        return self
    
    def save(self, path="data/knowledge_base"):
        faiss.write_index(self.index, f"{path}/faiss.index")
        with open(f"{path}/documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)
        print(f"Saved index + documents to {path}/")
    
    def load(self, path="data/knowledge_base"):
        self.index = faiss.read_index(f"{path}/faiss.index")
        with open(f"{path}/documents.pkl", "rb") as f:
            self.documents = pickle.load(f)
        print(f"Loaded {self.index.ntotal} vectors")
        return self
```

### 2.3 Retriever
```python
# src/rag/retriever.py

import faiss
import numpy as np

class MedicalRetriever:
    
    def __init__(self, knowledge_base, top_k=5):
        self.kb = knowledge_base
        self.top_k = top_k
    
    def retrieve(self, query):
        """Return top-k relevant medical documents for a query"""
        query_embedding = self.kb.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.kb.index.search(query_embedding, self.top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append({
                    "abstract": self.kb.documents[idx]["abstract"],
                    "score": float(score)
                })
        return results
    
    def format_for_prompt(self, query):
        """Retrieve and format as prompt-ready string"""
        docs = self.retrieve(query)
        formatted = "\n\n".join([
            f"[Evidence {i+1} | Relevance: {doc['score']:.2f}]:\n{doc['abstract'][:600]}"
            for i, doc in enumerate(docs)
        ])
        return formatted
```

---

## 🤖 STEP 3: Agent Design

### 3.1 Doctor A — Advocate Agent
```python
# src/agents/doctor_a.py

from openai import OpenAI
import re

client = OpenAI()

DOCTOR_A_SYSTEM = """
You are Doctor A, a senior physician in a clinical debate panel.
Your role: Advocate for the most likely diagnosis based on evidence.

Rules:
1. Use ONLY the provided medical literature as evidence
2. Address ALL presented symptoms
3. Directly respond to Doctor B's counterarguments
4. Honestly state your confidence (0-100%)
5. Adjust position if evidence demands it — do NOT be stubborn

ALWAYS end your response with EXACTLY this format (no exceptions):
---
ARGUMENT: [your full clinical argument]
DIAGNOSIS: [your diagnosis name]
CONFIDENCE: [integer 0-100]
KEY_EVIDENCE: [top 3 pieces of evidence from literature]
---
"""

DOCTOR_A_PROMPT = """
## Patient Case
{patient_case}

## Retrieved Medical Literature (your evidence base)
{retrieved_docs}

## Debate History So Far
{debate_history}

## Doctor B's Last Argument
{doctor_b_argument}

{instruction}

End with the required format.
"""

def run_doctor_a(patient_case, retrieved_docs, debate_history, 
                  doctor_b_argument, round_num, model="gpt-4o-mini"):
    
    if round_num == 1:
        instruction = "This is Round 1. Propose your initial diagnosis with evidence."
    else:
        instruction = f"This is Round {round_num}. Respond to Doctor B's argument. Defend or refine your position."
    
    prompt = DOCTOR_A_PROMPT.format(
        patient_case=patient_case,
        retrieved_docs=retrieved_docs,
        debate_history=debate_history,
        doctor_b_argument=doctor_b_argument if doctor_b_argument else "None yet.",
        instruction=instruction
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": DOCTOR_A_SYSTEM},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.7
    )
    
    return parse_agent_response(response.choices[0].message.content)


def parse_agent_response(text):
    """Extract structured fields from agent response"""
    result = {"raw": text}
    
    patterns = {
        "argument":     r"ARGUMENT:\s*(.+?)(?=\nDIAGNOSIS:|\Z)",
        "diagnosis":    r"DIAGNOSIS:\s*(.+?)(?=\nCONFIDENCE:|\Z)",
        "confidence":   r"CONFIDENCE:\s*(\d+)",
        "key_evidence": r"KEY_EVIDENCE:\s*(.+?)(?=\n---|$)"
    }
    
    for field, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        result[field] = match.group(1).strip() if match else ""
    
    try:
        result["confidence"] = float(result["confidence"])
    except:
        result["confidence"] = 50.0
    
    return result
```

### 3.2 Doctor B — Devil's Advocate Agent
```python
# src/agents/doctor_b.py

DOCTOR_B_SYSTEM = """
You are Doctor B, a senior physician playing devil's advocate.
Your role: Propose and defend an ALTERNATIVE competing diagnosis.

Rules:
1. NEVER agree with Doctor A without strong evidence
2. Challenge Doctor A's reasoning — find what it cannot explain
3. Propose a different diagnosis supported by the same symptoms
4. Use retrieved medical literature as your evidence
5. State your confidence honestly (0-100%)

ALWAYS end with EXACTLY this format:
---
ARGUMENT: [your full clinical argument]
DIAGNOSIS: [your alternative diagnosis]
CONFIDENCE: [integer 0-100]
WEAKNESS_IN_A: [what Doctor A's diagnosis fails to explain]
SUPPORTING_EVIDENCE: [evidence from literature supporting your diagnosis]
---
"""

DOCTOR_B_PROMPT = """
## Patient Case
{patient_case}

## Retrieved Medical Literature
{retrieved_docs}

## Debate History So Far
{debate_history}

## Doctor A's Last Argument
{doctor_a_argument}

{instruction}

End with the required format.
"""

def run_doctor_b(patient_case, retrieved_docs, debate_history,
                  doctor_a_argument, round_num, model="gpt-4o-mini"):
    
    if round_num == 1:
        instruction = "This is Round 1. Propose an alternative diagnosis that challenges Doctor A."
    else:
        instruction = f"This is Round {round_num}. Counter Doctor A's argument. Strengthen your alternative diagnosis."
    
    prompt = DOCTOR_B_PROMPT.format(
        patient_case=patient_case,
        retrieved_docs=retrieved_docs,
        debate_history=debate_history,
        doctor_a_argument=doctor_a_argument,
        instruction=instruction
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": DOCTOR_B_SYSTEM},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.7
    )
    
    # Reuse parse function from doctor_a.py
    from src.agents.doctor_a import parse_agent_response
    return parse_agent_response(response.choices[0].message.content)
```

### 3.3 Moderator Agent (GPT-4o version — before fine-tuning)
```python
# src/agents/moderator.py

MODERATOR_SYSTEM = """
You are a neutral senior medical moderator.
You do NOT propose diagnoses. You evaluate debate arguments objectively.

Your job:
1. Assess which doctor provided stronger clinical evidence
2. Identify symptoms left unexplained by either doctor
3. Give a final verdict based purely on argument quality
4. Assign a confidence score to your verdict

ALWAYS end with EXACTLY this format:
---
WINNER: [Doctor A | Doctor B | INCONCLUSIVE]
FINAL_DIAGNOSIS: [most supported diagnosis]
REASONING: [why this diagnosis won the debate]
UNEXPLAINED_SYMPTOMS: [symptoms neither doctor addressed well]
VERDICT_CONFIDENCE: [integer 0-100]
---
"""

MODERATOR_PROMPT = """
## Patient Case
{patient_case}

## Full Debate Transcript
{full_transcript}

## Doctor A Final Position
- Diagnosis: {a_diagnosis}
- Confidence: {a_confidence}%
- Key Evidence: {a_evidence}

## Doctor B Final Position  
- Diagnosis: {b_diagnosis}
- Confidence: {b_confidence}%
- Weakness found in A: {b_weakness}

Evaluate both arguments and deliver your verdict.
"""

def run_moderator(patient_case, full_transcript, state, 
                   use_finetuned=False, model="gpt-4o-mini"):
    
    prompt = MODERATOR_PROMPT.format(
        patient_case=patient_case,
        full_transcript=full_transcript,
        a_diagnosis=state["doctor_a_diagnoses"][-1],
        a_confidence=state["doctor_a_confidences"][-1],
        a_evidence=state["doctor_a_arguments"][-1][:300],
        b_diagnosis=state["doctor_b_diagnoses"][-1],
        b_confidence=state["doctor_b_confidences"][-1],
        b_weakness=state["doctor_b_arguments"][-1][:300]
    )
    
    if use_finetuned:
        # Use fine-tuned Llama moderator (Phase 2)
        return run_finetuned_moderator(prompt)
    else:
        # Use GPT-4o mini moderator (Phase 1)
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": MODERATOR_SYSTEM},
                {"role": "user",   "content": prompt}
            ],
            temperature=0.3  # Low temp for consistent verdicts
        )
        from src.agents.doctor_a import parse_agent_response
        return parse_agent_response(response.choices[0].message.content)
```

---

## 🔄 STEP 4: LangGraph Debate Orchestrator

### 4.1 State Schema
```python
# src/debate/state.py

from typing import TypedDict, List, Optional, Dict, Any

class DebateState(TypedDict):
    # ── Input ──────────────────────────────────────
    patient_case:           str
    ground_truth:           str
    
    # ── RAG ────────────────────────────────────────
    retrieved_docs_a:       str
    retrieved_docs_b:       str
    
    # ── Debate Control ─────────────────────────────
    current_round:          int
    max_rounds:             int
    
    # ── Doctor A Outputs (per round) ───────────────
    doctor_a_arguments:     List[str]
    doctor_a_confidences:   List[float]
    doctor_a_diagnoses:     List[str]
    
    # ── Doctor B Outputs (per round) ───────────────
    doctor_b_arguments:     List[str]
    doctor_b_confidences:   List[float]
    doctor_b_diagnoses:     List[str]
    
    # ── Moderator Output ───────────────────────────
    moderator_verdict:      Optional[str]
    final_diagnosis:        Optional[str]
    verdict_confidence:     Optional[float]
    
    # ── PDS (Novel Metric) ─────────────────────────
    position_drift_score:   Optional[float]
    pds_components:         Optional[Dict[str, float]]
    
    # ── HITL ───────────────────────────────────────
    escalate_to_human:      bool
    escalation_reason:      Optional[str]
    
    # ── Meta ───────────────────────────────────────
    use_finetuned_moderator: bool
```

### 4.2 LangGraph Orchestrator
```python
# src/debate/orchestrator.py

from langgraph.graph import StateGraph, END
from src.debate.state import DebateState
from src.agents.doctor_a import run_doctor_a
from src.agents.doctor_b import run_doctor_b
from src.agents.moderator import run_moderator
from src.uncertainty.pds import PositionDriftScore
from src.hitl.escalation import decide_escalation

PDS_THRESHOLD = 0.5


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


# ── Node Functions ─────────────────────────────────────────

def retrieve_node(state: DebateState) -> DebateState:
    """RAG retrieval for both agents"""
    from src.rag.retriever import retriever  # initialized globally
    
    query = state["patient_case"][:300]
    state["retrieved_docs_a"] = retriever.format_for_prompt(query)
    state["retrieved_docs_b"] = retriever.format_for_prompt(
        query + " alternative diagnosis differential"
    )
    return state


def doctor_a_node(state: DebateState) -> DebateState:
    """Run Doctor A for current round"""
    result = run_doctor_a(
        patient_case=state["patient_case"],
        retrieved_docs=state["retrieved_docs_a"],
        debate_history=format_debate_history(state),
        doctor_b_argument=state["doctor_b_arguments"][-1] if state["doctor_b_arguments"] else "",
        round_num=state["current_round"] + 1
    )
    
    state["doctor_a_arguments"].append(result["argument"])
    state["doctor_a_confidences"].append(result["confidence"])
    state["doctor_a_diagnoses"].append(result["diagnosis"])
    return state


def doctor_b_node(state: DebateState) -> DebateState:
    """Run Doctor B for current round"""
    result = run_doctor_b(
        patient_case=state["patient_case"],
        retrieved_docs=state["retrieved_docs_b"],
        debate_history=format_debate_history(state),
        doctor_a_argument=state["doctor_a_arguments"][-1],
        round_num=state["current_round"] + 1
    )
    
    state["doctor_b_arguments"].append(result["argument"])
    state["doctor_b_confidences"].append(result["confidence"])
    state["doctor_b_diagnoses"].append(result["diagnosis"])
    state["current_round"] += 1
    return state


def moderator_node(state: DebateState) -> DebateState:
    """Run Moderator after all debate rounds"""
    transcript = format_full_transcript(state)
    
    result = run_moderator(
        patient_case=state["patient_case"],
        full_transcript=transcript,
        state=state,
        use_finetuned=state.get("use_finetuned_moderator", False)
    )
    
    state["moderator_verdict"]  = result.get("winner", "INCONCLUSIVE")
    state["final_diagnosis"]    = result.get("final_diagnosis", "")
    state["verdict_confidence"] = result.get("verdict_confidence", 50.0)
    return state


def pds_node(state: DebateState) -> DebateState:
    """Compute Position Drift Score"""
    calculator = PositionDriftScore()
    pds, components = calculator.compute(
        doctor_a_confidences=state["doctor_a_confidences"],
        doctor_b_confidences=state["doctor_b_confidences"],
        doctor_a_arguments=state["doctor_a_arguments"],
        doctor_b_arguments=state["doctor_b_arguments"]
    )
    
    state["position_drift_score"] = pds
    state["pds_components"]       = components
    return state


def escalation_node(state: DebateState) -> DebateState:
    """HITL decision based on PDS"""
    escalate, reason = decide_escalation(
        pds_score=state["position_drift_score"],
        threshold=PDS_THRESHOLD
    )
    state["escalate_to_human"] = escalate
    state["escalation_reason"] = reason
    return state


# ── Conditional Edge ───────────────────────────────────────

def should_continue(state: DebateState) -> str:
    if state["current_round"] < state["max_rounds"]:
        return "continue"
    return "moderate"


# ── Build Graph ────────────────────────────────────────────

def build_debate_graph():
    graph = StateGraph(DebateState)
    
    graph.add_node("retrieve",   retrieve_node)
    graph.add_node("doctor_a",   doctor_a_node)
    graph.add_node("doctor_b",   doctor_b_node)
    graph.add_node("moderator",  moderator_node)
    graph.add_node("compute_pds", pds_node)
    graph.add_node("escalation", escalation_node)
    
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "doctor_a")
    graph.add_edge("doctor_a", "doctor_b")
    graph.add_conditional_edges(
        "doctor_b",
        should_continue,
        {"continue": "doctor_a", "moderate": "moderator"}
    )
    graph.add_edge("moderator",  "compute_pds")
    graph.add_edge("compute_pds", "escalation")
    graph.add_edge("escalation", END)
    
    return graph.compile()


# ── Run One Case ───────────────────────────────────────────

def run_debate(patient_case, ground_truth, max_rounds=3, 
               use_finetuned=False):
    graph = build_debate_graph()
    
    initial_state = {
        "patient_case":           patient_case,
        "ground_truth":           ground_truth,
        "retrieved_docs_a":       "",
        "retrieved_docs_b":       "",
        "current_round":          0,
        "max_rounds":             max_rounds,
        "doctor_a_arguments":     [],
        "doctor_b_arguments":     [],
        "doctor_a_confidences":   [],
        "doctor_b_confidences":   [],
        "doctor_a_diagnoses":     [],
        "doctor_b_diagnoses":     [],
        "moderator_verdict":      None,
        "final_diagnosis":        None,
        "verdict_confidence":     None,
        "position_drift_score":   None,
        "pds_components":         None,
        "escalate_to_human":      False,
        "escalation_reason":      None,
        "use_finetuned_moderator": use_finetuned
    }
    
    return graph.invoke(initial_state)
```

---

## 📐 STEP 5: Position Drift Score — NOVEL METRIC

```python
# src/uncertainty/pds.py

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
    
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.weights = {
            "confidence_drift":   0.35,
            "semantic_drift":     0.40,
            "final_disagreement": 0.25
        }
    
    # ── Component 1: Confidence Drift ─────────────────
    def confidence_drift(self, conf_a: list, conf_b: list) -> float:
        """
        How much did numeric confidence CHANGE across rounds?
        High std dev = unstable confidence = uncertain case
        """
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
        
        THIS IS THE TRULY NOVEL PART — no existing paper 
        measures uncertainty this way for medical debate.
        """
        if len(args_a) < 2 or len(args_b) < 2:
            return 0.0
        
        # Embed first and last arguments
        emb_a_first = self.embedder.encode(args_a[0],  convert_to_tensor=True)
        emb_a_last  = self.embedder.encode(args_a[-1], convert_to_tensor=True)
        emb_b_first = self.embedder.encode(args_b[0],  convert_to_tensor=True)
        emb_b_last  = self.embedder.encode(args_b[-1], convert_to_tensor=True)
        
        # Cosine similarity → drift = 1 - similarity
        sim_a = util.cos_sim(emb_a_first, emb_a_last).item()
        sim_b = util.cos_sim(emb_b_first, emb_b_last).item()
        
        drift_a = 1.0 - max(sim_a, 0)
        drift_b = 1.0 - max(sim_b, 0)
        
        return float((drift_a + drift_b) / 2.0)
    
    # ── Component 3: Final Disagreement ───────────────
    def final_disagreement(self, conf_a: list, conf_b: list) -> float:
        """
        How far apart are agents in the FINAL round?
        Still far apart at end = unresolved conflict = uncertain
        """
        gap = abs(conf_a[-1] - conf_b[-1]) / 100.0
        return float(gap)
    
    # ── Final PDS Score ───────────────────────────────
    def compute(self, doctor_a_confidences, doctor_b_confidences,
                doctor_a_arguments, doctor_b_arguments):
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
            "confidence_drift":    round(c1, 4),
            "semantic_drift":      round(c2, 4),
            "final_disagreement":  round(c3, 4),
            "pds_score":           round(pds, 4),
            "interpretation":      self.interpret(pds),
            "weights_used":        self.weights
        }
        
        return pds, components
    
    def interpret(self, pds: float) -> str:
        if pds < 0.20:
            return "HIGH CONFIDENCE — Safe to give diagnosis"
        elif pds < 0.50:
            return "MEDIUM CONFIDENCE — Give with caution"
        else:
            return "LOW CONFIDENCE — Escalate to human doctor"
```

---

## 🚨 STEP 6: HITL Escalation

```python
# src/hitl/escalation.py

def decide_escalation(pds_score: float, threshold: float = 0.5):
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
    """
    Format a structured summary for the human doctor
    when escalation is triggered
    """
    return f"""
🚨 CLINICAL AI ESCALATION NOTICE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PATIENT CASE:
{state['patient_case']}

AI DEBATE SUMMARY:
• Doctor A argued: {state['doctor_a_diagnoses'][-1]} 
  (Final confidence: {state['doctor_a_confidences'][-1]:.0f}%)
  
• Doctor B argued: {state['doctor_b_diagnoses'][-1]}
  (Final confidence: {state['doctor_b_confidences'][-1]:.0f}%)

UNCERTAINTY ANALYSIS (PDS):
• Overall PDS Score: {state['position_drift_score']:.3f}
• Confidence Drift:  {state['pds_components']['confidence_drift']:.3f}
• Semantic Drift:    {state['pds_components']['semantic_drift']:.3f}
• Final Disagreement:{state['pds_components']['final_disagreement']:.3f}

REASON FOR ESCALATION:
{state['escalation_reason']}

⚠️  Human physician review required before any clinical decision.
""".strip()
```

---

## 🧪 STEP 7: Fine-Tuning Moderator with QLoRA

### 7.1 Generate Training Data
```python
# src/finetuning/generate_training_data.py

"""
Strategy:
1. Run GPT-4o (strong model) as moderator on 1000 debate transcripts
2. Collect high-quality (transcript → verdict) pairs
3. Use these to fine-tune Llama 3.1 8B as moderator
4. Fine-tuned Llama moderator should match GPT-4o quality
   at much lower inference cost
"""

from openai import OpenAI
import json
from tqdm import tqdm

client = OpenAI()

def generate_moderator_training_sample(debate_transcript, patient_case):
    """Use GPT-4o to generate gold-standard moderator verdicts"""
    
    response = client.chat.completions.create(
        model="gpt-4o",   # Use strongest model for training data
        messages=[
            {"role": "system", "content": MODERATOR_SYSTEM},
            {"role": "user",   "content": f"Patient: {patient_case}\n\nTranscript:\n{debate_transcript}"}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content


def build_training_dataset(debates_df, output_path, n_samples=1000):
    """
    Generate n_samples training examples for moderator fine-tuning
    Format: instruction-following style for Llama
    """
    training_data = []
    
    for i, row in tqdm(debates_df.head(n_samples).iterrows()):
        verdict = generate_moderator_training_sample(
            row["debate_transcript"], 
            row["patient_case"]
        )
        
        training_data.append({
            "instruction": MODERATOR_SYSTEM,
            "input": f"Patient: {row['patient_case']}\n\nDebate:\n{row['debate_transcript']}",
            "output": verdict
        })
    
    with open(output_path, "w") as f:
        json.dump(training_data, f, indent=2)
    
    print(f"Generated {len(training_data)} training samples → {output_path}")
```

### 7.2 QLoRA Fine-Tuning Script
```python
# src/finetuning/finetune_moderator.py

"""
Fine-tune Llama 3.1 8B as medical debate moderator using QLoRA
Run on Google Colab A100 GPU (~2-3 hours)
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# ── 1. Model Config ───────────────────────────────────────
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# 4-bit quantization (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# ── 2. Load Model ─────────────────────────────────────────
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# ── 3. LoRA Config ────────────────────────────────────────
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,               # LoRA rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=[    # Target attention layers
        "q_proj", "k_proj", 
        "v_proj", "o_proj"
    ],
    bias="none"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected: ~0.5% trainable parameters → very efficient!

# ── 4. Load Training Data ─────────────────────────────────
dataset = load_dataset(
    "json", 
    data_files="data/moderator_training_data.json",
    split="train"
)

def format_training_example(example):
    return {
        "text": f"""### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}"""
    }

dataset = dataset.map(format_training_example)

# ── 5. Training Config ────────────────────────────────────
training_args = SFTConfig(
    output_dir="models/moderator-qlora",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    report_to="wandb",            # Track on W&B
    run_name="medebate-moderator-qlora"
)

# ── 6. Train ──────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
    dataset_text_field="text",
    max_seq_length=2048
)

trainer.train()
trainer.save_model("models/moderator-qlora-final")
print("Fine-tuning complete! Model saved.")
```

### 7.3 Load Fine-Tuned Moderator for Inference
```python
# src/agents/moderator.py (fine-tuned version)

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

def load_finetuned_moderator(
    base_model="meta-llama/Llama-3.1-8B-Instruct",
    adapter_path="models/moderator-qlora-final"
):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        load_in_4bit=True,
        device_map="auto",
        torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    return model, tokenizer


def run_finetuned_moderator(prompt, model, tokenizer, max_new_tokens=512):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    from src.agents.doctor_a import parse_agent_response
    return parse_agent_response(response)
```

---

## 📊 STEP 8: Baselines

```python
# src/evaluation/baselines.py

from openai import OpenAI
client = OpenAI()

# ── Baseline 1: Single LLM ────────────────────────────────
def baseline_single_llm(patient_case, model="gpt-4o-mini"):
    """No debate, no RAG — just ask GPT directly"""
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": 
            f"What is the most likely diagnosis for this patient?\n\n{patient_case}\n\nDiagnosis:"}]
    )
    return r.choices[0].message.content.strip()


# ── Baseline 2: Chain-of-Thought ──────────────────────────
def baseline_cot(patient_case, model="gpt-4o-mini"):
    """Single LLM with step-by-step reasoning"""
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": 
            f"Think step by step about this patient case and give the most likely diagnosis.\n\n{patient_case}\n\nStep-by-step reasoning:"}]
    )
    return r.choices[0].message.content.strip()


# ── Baseline 3: Self-Consistency ──────────────────────────
def baseline_self_consistency(patient_case, n=5, model="gpt-4o-mini"):
    """Sample 5 answers, return majority vote"""
    from collections import Counter
    answers = []
    for _ in range(n):
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": 
                f"What is the single most likely diagnosis?\n\n{patient_case}\n\nDiagnosis (one short phrase):"}],
            temperature=0.9
        )
        answers.append(r.choices[0].message.content.strip())
    return Counter(answers).most_common(1)[0][0]


# ── Baseline 4: RAG + Single LLM ─────────────────────────
def baseline_rag_single(patient_case, retriever, model="gpt-4o-mini"):
    """RAG retrieval + single LLM — no debate"""
    docs = retriever.format_for_prompt(patient_case[:200])
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": 
            f"Using the medical literature below, diagnose this patient.\n\nLiterature:\n{docs}\n\nPatient:\n{patient_case}\n\nDiagnosis:"}]
    )
    return r.choices[0].message.content.strip()
```

---

## 🔬 STEP 9: Experiments

### 9.1 Main Experiment
```python
# experiments/run_debate.py

import pandas as pd
from tqdm import tqdm
import wandb

wandb.init(project="medebate-rag", name="main-experiment")

def run_all_experiments(test_df, retriever, n_cases=200):
    results = []
    
    for i, row in tqdm(test_df.head(n_cases).iterrows()):
        patient_case = format_patient_case(row)
        ground_truth = row["pathology"]
        
        result = {"case_id": i, "ground_truth": ground_truth}
        
        # ── Baselines ────────────────────────────────────
        result["b1_single"]    = baseline_single_llm(patient_case)
        result["b2_cot"]       = baseline_cot(patient_case)
        result["b3_selfcon"]   = baseline_self_consistency(patient_case)
        result["b4_rag"]       = baseline_rag_single(patient_case, retriever)
        
        # ── Full Debate System ────────────────────────────
        final_state = run_debate(patient_case, ground_truth, max_rounds=3)
        result["debate_diagnosis"] = final_state["final_diagnosis"]
        result["pds_score"]        = final_state["position_drift_score"]
        result["escalated"]        = final_state["escalate_to_human"]
        result["pds_components"]   = final_state["pds_components"]
        result["a_confidences"]    = final_state["doctor_a_confidences"]
        result["b_confidences"]    = final_state["doctor_b_confidences"]
        
        results.append(result)
        
        # Log to W&B
        wandb.log({
            "pds_score": result["pds_score"],
            "escalated": result["escalated"]
        })
    
    df = pd.DataFrame(results)
    df.to_csv("experiments/results/main_results.csv", index=False)
    return df
```

### 9.2 Ablation Studies
```python
# experiments/run_ablations.py

"""
5 Ablation Studies:

A1: No RAG          → debate only (no retrieved docs)
A2: No PDS/HITL     → always give diagnosis, never escalate
A3: 1 round debate  → single exchange only
A4: 3 round debate  → standard (your main system)
A5: 5 round debate  → extended debate
A6: No debate       → single agent + moderator only

Each ablation runs on same 100 cases for fair comparison.
"""

ABLATION_CASES = 100

def ablation_no_rag(test_df):
    """A1: Disable RAG retrieval"""
    results = []
    for _, row in test_df.head(ABLATION_CASES).iterrows():
        # Run debate with empty retrieved docs
        state = run_debate(
            format_patient_case(row), 
            row["pathology"],
            max_rounds=3,
            override_retrieved_docs=""   # empty RAG
        )
        results.append({
            "ground_truth": row["pathology"],
            "diagnosis": state["final_diagnosis"],
            "pds": state["position_drift_score"]
        })
    return pd.DataFrame(results)


def ablation_rounds(test_df, rounds=[1, 2, 3, 5]):
    """A3-A5: Different debate lengths"""
    all_results = {}
    for r in rounds:
        results = []
        for _, row in test_df.head(ABLATION_CASES).iterrows():
            state = run_debate(
                format_patient_case(row),
                row["pathology"],
                max_rounds=r
            )
            results.append({
                "ground_truth": row["pathology"],
                "diagnosis": state["final_diagnosis"],
                "pds": state["position_drift_score"],
                "rounds": r
            })
        all_results[f"rounds_{r}"] = pd.DataFrame(results)
    return all_results
```

### 9.3 PDS Validation — Key Experiment
```python
# experiments/validate_pds.py

"""
HYPOTHESIS: High PDS → Lower Accuracy
            Low PDS  → Higher Accuracy

If true → PDS is a valid uncertainty metric!
This is the experiment that PROVES your novel contribution works.
"""

import matplotlib.pyplot as plt
import numpy as np

def validate_pds(results_df):
    
    def accuracy(df):
        return df.apply(
            lambda r: r["ground_truth"].lower() in r["debate_diagnosis"].lower(), 
            axis=1
        ).mean() * 100
    
    # Split by PDS bucket
    low  = results_df[results_df["pds_score"] < 0.2]
    mid  = results_df[(results_df["pds_score"] >= 0.2) & (results_df["pds_score"] < 0.5)]
    high = results_df[results_df["pds_score"] >= 0.5]
    
    acc = {
        "Low PDS\n(<0.2)":   accuracy(low),
        "Mid PDS\n(0.2-0.5)": accuracy(mid),
        "High PDS\n(>0.5)":   accuracy(high)
    }
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart: PDS bucket vs accuracy
    axes[0].bar(acc.keys(), acc.values(), color=["green","orange","red"], alpha=0.8)
    axes[0].set_ylabel("Diagnosis Accuracy (%)")
    axes[0].set_title("PDS Score Bucket vs Accuracy\n(Lower PDS = Higher Accuracy)")
    axes[0].set_ylim(0, 100)
    
    # Scatter: raw PDS vs correctness
    correct = results_df.apply(
        lambda r: r["ground_truth"].lower() in r["debate_diagnosis"].lower(), axis=1
    )
    axes[1].scatter(results_df["pds_score"], correct.astype(int), alpha=0.3)
    axes[1].set_xlabel("PDS Score")
    axes[1].set_ylabel("Correct Diagnosis (1=Yes, 0=No)")
    axes[1].set_title("PDS vs Correctness (Raw Scatter)")
    
    plt.tight_layout()
    plt.savefig("experiments/results/pds_validation.png", dpi=150)
    plt.show()
    
    print("PDS Validation Results:")
    for bucket, a in acc.items():
        print(f"  {bucket.replace(chr(10),' ')}: {a:.1f}%")


### 9.4 Dangerous Cases Experiment — Killer Result
def dangerous_cases_experiment(results_df):
    """
    Find cases where single LLM was WRONG but CONFIDENT.
    Show your system correctly ESCALATED these cases.
    This is your most impactful result for the paper.
    """
    
    # Cases where single LLM was wrong and overconfident
    # (proxy: single LLM wrong + debate system had high PDS)
    dangerous = results_df[
        (results_df.apply(
            lambda r: r["ground_truth"].lower() not in r["b1_single"].lower(), 
            axis=1)
        ) &
        (results_df["pds_score"] > 0.5)  # your system flagged these
    ]
    
    total_wrong = results_df.apply(
        lambda r: r["ground_truth"].lower() not in r["b1_single"].lower(), axis=1
    ).sum()
    
    caught = len(dangerous)
    catch_rate = caught / total_wrong * 100 if total_wrong > 0 else 0
    
    print(f"\n🚨 DANGEROUS CASES EXPERIMENT:")
    print(f"Cases where single LLM was wrong:         {total_wrong}")
    print(f"Cases your system correctly escalated:    {caught}")
    print(f"Catch rate:                               {catch_rate:.1f}%")
    print(f"\nThis means your system prevented {catch_rate:.0f}% of dangerous")
    print(f"overconfident misdiagnoses from reaching a clinical decision!")
    
    return dangerous, catch_rate
```

---

## 🎨 STEP 10: Streamlit Demo UI

```python
# app/streamlit_app.py

import streamlit as st
import sys
sys.path.append("..")
from src.debate.orchestrator import run_debate
from src.data.load_ddxplus import format_patient_case, load_ddxplus

st.set_page_config(page_title="MedDebate-RAG", layout="wide", 
                    page_icon="🏥")

# Header
st.title("🏥 MedDebate-RAG")
st.caption("Uncertainty-Aware Multi-Agent Clinical Debate | Frontiers of LLMs Project")

st.info("""
**How it works:** Two AI doctors debate a diagnosis using real medical literature. 
A moderator computes the **Position Drift Score (PDS)** to measure uncertainty. 
If PDS is too high → escalates to human doctor.
""")

# ── Sidebar Controls ──────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    max_rounds = st.slider("Debate Rounds", 1, 5, 3)
    pds_threshold = st.slider("Escalation Threshold", 0.1, 0.9, 0.5, 
                               help="PDS above this → escalate to human")
    use_finetuned = st.checkbox("Use Fine-tuned Moderator (Llama)", value=False)
    
    st.markdown("---")
    st.markdown("**PDS Thresholds:**")
    st.markdown("🟢 < 0.2 — High Confidence")
    st.markdown("🟡 0.2-0.5 — Medium Confidence")
    st.markdown("🔴 > 0.5 — Escalate to Human")

# ── Input ────────────────────────────────────────────────
st.subheader("📋 Patient Case Input")

col1, col2 = st.columns([2, 1])
with col1:
    patient_case = st.text_area(
        "Enter patient symptoms, age, medical history:",
        placeholder="Example: 45-year-old male presenting with chest pain, shortness of breath, sweating. Elevated troponin (2.5 ng/mL). History of hypertension and smoking.",
        height=150
    )

with col2:
    st.markdown("**Or load a sample case:**")
    if st.button("🎲 Load Random DDxPlus Case"):
        df = load_ddxplus()
        sample = df.sample(1).iloc[0]
        patient_case = format_patient_case(sample)
        st.session_state["ground_truth"] = sample["pathology"]
        st.rerun()
    
    if "ground_truth" in st.session_state:
        st.success(f"Ground Truth: {st.session_state['ground_truth']}")

# ── Run Debate ────────────────────────────────────────────
if st.button("🚀 Start Clinical Debate", type="primary", 
              disabled=not patient_case):
    
    with st.spinner("🔍 Retrieving medical literature from PubMed..."):
        final_state = run_debate(
            patient_case, 
            ground_truth="",
            max_rounds=max_rounds,
            use_finetuned=use_finetuned
        )
    
    # ── Debate Rounds ─────────────────────────────────────
    st.subheader("🎙️ Debate Transcript")
    
    for round_num in range(len(final_state["doctor_a_arguments"])):
        with st.expander(f"Round {round_num + 1}", expanded=(round_num == len(final_state["doctor_a_arguments"]) - 1)):
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("#### 🔵 Doctor A (Advocate)")
                st.write(final_state["doctor_a_arguments"][round_num])
                st.metric("Diagnosis", final_state["doctor_a_diagnoses"][round_num])
                st.metric("Confidence", f"{final_state['doctor_a_confidences'][round_num]:.0f}%")
            
            with col_b:
                st.markdown("#### 🔴 Doctor B (Devil's Advocate)")
                st.write(final_state["doctor_b_arguments"][round_num])
                st.metric("Diagnosis", final_state["doctor_b_diagnoses"][round_num])
                st.metric("Confidence", f"{final_state['doctor_b_confidences'][round_num]:.0f}%")
    
    # ── PDS Analysis ──────────────────────────────────────
    st.subheader("📐 Position Drift Score Analysis")
    
    pds = final_state["position_drift_score"]
    comp = final_state["pds_components"]
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Confidence Drift",    f"{comp['confidence_drift']:.3f}",   help="How much numeric confidence changed")
    c2.metric("Semantic Drift",      f"{comp['semantic_drift']:.3f}",     help="How much argument meaning changed")
    c3.metric("Final Disagreement",  f"{comp['final_disagreement']:.3f}", help="Gap between agents at end")
    c4.metric("🎯 PDS Score",        f"{pds:.3f}",
              delta="Below threshold ✅" if pds < pds_threshold else "Above threshold 🚨",
              delta_color="normal" if pds < pds_threshold else "inverse")
    
    # PDS gauge
    st.progress(pds, text=f"PDS: {pds:.3f} | {comp['interpretation']}")
    
    # ── Final Verdict ─────────────────────────────────────
    st.subheader("⚖️ Final Verdict")
    
    if not final_state["escalate_to_human"]:
        st.success(f"✅ **Final Diagnosis: {final_state['final_diagnosis']}**")
        st.info(f"Moderator Confidence: {final_state['verdict_confidence']:.0f}% | PDS: {pds:.3f} (safe)")
    else:
        st.error("🚨 ESCALATE TO HUMAN DOCTOR")
        st.warning(final_state["escalation_reason"])
        
        with st.expander("📋 Summary for Human Doctor"):
            from src.hitl.escalation import format_escalation_summary
            st.text(format_escalation_summary(final_state))
```

---

## 📄 STEP 11: Paper Structure

```
Paper Title: MedDebate-RAG: Uncertainty-Aware Multi-Agent Clinical 
             Reasoning with Human-in-the-Loop Escalation

Target: ACL 2025 / EMNLP 2025 / CHIL 2025 (Clinical NLP track)
Length: 8 pages + references

1. ABSTRACT (150 words)
   Problem → Method → PDS → Results → Contribution

2. INTRODUCTION
   2.1 Clinical diagnosis challenge
   2.2 Why single LLM fails (overconfidence problem)
   2.3 Our approach: debate-based stress testing
   2.4 Contributions (list 3 bullet points)

3. RELATED WORK
   3.1 Multi-Agent Debate (Du et al. 2023, Chan et al. 2023)
   3.2 Medical LLMs (MedPaLM, BioGPT, MedRAG)
   3.3 Uncertainty Quantification (Kadavath 2022, Kuhn 2023)
   3.4 Human-in-the-Loop AI (Mozannar 2020, Wilder 2021)

4. METHOD
   4.1 System Overview (architecture diagram)
   4.2 RAG Knowledge Base
   4.3 Agent Design
   4.4 ★ Position Drift Score — MOST IMPORTANT SECTION
       4.4.1 Confidence Drift
       4.4.2 Semantic Drift (novel)
       4.4.3 Final Disagreement
       4.4.4 PDS Formula
   4.5 HITL Escalation Protocol
   4.6 Moderator Fine-tuning (QLoRA)

5. EXPERIMENTS
   5.1 Datasets (DDxPlus + MedQA)
   5.2 Baselines (4 methods)
   5.3 Evaluation Metrics
   5.4 Main Results Table
   5.5 Ablation Studies (5 ablations)
   5.6 PDS Validation
   5.7 ★ Dangerous Cases Experiment (killer result)
   5.8 Fine-tuned vs GPT-4o Moderator Comparison

6. RESULTS & ANALYSIS
   - Table 1: Main accuracy comparison
   - Figure 1: PDS vs Accuracy validation plot
   - Figure 2: Ablation results bar chart
   - Figure 3: Dangerous cases caught
   - Table 2: Fine-tuned moderator comparison

7. DISCUSSION
   7.1 When debate helps vs. hurts
   7.2 PDS calibration analysis
   7.3 Limitations (synthetic data, no clinical validation)
   7.4 Ethical considerations & responsible AI

8. CONCLUSION
```

---

## 🗓️ Week-by-Week Timeline

| Week | Phase | Tasks | Deliverable |
|---|---|---|---|
| **1** | Setup | Env setup, load DDxPlus, explore data, fetch PubMed | Data pipeline working |
| **2** | RAG | Build FAISS index, test retriever on sample queries | Retriever returning docs |
| **3** | Agents | Build Doctor A + B with prompts, test single round | One debate working |
| **4** | Orchestration | LangGraph graph, multi-round loop, state management | Full 3-round debate |
| **5** | PDS | Implement all 3 PDS components, unit test each | PDS scores generating |
| **6** | HITL | Escalation logic, escalation summary formatter | End-to-end pipeline |
| **7** | Fine-tuning | Generate training data, QLoRA train Llama moderator | Fine-tuned moderator |
| **8** | Experiments | Run 200 cases, all 4 baselines, collect results | Numbers ready |
| **9** | Analysis | Ablations, PDS validation, dangerous cases experiment | All plots + tables |
| **10** | UI + Paper | Streamlit demo + write full paper | Final submission |

---

## 💰 Student Budget

| Item | Cost |
|---|---|
| Google Colab Pro (2 months) | $20 |
| OpenAI API (all experiments) | ~$5 |
| Everything else | $0 |
| **TOTAL** | **~$25** |

---

## 🔑 5 Research Questions

| # | Research Question | How to Measure |
|---|---|---|
| **RQ1** | Does multi-agent debate outperform single LLM? | Accuracy vs. 4 baselines on DDxPlus |
| **RQ2** | Does RAG grounding reduce hallucination? | Ablation: with vs. without RAG (RAGAS faithfulness) |
| **RQ3** | Is PDS a valid uncertainty metric? | PDS score vs. correctness correlation |
| **RQ4** | Does HITL escalation prevent dangerous errors? | Dangerous cases catch rate |
| **RQ5** | Does fine-tuned moderator match GPT-4o quality? | Verdict accuracy: Llama vs. GPT-4o moderator |

---

## ✅ Final Checklist — Definition of Done

### Infrastructure
- [ ] Google Colab Pro setup
- [ ] OpenAI API key configured
- [ ] W&B account connected
- [ ] GitHub repo created

### Data
- [ ] DDxPlus dataset loaded + formatted
- [ ] PubMed abstracts fetched (10,000+ docs)
- [ ] FAISS index built + saved
- [ ] MedQA loaded for secondary eval

### Core Pipeline
- [ ] Doctor A agent (GPT-4o mini + RAG)
- [ ] Doctor B agent (GPT-4o mini + RAG)
- [ ] LangGraph debate loop (3 rounds)
- [ ] PDS — Confidence Drift component
- [ ] PDS — Semantic Drift component
- [ ] PDS — Final Disagreement component
- [ ] PDS — Full weighted formula
- [ ] HITL escalation logic
- [ ] Escalation summary formatter

### Fine-Tuning
- [ ] 1000 training samples generated (GPT-4o moderator)
- [ ] QLoRA fine-tuning completed (Llama 3.1 8B)
- [ ] Fine-tuned moderator integrated into pipeline

### Experiments
- [ ] All 4 baselines implemented
- [ ] Main experiment: 200 cases
- [ ] Ablation A1: No RAG
- [ ] Ablation A2: No HITL
- [ ] Ablation A3-A5: 1, 3, 5 rounds
- [ ] PDS validation experiment
- [ ] Dangerous cases experiment
- [ ] Fine-tuned vs GPT-4o moderator comparison

### Output
- [ ] Streamlit demo deployed on Streamlit Cloud
- [ ] All plots generated for paper
- [ ] Paper written (8 pages)
- [ ] GitHub repo public with README

---

## 🎓 Skills Demonstrated in This Project

| Skill | Component |
|---|---|
| Prompt Engineering | Doctor A, B, Moderator prompts |
| Multi-Agent Systems | LangGraph orchestration |
| RAG Pipeline | FAISS + PubMed + SentenceTransformers |
| Novel Metric Design | Position Drift Score (PDS) |
| QLoRA Fine-Tuning | Moderator fine-tuning on Llama 3.1 8B |
| Uncertainty Quantification | PDS components |
| Human-in-the-Loop AI | HITL escalation |
| Experiment Design | 5 RQs + ablations + baselines |
| Data Science | Pandas, NumPy, Matplotlib analysis |
| Responsible AI | Ethical framing, safety-first design |
| Full-Stack Deployment | Streamlit Cloud demo |
| Research Writing | 8-page conference paper |

---

*MedDebate-RAG | Frontiers of LLMs | Novel Contribution: Position Drift Score (PDS)*
*Pipeline: Debate + RAG + PDS (UQ) + HITL + QLoRA Fine-Tuning*
