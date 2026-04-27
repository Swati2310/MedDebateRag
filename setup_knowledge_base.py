"""
One-time setup script: fetch PubMed abstracts and build FAISS index.

Run this FIRST before any experiments.

Usage:
    python setup_knowledge_base.py
"""

import os

from dotenv import load_dotenv

load_dotenv()

from src.rag.knowledge_base import MedicalKnowledgeBase, SEARCH_TERMS, fetch_pubmed_abstracts
import json

KB_PATH     = "data/knowledge_base"
PUBMED_FILE = f"{KB_PATH}/pubmed_abstracts.json"

os.makedirs(KB_PATH, exist_ok=True)

# Step 1: Fetch PubMed abstracts (skip if already done)
if not os.path.exists(PUBMED_FILE):
    print("Fetching PubMed abstracts...")
    docs = fetch_pubmed_abstracts(SEARCH_TERMS, max_per_term=200)
    print(f"Fetched {len(docs)} documents")
    with open(PUBMED_FILE, "w") as f:
        json.dump(docs, f)
    print(f"Saved to {PUBMED_FILE}")
else:
    print(f"PubMed abstracts already exist at {PUBMED_FILE} — skipping fetch.")

# Step 2: Build FAISS index
print("\nBuilding FAISS index...")
kb = MedicalKnowledgeBase()
kb.load_documents(PUBMED_FILE)
kb.build_index()
kb.save(KB_PATH)
print("Setup complete!")
