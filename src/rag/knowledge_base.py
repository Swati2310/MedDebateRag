import json
import pickle
import time

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# PubMed Fetching
# ---------------------------------------------------------------------------

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
    "abdominal pain differential diagnosis",
]


def fetch_pubmed_abstracts(search_terms=None, max_per_term=200):
    """Fetch PubMed abstracts for medical topics. Free API — no key needed."""
    if search_terms is None:
        search_terms = SEARCH_TERMS

    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    all_docs = []

    for term in tqdm(search_terms, desc="Fetching PubMed"):
        search_url = f"{base_url}esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": term,
            "retmax": max_per_term,
            "retmode": "json",
        }

        r = requests.get(search_url, params=params, timeout=30)
        pmids = r.json()["esearchresult"]["idlist"]

        if not pmids:
            continue

        fetch_url = f"{base_url}efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "text",
        }

        r = requests.get(fetch_url, params=fetch_params, timeout=60)
        abstracts = r.text.split("\n\n")

        for abstract in abstracts:
            if len(abstract) > 100:
                all_docs.append({"term": term, "abstract": abstract.strip()})

        time.sleep(0.5)  # Be respectful to PubMed API

    return all_docs


# ---------------------------------------------------------------------------
# FAISS Knowledge Base
# ---------------------------------------------------------------------------

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
        texts = [doc["abstract"] for doc in self.documents]

        print("Embedding documents...")
        embeddings = self.embedder.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        faiss.normalize_L2(embeddings)

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


# ---------------------------------------------------------------------------
# Entry point — run once to build knowledge base
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    os.makedirs("data/knowledge_base", exist_ok=True)

    print("Fetching PubMed abstracts...")
    docs = fetch_pubmed_abstracts(SEARCH_TERMS, max_per_term=200)
    print(f"Fetched {len(docs)} documents")

    with open("data/knowledge_base/pubmed_abstracts.json", "w") as f:
        json.dump(docs, f)
    print("Saved to data/knowledge_base/pubmed_abstracts.json")

    print("\nBuilding FAISS index...")
    kb = MedicalKnowledgeBase()
    kb.load_documents("data/knowledge_base/pubmed_abstracts.json")
    kb.build_index()
    kb.save("data/knowledge_base")
    print("Done!")
