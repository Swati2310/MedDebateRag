import faiss
import numpy as np

from src.rag.knowledge_base import MedicalKnowledgeBase


class MedicalRetriever:

    def __init__(self, knowledge_base: MedicalKnowledgeBase, top_k: int = 5):
        self.kb = knowledge_base
        self.top_k = top_k

    def retrieve(self, query: str) -> list[dict]:
        """Return top-k relevant medical documents for a query."""
        query_embedding = self.kb.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)

        scores, indices = self.kb.index.search(query_embedding, self.top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append({
                    "abstract": self.kb.documents[idx]["abstract"],
                    "score": float(score),
                })
        return results

    def format_for_prompt(self, query: str) -> str:
        """Retrieve and format as prompt-ready string."""
        docs = self.retrieve(query)
        return "\n\n".join([
            f"[Evidence {i+1} | Relevance: {doc['score']:.2f}]:\n{doc['abstract'][:600]}"
            for i, doc in enumerate(docs)
        ])


def load_retriever(kb_path: str = "data/knowledge_base", top_k: int = 5) -> MedicalRetriever:
    """Convenience loader — loads KB from disk and returns a retriever."""
    kb = MedicalKnowledgeBase()
    kb.load(kb_path)
    return MedicalRetriever(kb, top_k=top_k)
