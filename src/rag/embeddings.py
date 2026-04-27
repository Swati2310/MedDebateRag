from sentence_transformers import SentenceTransformer

_model = None


def get_embedder(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Return a cached SentenceTransformer instance."""
    global _model
    if _model is None:
        _model = SentenceTransformer(model_name)
    return _model
