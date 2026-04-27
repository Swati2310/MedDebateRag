from sentence_transformers import SentenceTransformer, util

_embedder = None


def _get_embedder(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(model_name)
    return _embedder


def compute_semantic_drift(args_a: list[str], args_b: list[str]) -> float:
    """
    Semantic Drift — Component 2 of PDS (the novel part).

    Compares Round 1 argument embedding vs Final Round embedding
    for each agent. High cosine distance = position changed.

    Returns a value in [0, 1].
    """
    if len(args_a) < 2 or len(args_b) < 2:
        return 0.0

    embedder = _get_embedder()

    emb_a_first = embedder.encode(args_a[0],  convert_to_tensor=True)
    emb_a_last  = embedder.encode(args_a[-1], convert_to_tensor=True)
    emb_b_first = embedder.encode(args_b[0],  convert_to_tensor=True)
    emb_b_last  = embedder.encode(args_b[-1], convert_to_tensor=True)

    sim_a = util.cos_sim(emb_a_first, emb_a_last).item()
    sim_b = util.cos_sim(emb_b_first, emb_b_last).item()

    drift_a = 1.0 - max(sim_a, 0)
    drift_b = 1.0 - max(sim_b, 0)

    return float((drift_a + drift_b) / 2.0)
