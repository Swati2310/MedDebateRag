def compute_final_disagreement(conf_a: list[float], conf_b: list[float]) -> float:
    """
    Final Disagreement — Component 3 of PDS.

    Measures the absolute confidence gap between agents in the
    last round. Still far apart at end = unresolved conflict.

    Returns a value in [0, 1].
    """
    gap = abs(conf_a[-1] - conf_b[-1]) / 100.0
    return float(gap)
