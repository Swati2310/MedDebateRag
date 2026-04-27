import numpy as np


def compute_confidence_drift(conf_a: list[float], conf_b: list[float]) -> float:
    """
    Confidence Drift — Component 1 of PDS.

    Measures how unstable each agent's numeric confidence was
    across debate rounds. High standard deviation = unstable.

    Returns a value in [0, 1].
    """
    std_a = np.std(conf_a) / 100.0
    std_b = np.std(conf_b) / 100.0
    return float((std_a + std_b) / 2.0)
