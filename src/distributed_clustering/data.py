import numpy as np
from .runtime import normalize


def cluster_demo(rows=8000, dim=64, clusters=12, outlier_fraction=0.03, seed=42):
    if rows < 2 or dim < 2 or not 1 <= clusters <= rows or not 0 <= outlier_fraction < 0.5:
        raise ValueError("Invalid clustering demo dimensions")
    rng = np.random.default_rng(seed)
    prototypes = normalize(rng.normal(size=(clusters, dim)))
    truth = rng.integers(clusters, size=rows)
    x = normalize(prototypes[truth] + rng.normal(scale=0.055, size=(rows, dim)))
    outlier_count = int(rows*outlier_fraction)
    anomalous = np.zeros(rows, dtype=bool)
    if outlier_count:
        chosen = rng.choice(rows, size=outlier_count, replace=False)
        x[chosen] = normalize(rng.normal(size=(outlier_count, dim)))
        anomalous[chosen] = True
        truth[chosen] = -1
    return x, truth, anomalous
