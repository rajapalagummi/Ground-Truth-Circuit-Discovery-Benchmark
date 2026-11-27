from collections import Counter
from collections.abc import Iterable
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from discovery.harness import CircuitBench

Window = tuple[str, ...]


def windows(nodes: list[str], sizes: list[int]) -> list[Window]:
    out: list[Window] = []
    for W in sizes:
        for start in range(0, len(nodes) - W + 1, max(W // 2, 1)):
            out.append(tuple(nodes[start : start + W]))
    return out


def group_knn(
    bench: CircuitBench, sizes: list[int], n_clusters: int, n_neighbors: int, prune: float, seed: int
) -> tuple[dict[str, set[str]], list[dict[str, Any]]]:
    tasks = bench.tasks

    def effect(group: Iterable[str]) -> dict[str, float]:
        return {t: 1.0 - f for t, f in bench.ablate(group).items()}

    wins = windows(bench.nodes, sizes)
    scores = np.array([[effect(w)[t] for t in tasks] for w in wins], dtype=np.float64)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(scores)
    names = [tasks[int(np.argmax(c))] if float(c.max()) >= prune else "NULL" for c in km.cluster_centers_]
    raw = [names[int(c)] for c in km.labels_]
    neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine").fit(scores + 1e-6)
    idx = neighbors.kneighbors(scores + 1e-6, return_distance=False)
    labels = [Counter(raw[int(j)] for j in row).most_common(1)[0][0] for row in idx]

    def refine(group: Window, t: str) -> set[str]:
        if len(group) == 1:
            return set(group)
        half = len(group) // 2
        out: set[str] = set()
        for part in (group[:half], group[half:]):
            if effect(part)[t] >= prune:
                out |= refine(part, t)
        return out

    pred: dict[str, set[str]] = {}
    for t in tasks:
        roots = [w for w, lab in zip(wins, labels, strict=True) if len(w) == max(sizes) and lab == t and effect(w)[t] >= prune]
        pred[t] = set[str]().union(*[refine(w, t) for w in roots])
    records: list[dict[str, Any]] = [
        {"nodes": list(w), "W": len(w), "scores": [float(v) for v in s], "label": lab}
        for w, s, lab in zip(wins, scores, labels, strict=True)
    ]
    return pred, records
