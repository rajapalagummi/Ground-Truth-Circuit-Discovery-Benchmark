from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


def windows(nodes, sizes):
    out = []
    for W in sizes:
        for start in range(0, len(nodes) - W + 1, max(W // 2, 1)):
            out.append(tuple(nodes[start:start + W]))
    return out


def group_knn(bench, sizes, n_clusters, n_neighbors, prune, seed):
    tasks = bench.tasks
    effect = lambda group: {t: 1.0 - f for t, f in bench.ablate(group).items()}
    wins = windows(bench.nodes, sizes)
    scores = np.array([[effect(w)[t] for t in tasks] for w in wins])
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(scores)
    names = [tasks[int(np.argmax(c))] if c.max() >= prune else "NULL" for c in km.cluster_centers_]
    raw = [names[c] for c in km.labels_]
    neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine").fit(scores + 1e-6)
    idx = neighbors.kneighbors(scores + 1e-6, return_distance=False)
    labels = [Counter(raw[j] for j in row).most_common(1)[0][0] for row in idx]

    def refine(group, t):
        if len(group) == 1:
            return set(group)
        half = len(group) // 2
        out = set()
        for part in (group[:half], group[half:]):
            if effect(part)[t] >= prune:
                out |= refine(part, t)
        return out

    pred = {}
    for t in tasks:
        roots = [w for w, l in zip(wins, labels) if len(w) == max(sizes) and l == t and effect(w)[t] >= prune]
        pred[t] = set().union(*[refine(w, t) for w in roots])
    records = [{"nodes": list(w), "W": len(w), "scores": s.tolist(), "label": l} for w, s, l in zip(wins, scores, labels)]
    return pred, records
