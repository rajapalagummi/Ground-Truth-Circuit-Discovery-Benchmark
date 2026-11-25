import itertools
import random

from data.build_datasets import SPECIALIST


def head(L, H):
    return f"blocks.{L}.attn.heads.{H}"


def mlp(L, M):
    return f"blocks.{L}.mlps.{M}"


def pick_noncontiguous(pool, k, rng):
    for size in range(min(k, len(pool)), 0, -1):
        options = [c for c in itertools.combinations(sorted(pool), size) if size == 1 or c[-1] - c[0] != size - 1]
        if options:
            return list(rng.choice(options))
    return []


def generate_layout(seed, node_names, node_layers, n_layers, n_heads, n_mlps, hub_fraction):
    rng = random.Random(seed)
    spans = {}
    for t, start in zip(SPECIALIST, rng.sample(range(3), 3)):
        spans[t] = list(range(start, min(start + rng.choice([3, 4]), n_layers)))
    circuits = {t: set() for t in SPECIALIST}
    for L in range(n_layers):
        free_heads, free_mlps = list(range(n_heads)), list(range(n_mlps))
        for t in rng.sample(SPECIALIST, 3):
            if L not in spans[t]:
                continue
            heads = pick_noncontiguous(free_heads, rng.choice([2, 3]), rng)
            m = rng.choice(free_mlps)
            free_heads = [h for h in free_heads if h not in heads]
            free_mlps.remove(m)
            circuits[t] |= {head(L, h) for h in heads} | {mlp(L, m)}
    n_hubs = round(hub_fraction * sum(len(c) for c in circuits.values()))
    candidates = [(n, t2) for t in SPECIALIST for n in sorted(circuits[t]) for t2 in SPECIALIST if t2 != t and node_layers[n] in spans[t2]]
    rng.shuffle(candidates)
    hubs = set()
    for n, t2 in candidates:
        if len(hubs) == n_hubs:
            break
        if n not in hubs:
            circuits[t2].add(n)
            hubs.add(n)
    order = {n: i for i, n in enumerate(node_names)}
    circuits = {t: sorted(c, key=order.get) for t, c in circuits.items()}
    circuits["TASK_GENERAL"] = list(node_names)
    return {"seed": seed, "spans": spans, "circuits": circuits, "hubs": sorted(hubs, key=order.get)}
