import torch
from torch import Tensor

from discovery.harness import CircuitBench
from discovery.metrics import logit_diff


def greedy_acdc(bench: CircuitBench, task: str, tau: float) -> set[str]:
    order = sorted(bench.nodes, key=lambda n: (bench.node_layers[n], bench.nodes.index(n)), reverse=True)
    removed: set[str] = set()
    current = bench.ablate(removed, task)[task]
    for node in order:
        f = bench.ablate(removed | {node}, task)[task]
        if current - f < tau:
            removed.add(node)
            current = f
    return set(bench.nodes) - removed


def gradient_attribution(bench: CircuitBench, task: str, target: float) -> set[str]:
    a, b = bench.rows(task)
    cache: dict[str, Tensor] = {}
    with torch.enable_grad():
        logits = bench.model(bench.clean[a:b], patch_cache=cache, patch_mode="record")
        ld = logit_diff(logits, bench.pos[a:b], bench.cid[a:b], bench.xid[a:b]).mean()
        grads = torch.autograd.grad(ld, [cache[n] for n in bench.nodes])
    bench.calls += 1
    scores: dict[str, float] = {}
    for n, g in zip(bench.nodes, grads, strict=True):
        delta = bench.corrupt_cache[n][a:b] - cache[n].detach()
        scores[n] = abs(float((delta * g).sum(dim=(1, 2)).mean().item()))
    ranked = sorted(bench.nodes, key=lambda n: -scores[n])
    for k in range(1, len(ranked) + 1):
        if bench.keep(ranked[:k], task)[task] >= target:
            return set(ranked[:k])
    return set(ranked)
