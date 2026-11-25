import torch

from discovery.metrics import logit_diff


def greedy_acdc(bench, task, tau):
    order = sorted(bench.nodes, key=lambda n: (bench.node_layers[n], bench.nodes.index(n)), reverse=True)
    removed = set()
    current = bench.ablate(removed, task)[task]
    for node in order:
        f = bench.ablate(removed | {node}, task)[task]
        if current - f < tau:
            removed.add(node)
            current = f
    return set(bench.nodes) - removed


def gradient_attribution(bench, task, target):
    sl = bench.slices[task]
    cache = {}
    with torch.enable_grad():
        logits = bench.model(bench.clean[sl], patch_cache=cache, patch_mode="record")
        ld = logit_diff(logits, bench.pos[sl], bench.cid[sl], bench.xid[sl]).mean()
        grads = torch.autograd.grad(ld, [cache[n] for n in bench.nodes])
    bench.calls += 1
    scores = {
        n: abs(((bench.corrupt_cache[n][sl] - cache[n].detach()) * g).sum(dim=(1, 2)).mean().item())
        for n, g in zip(bench.nodes, grads)
    }
    ranked = sorted(bench.nodes, key=lambda n: -scores[n])
    for k in range(1, len(ranked) + 1):
        if bench.keep(ranked[:k], task)[task] >= target:
            return set(ranked[:k])
    return set(ranked)
