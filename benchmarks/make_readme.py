import json
from pathlib import Path

from benchmarks.evaluate_identifiability import METHODS
from data.build_datasets import SPECIALIST

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def ms(d, digits=3):
    return f"{d['mean']:.{digits}f} ± {d['std']:.{digits}f}"


def build():
    table = json.loads((RESULTS / "benchmark_table.json").read_text())
    s, seeds = table["summary"], table["seeds"]
    gates = {seed: json.loads((RESULTS / f"seed_{seed}" / "gate.json").read_text()) for seed in seeds}
    best_f1 = max(METHODS, key=lambda m: s[m]["mean_f1"]["mean"])
    fastest = min(METHODS, key=lambda m: s[m]["runtime_s"]["mean"])
    best_hub = max(METHODS, key=lambda m: s[m]["hub_recovery"]["mean"])
    least_leak = min(METHODS, key=lambda m: s[m]["cross_task_leakage"]["mean"])
    gk = s["Group-KNN"]
    method_rows = "\n".join(
        f"| {m} | {ms(s[m]['mean_f1'])} | {ms(s[m]['runtime_s'], 2)} | {ms(s[m]['forward_passes'], 0)} | {ms(s[m]['hub_recovery'])} | {ms(s[m]['cross_task_leakage'])} |"
        for m in METHODS
    )
    task_rows = "\n".join(
        f"| {m} | {t[5:].title()} | {ms(s[m]['tasks'][t]['precision'])} | {ms(s[m]['tasks'][t]['recall'])} | {ms(s[m]['tasks'][t]['f1'])} | {ms(s[m]['tasks'][t]['faithfulness'])} | {ms(s[m]['tasks'][t]['minimality'])} |"
        for m in METHODS for t in SPECIALIST
    )
    gate_rows = "\n".join(
        f"| {seed} | {g[-1]['lambda_off']} | {len(g)} | " + " | ".join(f"{g[-1][t]['keep']:.3f} / {g[-1][t]['remove']:.3f} | {g[-1][t]['gt_nodes_individually_necessary']}/{g[-1][t]['gt_size']}" for t in SPECIALIST) + " |"
        for seed, g in gates.items()
    )
    verdict = (
        f"Highest mean F1: **{best_f1}** ({s[best_f1]['mean_f1']['mean']:.3f}). "
        f"Fastest: **{fastest}** ({s[fastest]['runtime_s']['mean']:.2f}s per seed). "
        f"Best hub recovery: **{best_hub}**. Lowest cross-task leakage: **{least_leak}**. "
        f"Pareto front on F1 vs runtime: **{', '.join(table['pareto_front_f1_vs_runtime'])}**. "
        f"Group-KNN: mean F1 {gk['mean_f1']['mean']:.3f}, runtime {gk['runtime_s']['mean']:.2f}s, "
        f"{gk['forward_passes']['mean']:.0f} forward passes (each pass scores all three tasks at once)."
    )
    text = f"""# circuit-bench: a ground-truth benchmark for circuit discovery

Circuit-discovery methods are usually graded on models where nobody knows the true circuit. circuit-bench trains a small transformer with **planted, verified circuits**, so every method's output can be scored as precision and recall against a known answer.

## Problem

Mechanistic-interpretability methods such as ACDC and attribution patching claim to find "the circuit" for a task. On real models there is no answer key: a method that returns a plausible subgraph cannot be distinguished from one that returns the right subgraph. Method comparisons fall back on faithfulness alone, which rewards large circuits and says nothing about whether the recovered nodes are the ones the model actually uses.

## What's missing

Why current evaluation falls short:
- Faithfulness is satisfiable by many different node sets, so it cannot measure identifiability.
- Hand-annotated circuits (IOI, greater-than) are few, partial and only cover one task each.
- Shared ("hub") nodes used by several tasks are rarely tested, even though polysemantic components are the norm.

Why this approach:
- A **skeletal transformer** where every attention head and every MLP module is a separate, addressable node (72 nodes = 6 layers × (8 heads + 4 MLP modules)).
- **Planted circuits per seed**: each task gets a randomized, non-contiguous set of heads and MLPs over a staggered layer span, with {round(100 * table['config']['hub_fraction'])}% of circuit nodes shared between two tasks.
- Training has two phases. First a **masked warm-up**: specialist batches run with off-circuit nodes zeroed, so each planted circuit learns its task. Then **all nodes stay active** with an L2 output penalty on off-circuit nodes plus **interchange-intervention training**: the clean target must survive patching every off-circuit node from the corrupt prompt, and patching the circuit must flip the output to the corrupt target. The full, unmasked model is what gets evaluated. An L2 penalty alone failed the gate in testing, because off-circuit nodes kept carrying the task signal.
- A **hard validation gate**: the planted circuit alone must keep ≥ {table['config']['gate_keep_min']} faithfulness and removing it must drop faithfulness to ≤ {table['config']['gate_remove_max']}, or the run halts.
- Three methods are compared on identical data and identical patching: Group-KNN (proposed), Greedy-ACDC (node-level) and Gradient Attribution.

## Solution

```mermaid
flowchart LR
    A[Synthetic task generators<br/>syntax / arith / rhyme / general] --> B[Word-level tokenizer<br/>offline, no downloads]
    S[Seed] --> L[Random circuit layout<br/>staggered spans, hubs,<br/>non-contiguous heads]
    B --> T[Train SkeletalTransformer<br/>masked warm-up, then all nodes on<br/>+ L2 off-circuit penalty + IIT]
    L --> T
    T --> G{{Validation gate<br/>keep ≥ {table['config']['gate_keep_min']}, remove ≤ {table['config']['gate_remove_max']}}}
    G -- fail --> R[Retry with stronger penalty<br/>then halt]
    G -- pass --> H[CircuitBench harness<br/>HookableModelProtocol<br/>activation patching]
    H --> K[Group-KNN]
    H --> C[Greedy-ACDC]
    H --> D[Gradient Attribution]
    K & C & D --> M[Precision / Recall / F1 / faithfulness /<br/>minimality / runtime / hubs / leakage]
```

Tasks (all generated locally, minimal clean/corrupt pairs with different single-token targets):
- **Syntax**: subject-verb agreement across a distractor, `the keys near the door` → `are` vs `the key near the door` → `is`.
- **Arithmetic**: `14 + 23 =` → `37` vs `14 + 25 =` → `39` (10 ≤ A, B ≤ 49).
- **Rhyme**: words are split into onset + rime tokens, `rhyme : l ight -> n ight . s ake -> m` → `ake`. Train, val and test use disjoint word pairs (140/30/30).
- **General**: synthetic multi-clause English, trained with next-token loss at 0.2 weight, using all nodes.

The harness only depends on `discovery/protocol.py::HookableModelProtocol` (`node_names`, `node_layers`, and a forward that supports `record` / `patch`), so any PyTorch transformer that implements it can be benchmarked.

## Outcome

Results over seeds {seeds} on `{table['device']}` with {table['max_steps']} training steps per seed. Values are mean ± std across seeds.

{verdict}

### Method summary

| Method | Mean F1 | Runtime (s) | Forward passes | Hub recovery | Cross-task leakage |
|---|---|---|---|---|---|
{method_rows}

### Per-task results

| Method | Task | Precision | Recall | F1 | Faithfulness | Minimality |
|---|---|---|---|---|---|---|
{task_rows}

### Ground-truth validation gate (val split)

Each cell shows keep / remove faithfulness, then how many ground-truth nodes individually drop faithfulness by ≥ {table['config']['knn_prune']}.

| Seed | λ_off used | Attempts | Syntax keep / remove | Syntax nodes needed | Arith keep / remove | Arith nodes needed | Rhyme keep / remove | Rhyme nodes needed |
|---|---|---|---|---|---|---|---|---|
{gate_rows}

![F1 comparison](results/f1_comparison.png)
![Circuit heatmap](results/circuit_heatmap.png)
![Group-KNN clusters](results/knn_task_clusters.png)

## Limitations

- The gate proves the planted set is sufficient and necessary **as a whole**. Individual planted nodes can be redundant (see "nodes needed" above), which caps the recall any node-level method can honestly reach.
- Node-level only: edges between nodes are not scored, so Greedy-ACDC here is a node-level variant of ACDC.
- Group-KNN scores a window by ablating it (patching the window with corrupt activations). Keeping only an 8-node window and corrupting everything else gives near-zero faithfulness for most tasks, because each circuit spans 10+ nodes over several layers.
- Runtime depends on hardware; forward-pass counts are the hardware-independent cost measure.

## Run it

```bash
python -m data.build_datasets
python -m pytest -q
python -m benchmarks.evaluate_identifiability --seeds 0 1 2 --device mps
python -m benchmarks.plot_results
python -m benchmarks.make_readme
```

## Structure

```
configs/      model, training and benchmark settings
data/         offline task generators and tokenizer
model/        SkeletalTransformer (every head and MLP module is a node)
train/        training with the off-circuit L2 penalty
discovery/    HookableModelProtocol, CircuitBench harness, Group-KNN, Greedy-ACDC, Gradient Attribution
benchmarks/   layout generator, gate + evaluation, plots, README builder
tests/        pytest suite
results/      per-seed checkpoints, layouts, gate logs, predictions, benchmark_table.json, figures
```
"""
    (ROOT / "README.md").write_text(text)
    print("wrote README.md")


if __name__ == "__main__":
    build()
