import json
from pathlib import Path
from typing import Any

from benchmarks.evaluate_identifiability import METHODS
from data.build_datasets import SPECIALIST

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def ms(d: dict[str, float], digits: int = 3) -> str:
    return f"{d['mean']:.{digits}f} ± {d['std']:.{digits}f}"


def build() -> None:
    table: dict[str, Any] = json.loads((RESULTS / "benchmark_table.json").read_text())
    s: dict[str, Any] = table["summary"]
    seeds: list[int] = table["seeds"]
    cfg: dict[str, Any] = table["config"]
    gates: dict[int, dict[str, Any]] = {}
    attempts: dict[int, int] = {}
    for seed in seeds:
        log: list[dict[str, Any]] = json.loads((RESULTS / f"seed_{seed}" / "gate.json").read_text())
        gates[seed] = log[-1]
        attempts[seed] = len(log)

    def mean_of(m: str, key: str) -> float:
        return float(s[m][key]["mean"])

    best_f1 = max(METHODS, key=lambda m: mean_of(m, "mean_f1"))
    fastest = min(METHODS, key=lambda m: mean_of(m, "runtime_s"))
    best_hub = max(METHODS, key=lambda m: mean_of(m, "hub_recovery"))
    pareto: list[str] = table["pareto_front_f1_vs_runtime"]
    planted = sum(gates[seed][t]["gt_size"] for seed in seeds for t in SPECIALIST)
    needed = sum(gates[seed][t]["gt_nodes_individually_necessary"] for seed in seeds for t in SPECIALIST)
    ratios = [gates[seed][t]["gt_nodes_individually_necessary"] / gates[seed][t]["gt_size"] for seed in seeds for t in SPECIALIST]

    pareto_text = (
        f"**{best_f1}** has the highest mean F1 ({mean_of(best_f1, 'mean_f1'):.3f}) and **{fastest}** is the fastest "
        f"({mean_of(fastest, 'runtime_s'):.2f}s, {mean_of(fastest, 'forward_passes'):.0f} forward passes per seed). "
        f"Pareto front on F1 vs runtime: **{', '.join(pareto)}**."
    )
    hub_text = (
        f"**{best_hub}** recovers the most shared hub nodes in every task that uses them "
        f"({mean_of(best_hub, 'hub_recovery'):.3f}); see the hub recovery column below for all methods."
    )
    compression_text = (
        f"After training, {needed} of {planted} planted nodes ({100 * needed / planted:.0f}%) are individually needed "
        f"(range {100 * min(ratios):.0f}–{100 * max(ratios):.0f}% per task and seed). The rest are redundant slack that "
        f"a minimal method can legitimately skip, which is why effective recall is reported next to raw recall."
    )
    method_rows = "\n".join(
        f"| {m} | {ms(s[m]['mean_f1'])} | {ms(s[m]['mean_effective_f1'])} | {ms(s[m]['runtime_s'], 2)} | "
        f"{ms(s[m]['forward_passes'], 0)} | {ms(s[m]['hub_recovery'])} | {ms(s[m]['cross_task_leakage'])} |"
        for m in METHODS
    )
    task_rows = "\n".join(
        f"| {m} | {t[5:].title()} | {ms(s[m]['tasks'][t]['precision'])} | {ms(s[m]['tasks'][t]['recall'])} | {ms(s[m]['tasks'][t]['f1'])} | "
        f"{ms(s[m]['tasks'][t]['effective_recall'])} | {ms(s[m]['tasks'][t]['effective_f1'])} | {ms(s[m]['tasks'][t]['faithfulness'])} |"
        for m in METHODS
        for t in SPECIALIST
    )
    gate_rows = "\n".join(
        f"| {seed} | {gates[seed]['lambda_off']} | {attempts[seed]} | "
        + " | ".join(
            f"{gates[seed][t]['keep']:.3f} / {gates[seed][t]['remove']:.3f} | "
            f"{gates[seed][t]['gt_nodes_individually_necessary']}/{gates[seed][t]['gt_size']}"
            for t in SPECIALIST
        )
        + " |"
        for seed in seeds
    )
    text = f"""# circuit-bench: a controlled multi-task circuit identifiability benchmark

circuit-bench trains a small transformer in which three tasks run concurrently on planted circuits that share hub nodes, verifies every planted circuit with a hard causal gate, and then scores circuit-discovery methods against that known answer key with precision, recall, F1, faithfulness, minimality and cost.

## Key findings

Seeds {seeds}, `{table["device"]}`, {table["max_steps"]} training steps per seed, mean ± std across seeds.

1. **Speed vs accuracy.** {pareto_text}
2. **Shared polysemantic hubs.** {hub_text}
3. **Post-IIT circuit compression.** {compression_text}

## Problem

Circuit-discovery methods are usually judged on models where nobody knows the true circuit, so faithfulness is the only available score. Many different node sets can be faithful, so faithfulness alone cannot say whether a method found the nodes the model actually uses. A benchmark needs models whose circuits are known and causally verified.

## Benchmark design

- **Skeletal transformer.** Every attention head and every MLP module is a separate, addressable node: 72 nodes = 6 layers × (8 heads + 4 MLP modules), about 4.9M parameters.
- **Tasks**, all generated offline as minimal clean/corrupt pairs with different single-token targets:
  - Syntax: subject-verb agreement across a distractor, `the keys near the door` → `are` vs `the key near the door` → `is`.
  - Arithmetic: `14 + 23 =` → `37` vs `14 + 25 =` → `39` (10 ≤ A, B ≤ 49).
  - Rhyme: words split into onset + rime tokens, `rhyme : l ight -> n ight . s ake -> m` → `ake`. Train, val and test use disjoint word pairs (140/30/30).
  - General: synthetic multi-clause English with next-token loss at 0.2 weight, using all nodes.
- **Randomized layouts per seed.** Each task gets a staggered span of layers and a non-contiguous set of heads plus one MLP module per layer; about {round(100 * cfg["hub_fraction"])}% of circuit nodes are hubs shared by exactly two tasks.
- **Training.** A masked warm-up (off-circuit nodes zeroed for specialist batches), then all nodes active with an L2 penalty on off-circuit outputs plus interchange-intervention training: patching every off-circuit node from the corrupt prompt must keep the clean answer, and patching the circuit must produce the corrupt answer.
- **Validation gate.** On the validation split, the planted circuit alone must keep faithfulness ≥ {cfg["gate_keep_min"]} and removing it must drop faithfulness to ≤ {cfg["gate_remove_max"]}; otherwise training retries once with a stronger penalty and then halts.
- **Methods.** Greedy-ACDC (node-level, reverse topological, τ = {cfg["acdc_tau"]}), Gradient Attribution (attribution patching, smallest top-k reaching {cfg["grad_target_faithfulness"]} faithfulness) and Group-KNN (overlapping window ablation, k-means plus k-nearest-neighbour task clustering, recursive binary splitting). All three share one patching harness that depends only on `discovery/protocol.py::HookableModelProtocol`, so any PyTorch transformer that implements it can be benchmarked.
- **Metrics.** Precision is always measured against all planted nodes. Recall and F1 are reported twice: raw (against all planted nodes) and effective (against the individually needed nodes, those whose single ablation drops faithfulness by ≥ {cfg["needed_drop"]}).

```mermaid
flowchart LR
    A[Offline task generators] --> T[Train SkeletalTransformer<br/>masked warm-up, then all nodes on<br/>+ L2 off-circuit penalty + IIT]
    S[Seed] --> L[Random layout<br/>staggered spans, hubs,<br/>non-contiguous heads] --> T
    T --> G{{Validation gate}}
    G -- fail --> R[Retry with stronger penalty, then halt]
    G -- pass --> H[CircuitBench harness<br/>HookableModelProtocol]
    H --> C[Greedy-ACDC]
    H --> D[Gradient Attribution]
    H --> K[Group-KNN]
    C & D & K --> M[Precision, recall, F1, effective recall/F1,<br/>faithfulness, cost, hubs, leakage]
```

## Results

### Method summary

| Method | Mean F1 | Mean effective F1 | Runtime (s) | Forward passes | Hub recovery | Cross-task leakage |
|---|---|---|---|---|---|---|
{method_rows}

### Per-task results

| Method | Task | Precision | Recall | F1 | Effective recall | Effective F1 | Faithfulness |
|---|---|---|---|---|---|---|---|
{task_rows}

### Validation gate (val split)

Each task shows keep / remove faithfulness, then individually needed nodes / planted nodes.

| Seed | λ_off | Attempts | Syntax keep / remove | Syntax needed | Arith keep / remove | Arith needed | Rhyme keep / remove | Rhyme needed |
|---|---|---|---|---|---|---|---|---|
{gate_rows}

![F1 comparison](results/f1_comparison.png)
![Circuit heatmap](results/circuit_heatmap.png)
![Group-KNN clusters](results/knn_task_clusters.png)

## Related work

- **Tracr** (Lindner, Kramár, Farquhar, Rahtz, McGrath and Mikulik, 2023, arXiv:2301.05062) compiles human-readable RASP programs into standard decoder-only transformers. The compiled models have known structure, which the authors propose as ground truth for evaluating interpretability methods. Their weights are constructed rather than learned.
- **InterpBench** (Gupta, Arcuschin, Kwa and Garriga-Alonso, NeurIPS 2024 Datasets and Benchmarks) trains 86 semi-synthetic transformers (85 Tracr circuits plus a simplified IOI circuit) with Strict Interchange Intervention Training (SIIT). SIIT also intervenes on components outside the circuit and penalizes any change in the output, so non-circuit components cannot affect it. Each model implements one algorithmic circuit, and the paper evaluates ACDC, Subnetwork Probing, Edgewise SP, Edge Attribution Patching and EAP with integrated gradients at the edge level. Training models with multiple circuits is listed as future work.
- **Relation to circuit-bench.** The training objective here uses the same idea as SIIT: the sufficiency term (patch every off-circuit node from a corrupt prompt and require the clean answer) plays the role of SIIT's strictness loss, and full credit for that idea belongs to InterpBench. The differences are in what is planted and measured:
  - three tasks run concurrently in one model instead of one circuit per model;
  - about {round(100 * cfg["hub_fraction"])}% of circuit nodes are shared polysemantic hubs used by two tasks, scored with hub recovery and cross-task leakage;
  - layouts are procedurally randomized per seed (staggered layer spans, non-contiguous heads), so results are reported as mean ± std over layouts;
  - post-IIT compression is measured directly: how many planted nodes remain individually needed after training, reported through effective recall and effective F1.

## Limitations

1. **Node-level granularity only.** Heads and MLP modules are scored as nodes; edges between nodes are not scored, so Greedy-ACDC here is a node-level variant of ACDC.
2. **Whole-circuit gate vs individual redundancy.** The gate guarantees that the planted set is sufficient and necessary as a whole. Training then compresses the computation onto a subset of the planted nodes, leaving redundant slack nodes that cap the raw recall any minimal method can reach.
3. **OR-pair nuance in effective recall and effective F1.** `needed_nodes` tests necessity by single-node ablation. Two redundant nodes where either one alone sustains the task both look individually unneeded, so effective recall and effective F1 can flatter a method that recovers only one node of such a pair. Precision stays anchored to all planted nodes to limit this.

## Run it

```bash
uv sync
uv run python -m data.build_datasets
uv run pytest -q
uv run python -m benchmarks.evaluate_identifiability --seeds 0 1 2 --device mps
uv run python -m benchmarks.plot_results
uv run python -m benchmarks.make_readme
```

Quality checks run in CI on every push and pull request: `ruff format --check`, `ruff check`, `pyright` (strict) and `pytest`.

## Structure

```
configs/      model, training and benchmark settings
data/         offline task generators and tokenizer
model/        SkeletalTransformer (every head and MLP module is a node)
train/        masked warm-up, L2 off-circuit penalty and interchange-intervention training
discovery/    HookableModelProtocol, CircuitBench harness, Greedy-ACDC, Gradient Attribution, Group-KNN
benchmarks/   layout generator, gate and evaluation, plots, README builder
typings/      local type stubs for scikit-learn
tests/        pytest suite, including a no-comments check
results/      per-seed layouts, gate logs, predictions, benchmark_table.json and figures
```

## License

MIT
"""
    (ROOT / "README.md").write_text(text)
    print("wrote README.md")


if __name__ == "__main__":
    build()
