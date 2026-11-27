import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from data.build_datasets import SPECIALIST, TaskData, load_split
from discovery.baseline_acdc import gradient_attribution, greedy_acdc
from discovery.group_knn_discovery import group_knn
from discovery.harness import CircuitBench
from discovery.metrics import harmonic, precision_recall_f1
from model.skeletal_transformer import SkeletalTransformer
from train.train_skeletal import BENCH_CONFIG, TRAIN_CONFIG, load_checkpoint, train

ROOT = Path(__file__).resolve().parent.parent
CFG = BENCH_CONFIG
METHODS = ["Greedy-ACDC", "Gradient-Attribution", "Group-KNN"]
TASK_METRICS = ["precision", "recall", "f1", "effective_recall", "effective_f1", "faithfulness", "minimality", "size"]
METHOD_METRICS = ["mean_f1", "mean_effective_f1", "runtime_s", "forward_passes", "hub_recovery", "cross_task_leakage"]


def subset(data: dict[str, TaskData], n: int) -> dict[str, TaskData]:
    return {t: {k: v[:n] for k, v in data[t].items()} for t in SPECIALIST}


def gate(model: SkeletalTransformer, layout: dict[str, Any], device: str) -> dict[str, Any]:
    bench = CircuitBench(model, subset(load_split("val"), CFG["n_eval_samples"]), device)
    circuits: dict[str, list[str]] = layout["circuits"]
    out: dict[str, Any] = {}
    for t in SPECIALIST:
        c = circuits[t]
        needed = [n for n in c if 1.0 - bench.ablate({n}, t)[t] >= CFG["needed_drop"]]
        out[t] = {
            "keep": bench.keep(c, t)[t],
            "remove": bench.ablate(c, t)[t],
            "gt_size": len(c),
            "gt_nodes_individually_necessary": len(needed),
            "needed_nodes": needed,
        }
    out["passed"] = all(out[t]["keep"] >= CFG["gate_keep_min"] and out[t]["remove"] <= CFG["gate_remove_max"] for t in SPECIALIST)
    return out


def report(seed: int, attempt: dict[str, Any]) -> None:
    cells = " ".join(f"{t[5:]} keep={attempt[t]['keep']:.3f} remove={attempt[t]['remove']:.3f}" for t in SPECIALIST)
    print(f"seed {seed} gate (lambda_off={attempt['lambda_off']}): {cells}", flush=True)


def train_and_gate(
    seed: int, device: str, max_steps: int, batch_size: int, reuse: bool
) -> tuple[SkeletalTransformer, dict[str, Any], dict[str, list[str]]]:
    out_dir = ROOT / "results" / f"seed_{seed}"
    ckpt = out_dir / "skeletal_model.pt"
    if reuse and ckpt.exists():
        model, layout, lam = load_checkpoint(ckpt, device)
    else:
        lam = TRAIN_CONFIG["lambda_off"]
        model, layout = train(seed, out_dir, device, max_steps, batch_size, lam, TRAIN_CONFIG["iit_weight"])
    attempts: list[dict[str, Any]] = [{"lambda_off": lam, **gate(model, layout, device)}]
    report(seed, attempts[-1])
    if not attempts[-1]["passed"]:
        lam = TRAIN_CONFIG["lambda_off_retry"]
        model, layout = train(seed, out_dir, device, max_steps, batch_size, lam, TRAIN_CONFIG["iit_weight"])
        attempts.append({"lambda_off": lam, **gate(model, layout, device)})
        report(seed, attempts[-1])
    (out_dir / "gate.json").write_text(json.dumps(attempts, indent=2))
    assert attempts[-1]["passed"], f"seed {seed} failed the ground-truth validation gate: {attempts[-1]}"
    needed: dict[str, list[str]] = {t: attempts[-1][t]["needed_nodes"] for t in SPECIALIST}
    return model, layout, needed


def hub_recovery(pred: dict[str, set[str]], gt: dict[str, set[str]], hubs: list[str]) -> float | None:
    if not hubs:
        return None
    return float(np.mean([all(h in pred[t] for t in SPECIALIST if h in gt[t]) for h in hubs]))


def cross_task_leakage(pred: dict[str, set[str]], gt: dict[str, set[str]]) -> float:
    rates: list[float] = []
    for t in SPECIALIST:
        others = set[str]().union(*[gt[o] for o in SPECIALIST if o != t]) - gt[t]
        rates.append(len(pred[t] & others) / max(len(pred[t]), 1))
    return float(np.mean(rates))


def evaluate_seed(
    model: SkeletalTransformer, layout: dict[str, Any], needed: dict[str, list[str]], device: str, seed: int
) -> dict[str, Any]:
    bench = CircuitBench(model, subset(load_split("test"), CFG["n_eval_samples"]), device)
    circuits: dict[str, list[str]] = layout["circuits"]
    gt = {t: set(circuits[t]) for t in SPECIALIST}
    need = {t: set(needed[t]) for t in SPECIALIST}
    preds: dict[str, dict[str, set[str]]] = {}
    cost: dict[str, tuple[float, int]] = {}

    bench.reset()
    start = time.perf_counter()
    preds["Greedy-ACDC"] = {t: greedy_acdc(bench, t, CFG["acdc_tau"]) for t in SPECIALIST}
    cost["Greedy-ACDC"] = (time.perf_counter() - start, bench.calls)

    bench.reset()
    start = time.perf_counter()
    preds["Gradient-Attribution"] = {t: gradient_attribution(bench, t, CFG["grad_target_faithfulness"]) for t in SPECIALIST}
    cost["Gradient-Attribution"] = (time.perf_counter() - start, bench.calls)

    bench.reset()
    start = time.perf_counter()
    preds["Group-KNN"], records = group_knn(bench, CFG["knn_windows"], CFG["knn_clusters"], CFG["knn_neighbors"], CFG["knn_prune"], seed)
    cost["Group-KNN"] = (time.perf_counter() - start, bench.calls)

    bench.reset()
    metrics: dict[str, Any] = {}
    for m in METHODS:
        tasks: dict[str, dict[str, float]] = {}
        for t in SPECIALIST:
            p, r, f = precision_recall_f1(preds[m][t], gt[t])
            er = precision_recall_f1(preds[m][t], need[t])[1]
            tasks[t] = {
                "precision": p,
                "recall": r,
                "f1": f,
                "effective_recall": er,
                "effective_f1": harmonic(p, er),
                "faithfulness": bench.keep(preds[m][t], t)[t],
                "minimality": 1.0 - len(preds[m][t]) / len(bench.nodes),
                "size": float(len(preds[m][t])),
            }
        metrics[m] = {
            "tasks": tasks,
            "runtime_s": cost[m][0],
            "forward_passes": float(cost[m][1]),
            "mean_f1": float(np.mean([tasks[t]["f1"] for t in SPECIALIST])),
            "mean_effective_f1": float(np.mean([tasks[t]["effective_f1"] for t in SPECIALIST])),
            "hub_recovery": hub_recovery(preds[m], gt, layout["hubs"]),
            "cross_task_leakage": cross_task_leakage(preds[m], gt),
        }
    out_dir = ROOT / "results" / f"seed_{seed}"
    (out_dir / "predictions.json").write_text(json.dumps({m: {t: sorted(p) for t, p in preds[m].items()} for m in METHODS}, indent=2))
    (out_dir / "knn_windows.json").write_text(json.dumps(records, indent=2))
    metrics["ground_truth_test_faithfulness"] = {t: bench.keep(gt[t], t)[t] for t in SPECIALIST}
    return metrics


def stat(values: list[Any]) -> dict[str, float]:
    kept = [float(v) for v in values if v is not None]
    return {"mean": float(np.mean(kept)), "std": float(np.std(kept, ddof=1)) if len(kept) > 1 else 0.0}


def aggregate(per_seed: dict[str, dict[str, Any]]) -> dict[str, Any]:
    seeds = list(per_seed.values())
    summary: dict[str, Any] = {}
    for m in METHODS:
        tasks = {t: {k: stat([s[m]["tasks"][t][k] for s in seeds]) for k in TASK_METRICS} for t in SPECIALIST}
        summary[m] = {"tasks": tasks, **{k: stat([s[m][k] for s in seeds]) for k in METHOD_METRICS}}
    point = {m: (summary[m]["mean_f1"]["mean"], summary[m]["runtime_s"]["mean"]) for m in METHODS}
    pareto = [
        m
        for m in METHODS
        if not any(o != m and point[o][0] >= point[m][0] and point[o][1] <= point[m][1] and point[o] != point[m] for o in METHODS)
    ]
    return {"summary": summary, "pareto_front_f1_vs_runtime": pareto}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=CFG["seeds"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--max_steps", type=int, default=TRAIN_CONFIG["max_steps"])
    p.add_argument("--batch_size", type=int, default=TRAIN_CONFIG["batch_size"])
    p.add_argument("--reuse", action="store_true")
    p.add_argument("--output", default=str(ROOT / "results" / "benchmark_table.json"))
    a = p.parse_args()
    per_seed: dict[str, dict[str, Any]] = {}
    for seed in a.seeds:
        model, layout, needed = train_and_gate(seed, a.device, a.max_steps, a.batch_size, a.reuse)
        result = evaluate_seed(model, layout, needed, a.device, seed)
        per_seed[str(seed)] = result
        line = " ".join(
            f"{m} F1={result[m]['mean_f1']:.3f} effF1={result[m]['mean_effective_f1']:.3f} t={result[m]['runtime_s']:.1f}s" for m in METHODS
        )
        print(f"seed {seed}: {line}", flush=True)
    table = {"seeds": a.seeds, "device": a.device, "max_steps": a.max_steps, "config": CFG, **aggregate(per_seed), "per_seed": per_seed}
    Path(a.output).write_text(json.dumps(table, indent=2))
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
