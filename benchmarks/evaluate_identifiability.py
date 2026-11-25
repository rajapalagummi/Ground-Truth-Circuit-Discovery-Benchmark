import argparse
import json
import time
from pathlib import Path

import numpy as np

from data.build_datasets import SPECIALIST, load_split
from discovery.baseline_acdc import gradient_attribution, greedy_acdc
from discovery.group_knn_discovery import group_knn
from discovery.harness import CircuitBench
from discovery.metrics import precision_recall_f1
from train.train_skeletal import BENCH_CONFIG as CFG
from train.train_skeletal import TRAIN_CONFIG, load_checkpoint, train

ROOT = Path(__file__).resolve().parent.parent
METHODS = ["Group-KNN", "Greedy-ACDC", "Gradient-Attribution"]
TASK_METRICS = ["precision", "recall", "f1", "faithfulness", "minimality", "size"]
METHOD_METRICS = ["mean_f1", "runtime_s", "forward_passes", "hub_recovery", "cross_task_leakage"]


def subset(data, n):
    return {t: {k: v[:n] for k, v in data[t].items()} for t in SPECIALIST}


def gate(model, layout, device):
    bench = CircuitBench(model, subset(load_split("val"), CFG["n_eval_samples"]), device)
    out = {}
    for t in SPECIALIST:
        c = layout["circuits"][t]
        necessary = [n for n in c if 1.0 - bench.ablate({n}, t)[t] >= CFG["knn_prune"]]
        out[t] = {"keep": bench.keep(c, t)[t], "remove": bench.ablate(c, t)[t], "gt_size": len(c), "gt_nodes_individually_necessary": len(necessary)}
    out["passed"] = all(out[t]["keep"] >= CFG["gate_keep_min"] and out[t]["remove"] <= CFG["gate_remove_max"] for t in SPECIALIST)
    return out


def train_and_gate(seed, device, max_steps, batch_size, reuse):
    out_dir = ROOT / "results" / f"seed_{seed}"
    ckpt = out_dir / "skeletal_model.pt"
    attempts = []
    for i, lam in enumerate([TRAIN_CONFIG["lambda_off"], TRAIN_CONFIG["lambda_off_retry"]]):
        if reuse and ckpt.exists() and i == 0:
            model, layout, lam = load_checkpoint(ckpt, device)
        else:
            model, layout = train(seed, out_dir, device, max_steps, batch_size, lam, TRAIN_CONFIG["iit_weight"])
        attempts.append({"lambda_off": lam, **gate(model, layout, device)})
        print(f"seed {seed} gate (lambda_off={lam}): " + " ".join(f"{t[5:]} keep={attempts[-1][t]['keep']:.3f} remove={attempts[-1][t]['remove']:.3f}" for t in SPECIALIST), flush=True)
        if attempts[-1]["passed"]:
            break
    (out_dir / "gate.json").write_text(json.dumps(attempts, indent=2))
    assert attempts[-1]["passed"], f"seed {seed} failed the ground-truth validation gate: {attempts[-1]}"
    return model, layout


def hub_recovery(pred, gt, hubs):
    if not hubs:
        return None
    return float(np.mean([all(h in pred[t] for t in SPECIALIST if h in gt[t]) for h in hubs]))


def cross_task_leakage(pred, gt):
    rates = []
    for t in SPECIALIST:
        others = set().union(*[gt[o] for o in SPECIALIST if o != t]) - gt[t]
        rates.append(len(pred[t] & others) / max(len(pred[t]), 1))
    return float(np.mean(rates))


def evaluate_seed(model, layout, device, seed):
    bench = CircuitBench(model, subset(load_split("test"), CFG["n_eval_samples"]), device)
    gt = {t: set(layout["circuits"][t]) for t in SPECIALIST}
    runs = {
        "Group-KNN": lambda: group_knn(bench, CFG["knn_windows"], CFG["knn_clusters"], CFG["knn_neighbors"], CFG["knn_prune"], seed),
        "Greedy-ACDC": lambda: ({t: greedy_acdc(bench, t, CFG["acdc_tau"]) for t in SPECIALIST}, None),
        "Gradient-Attribution": lambda: ({t: gradient_attribution(bench, t, CFG["grad_target_faithfulness"]) for t in SPECIALIST}, None),
    }
    preds, metrics, records = {}, {}, None
    for m in METHODS:
        bench.reset()
        start = time.perf_counter()
        pred, extra = runs[m]()
        runtime = time.perf_counter() - start
        records = extra if extra is not None else records
        preds[m] = pred
        metrics[m] = {"runtime_s": runtime, "forward_passes": bench.calls, "tasks": {}}
    bench.reset()
    for m in METHODS:
        for t in SPECIALIST:
            p, r, f = precision_recall_f1(preds[m][t], gt[t])
            metrics[m]["tasks"][t] = {
                "precision": p, "recall": r, "f1": f,
                "faithfulness": bench.keep(preds[m][t], t)[t],
                "minimality": 1.0 - len(preds[m][t]) / len(bench.nodes),
                "size": len(preds[m][t]),
            }
        metrics[m]["mean_f1"] = float(np.mean([metrics[m]["tasks"][t]["f1"] for t in SPECIALIST]))
        metrics[m]["hub_recovery"] = hub_recovery(preds[m], gt, layout["hubs"])
        metrics[m]["cross_task_leakage"] = cross_task_leakage(preds[m], gt)
    out_dir = ROOT / "results" / f"seed_{seed}"
    (out_dir / "predictions.json").write_text(json.dumps({m: {t: sorted(p) for t, p in preds[m].items()} for m in METHODS}, indent=2))
    (out_dir / "knn_windows.json").write_text(json.dumps(records, indent=2))
    metrics["ground_truth_test_faithfulness"] = {t: bench.keep(gt[t], t)[t] for t in SPECIALIST}
    return metrics


def stat(values):
    values = [v for v in values if v is not None]
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0}


def aggregate(per_seed):
    seeds = list(per_seed.values())
    summary = {}
    for m in METHODS:
        summary[m] = {"tasks": {t: {k: stat([s[m]["tasks"][t][k] for s in seeds]) for k in TASK_METRICS} for t in SPECIALIST}}
        summary[m].update({k: stat([s[m][k] for s in seeds]) for k in METHOD_METRICS})
    point = {m: (summary[m]["mean_f1"]["mean"], summary[m]["runtime_s"]["mean"]) for m in METHODS}
    pareto = [m for m in METHODS if not any(o != m and point[o][0] >= point[m][0] and point[o][1] <= point[m][1] and point[o] != point[m] for o in METHODS)]
    return {"summary": summary, "pareto_front_f1_vs_runtime": pareto}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=CFG["seeds"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--max_steps", type=int, default=TRAIN_CONFIG["max_steps"])
    p.add_argument("--batch_size", type=int, default=TRAIN_CONFIG["batch_size"])
    p.add_argument("--reuse", action="store_true")
    p.add_argument("--output", default=str(ROOT / "results" / "benchmark_table.json"))
    a = p.parse_args()
    per_seed = {}
    for seed in a.seeds:
        model, layout = train_and_gate(seed, a.device, a.max_steps, a.batch_size, a.reuse)
        per_seed[str(seed)] = evaluate_seed(model, layout, a.device, seed)
        print(f"seed {seed}: " + " ".join(f"{m} F1={per_seed[str(seed)][m]['mean_f1']:.3f} t={per_seed[str(seed)][m]['runtime_s']:.1f}s" for m in METHODS), flush=True)
    table = {"seeds": a.seeds, "device": a.device, "max_steps": a.max_steps, "config": CFG, **aggregate(per_seed), "per_seed": per_seed}
    Path(a.output).write_text(json.dumps(table, indent=2))
    print(f"wrote {a.output}")


if __name__ == "__main__":
    main()
