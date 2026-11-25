import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from benchmarks.evaluate_identifiability import METHODS
from data.build_datasets import SPECIALIST

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
MODEL = json.loads((ROOT / "configs" / "model_config.json").read_text())
COLORS = {"Group-KNN": "#2a78d6", "Greedy-ACDC": "#eb6834", "Gradient-Attribution": "#1baf7a"}
CLUSTER_COLORS = {"TASK_SYNTAX": "#2a78d6", "TASK_ARITH": "#eb6834", "TASK_RHYME": "#1baf7a", "NULL": "#b8b6ad"}
INK, SURFACE, EMPTY = "#52514e", "#fcfcfb", "#ecebe6"
plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.edgecolor": "#b8b6ad", "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})


def label(t):
    return t[5:].title()


def f1_comparison(table):
    s = table["summary"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    x, w = np.arange(len(SPECIALIST)), 0.26
    for ax, metric in zip(axes, ["precision", "recall", "f1"]):
        for i, m in enumerate(METHODS):
            mean = [s[m]["tasks"][t][metric]["mean"] for t in SPECIALIST]
            std = [s[m]["tasks"][t][metric]["std"] for t in SPECIALIST]
            ax.bar(x + (i - 1) * w, mean, w - 0.02, yerr=std, color=COLORS[m], label=m, capsize=3, error_kw={"ecolor": INK, "lw": 1})
        ax.set_xticks(x, [label(t) for t in SPECIALIST])
        ax.set_title("F1" if metric == "f1" else metric.title())
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", color=EMPTY, lw=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle(f"Circuit recovery against planted ground truth (mean ± std over {len(table['seeds'])} seeds)")
    fig.tight_layout()
    fig.savefig(RESULTS / "f1_comparison.png", dpi=150)
    plt.close(fig)


def grid(nodes, value):
    g = np.zeros((MODEL["n_layers"], MODEL["n_heads"] + MODEL["mlp_modules_per_layer"]))
    for n in nodes:
        parts = n.split(".")
        col = int(parts[-1]) if parts[2] == "attn" else MODEL["n_heads"] + int(parts[-1])
        g[int(parts[1]), col] = value
    return g


def circuit_heatmap(seed):
    layout = json.loads((RESULTS / f"seed_{seed}" / "ground_truth_circuits.json").read_text())
    preds = json.loads((RESULTS / f"seed_{seed}" / "predictions.json").read_text())
    cmap = ListedColormap([EMPTY, "#1baf7a", "#eb6834", "#6b6a64"])
    cols = ["Ground truth"] + METHODS
    fig, axes = plt.subplots(len(SPECIALIST), len(cols), figsize=(17, 8))
    ticks = [f"H{h}" for h in range(MODEL["n_heads"])] + [f"M{m}" for m in range(MODEL["mlp_modules_per_layer"])]
    for r, t in enumerate(SPECIALIST):
        gt = set(layout["circuits"][t])
        for c, name in enumerate(cols):
            if name == "Ground truth":
                g = grid(gt, 1)
            else:
                p = set(preds[name][t])
                g = grid(gt & p, 1) + grid(p - gt, 2) + grid(gt - p, 3)
            ax = axes[r, c]
            ax.imshow(g, cmap=cmap, vmin=0, vmax=3)
            ax.set_xticks(range(len(ticks)), ticks, fontsize=7)
            ax.set_yticks(range(MODEL["n_layers"]), [f"L{L}" for L in range(MODEL["n_layers"])], fontsize=8)
            ax.set_xticks(np.arange(-0.5, len(ticks)), minor=True)
            ax.set_yticks(np.arange(-0.5, MODEL["n_layers"]), minor=True)
            ax.grid(which="minor", color=SURFACE, lw=2)
            ax.tick_params(which="minor", length=0)
            ax.axvline(MODEL["n_heads"] - 0.5, color=INK, lw=1)
            if r == 0:
                ax.set_title(name)
            if c == 0:
                ax.set_ylabel(label(t), fontsize=11)
    fig.legend(handles=[Patch(color="#1baf7a", label="Ground truth / recovered (TP)"), Patch(color="#eb6834", label="Recovered, not in ground truth (FP)"), Patch(color="#6b6a64", label="Missed (FN)")], loc="lower center", ncol=3, frameon=False)
    fig.suptitle(f"Ground-truth vs recovered circuits, seed {seed} (rows = layers, columns = heads | MLP modules)")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(RESULTS / "circuit_heatmap.png", dpi=150)
    plt.close(fig)


def knn_clusters(seed):
    records = json.loads((RESULTS / f"seed_{seed}" / "knn_windows.json").read_text())
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(projection="3d")
    for name, color in CLUSTER_COLORS.items():
        pts = [r for r in records if r["label"] == name]
        if pts:
            s = np.array([r["scores"] for r in pts])
            ax.scatter(s[:, 0], s[:, 1], s[:, 2], color=color, s=[12 + 6 * r["W"] for r in pts], label=f"{label(name) if name != 'NULL' else 'Null/Noise'} ({len(pts)})", edgecolors=SURFACE, linewidths=0.5)
    ax.set_xlabel("Syntax effect")
    ax.set_ylabel("Arith effect")
    ax.set_zlabel("Rhyme effect")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title(f"Group-KNN window ablation effects by cluster, seed {seed} (marker size = window width)")
    fig.tight_layout()
    fig.savefig(RESULTS / "knn_task_clusters.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    table = json.loads((RESULTS / "benchmark_table.json").read_text())
    f1_comparison(table)
    circuit_heatmap(table["seeds"][0])
    knn_clusters(table["seeds"][0])
    print("wrote results/f1_comparison.png, results/circuit_heatmap.png, results/knn_task_clusters.png")
