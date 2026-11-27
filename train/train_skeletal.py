import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from benchmarks.layout import generate_layout
from data.build_datasets import SPECIALIST, TASKS, TaskData, load_split, load_vocab
from model.skeletal_transformer import CONFIG, SkeletalTransformer, build_model

ROOT = Path(__file__).resolve().parent.parent
TRAIN_CONFIG: dict[str, Any] = json.loads((ROOT / "configs" / "train_config.json").read_text())
BENCH_CONFIG: dict[str, Any] = json.loads((ROOT / "configs" / "benchmark_config.json").read_text())


def make_layout(seed: int, model: SkeletalTransformer) -> dict[str, Any]:
    cfg: dict[str, Any] = json.loads(CONFIG.read_text())
    return generate_layout(
        seed,
        model.node_names,
        model.node_layers,
        cfg["n_layers"],
        cfg["n_heads"],
        cfg["mlp_modules_per_layer"],
        BENCH_CONFIG["hub_fraction"],
    )


def accuracy(model: SkeletalTransformer, data: dict[str, TaskData], device: str) -> dict[str, float]:
    out: dict[str, float] = {}
    with torch.no_grad():
        for t in SPECIALIST:
            d = data[t]
            T = int(d["target_pos"].max()) + 1
            logits: Tensor = model(d["clean_ids"][:, :T].to(device))
            pred = logits[torch.arange(len(logits)), d["target_pos"].to(device)].argmax(-1).cpu()
            out[t] = float((pred == d["clean_target_id"]).float().mean().item())
    return out


def train(
    seed: int, out_dir: Path, device: str, max_steps: int, batch_size: int, lambda_off: float, iit_weight: float
) -> tuple[SkeletalTransformer, dict[str, Any]]:
    cfg = TRAIN_CONFIG
    torch.manual_seed(seed)
    vocab_size = len(load_vocab())
    model = build_model(vocab_size).to(device)
    layout = make_layout(seed, model)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ground_truth_circuits.json").write_text(json.dumps(layout, indent=2))
    train_data, val_data = load_split("train"), load_split("val")
    circuits: dict[str, list[str]] = layout["circuits"]
    off = {t: [n for n in model.node_names if n not in circuits[t]] for t in SPECIALIST}
    masks = {t: dict.fromkeys(off[t], 0.0) for t in SPECIALIST}
    warm_steps = int(cfg["mask_warmup_fraction"] * max_steps)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], betas=(cfg["betas"][0], cfg["betas"][1]))
    warmup: int = min(cfg["warmup_steps"], max_steps // 2)

    def schedule(s: int) -> float:
        if s < warmup:
            return (s + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(max_steps - warmup, 1)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
    gen = torch.Generator().manual_seed(seed)
    rows = torch.arange(batch_size, device=device)
    log: list[dict[str, Any]] = []
    model.train()
    for step in range(1, max_steps + 1):
        total = torch.zeros((), device=device)
        parts: dict[str, float] = {}
        warm = step <= warm_steps
        for t in TASKS:
            d = train_data[t]
            idx = torch.randint(len(d["target_pos"]), (batch_size,), generator=gen)
            pos = d["target_pos"][idx].to(device)
            T = int(pos.max()) + 1
            ids = d["clean_ids"][idx, :T].to(device)
            clean_target = d["clean_target_id"][idx].to(device)
            if t == "TASK_GENERAL":
                logits: Tensor = model(ids)
                loss = cfg["general_loss_weight"] * F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), ignore_index=0
                )
            elif warm:
                logits = model(ids, active_mask=masks[t])
                loss = F.cross_entropy(logits[rows, pos], clean_target)
            else:
                cache: dict[str, Tensor] = {}
                logits = model(ids, patch_cache=cache, patch_mode="record")
                ce = F.cross_entropy(logits[rows, pos], clean_target)
                valid = (torch.arange(T, device=device)[None, :] <= pos[:, None]).float()
                penalty = torch.stack([(cache[n].pow(2).sum(-1) * valid).sum() / valid.sum() for n in off[t]]).mean()
                source: dict[str, Tensor] = {}
                model(d["corrupt_ids"][idx, :T].to(device), patch_cache=source, patch_mode="record")
                sufficient: Tensor = model(ids, patch_cache=source, patch_mode="patch", patch_nodes=set(off[t]))
                necessary: Tensor = model(ids, patch_cache=source, patch_mode="patch", patch_nodes=set(circuits[t]))
                iit = F.cross_entropy(sufficient[rows, pos], clean_target) + F.cross_entropy(
                    necessary[rows, pos], d["corrupt_target_id"][idx].to(device)
                )
                loss = ce + lambda_off * penalty + iit_weight * iit
                parts[f"{t}_penalty"] = float(penalty.item())
                parts[f"{t}_iit"] = float(iit.item())
            parts[t] = float(loss.item())
            total = total + loss
        opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % cfg["eval_every"] == 0 or step == max_steps or step == 1:
            model.eval()
            acc = accuracy(model, val_data, device)
            model.train()
            log.append({"step": step, "loss": float(total.item()), **parts, "val_acc": acc})
            print(f"seed {seed} step {step} loss {total.item():.4f} " + " ".join(f"{t[5:]}={a:.3f}" for t, a in acc.items()), flush=True)
        if step % cfg["checkpoint_every"] == 0 or step == max_steps:
            torch.save(
                {
                    "model": model.state_dict(),
                    "vocab_size": vocab_size,
                    "step": step,
                    "lambda_off": lambda_off,
                    "iit_weight": iit_weight,
                    "layout": layout,
                    "log": log,
                },
                out_dir / "skeletal_model.pt",
            )
    summary = {"lambda_off": lambda_off, "iit_weight": iit_weight, "max_steps": max_steps, "batch_size": batch_size, "log": log}
    (out_dir / "train_log.json").write_text(json.dumps(summary, indent=2))
    model.eval()
    return model, layout


def load_checkpoint(path: Path, device: str) -> tuple[SkeletalTransformer, dict[str, Any], float]:
    ckpt: dict[str, Any] = torch.load(path, map_location=device)
    model = build_model(ckpt["vocab_size"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt["layout"], ckpt["lambda_off"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=TRAIN_CONFIG["max_steps"])
    p.add_argument("--batch_size", type=int, default=TRAIN_CONFIG["batch_size"])
    p.add_argument("--lambda_off", type=float, default=TRAIN_CONFIG["lambda_off"])
    p.add_argument("--iit_weight", type=float, default=TRAIN_CONFIG["iit_weight"])
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    out = Path(a.out) if a.out else ROOT / "results" / f"seed_{a.seed}"
    train(a.seed, out, a.device, a.max_steps, a.batch_size, a.lambda_off, a.iit_weight)


if __name__ == "__main__":
    main()
