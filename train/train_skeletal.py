import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmarks.layout import generate_layout
from data.build_datasets import SPECIALIST, TASKS, load_split, load_vocab
from model.skeletal_transformer import CONFIG, build_model

ROOT = Path(__file__).resolve().parent.parent
TRAIN_CONFIG = json.loads((ROOT / "configs" / "train_config.json").read_text())
BENCH_CONFIG = json.loads((ROOT / "configs" / "benchmark_config.json").read_text())


def make_layout(seed, model):
    cfg = json.loads(CONFIG.read_text())
    return generate_layout(seed, model.node_names, model.node_layers, cfg["n_layers"], cfg["n_heads"], cfg["mlp_modules_per_layer"], BENCH_CONFIG["hub_fraction"])


def accuracy(model, data, device):
    out = {}
    with torch.no_grad():
        for t in SPECIALIST:
            d = data[t]
            T = int(d["target_pos"].max()) + 1
            logits = model(d["clean_ids"][:, :T].to(device))
            pred = logits[torch.arange(len(logits)), d["target_pos"].to(device)].argmax(-1).cpu()
            out[t] = (pred == d["clean_target_id"]).float().mean().item()
    return out


def train(seed, out_dir, device, max_steps, batch_size, lambda_off, iit_weight):
    cfg = TRAIN_CONFIG
    torch.manual_seed(seed)
    model = build_model(len(load_vocab())).to(device)
    layout = make_layout(seed, model)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ground_truth_circuits.json").write_text(json.dumps(layout, indent=2))
    train_data, val_data = load_split("train"), load_split("val")
    off = {t: [n for n in model.node_names if n not in layout["circuits"][t]] for t in SPECIALIST}
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], betas=tuple(cfg["betas"]))
    warmup = min(cfg["warmup_steps"], max_steps // 2)
    schedule = lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(max_steps - warmup, 1)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
    masks = {t: {n: 0.0 for n in off[t]} for t in SPECIALIST}
    warm_steps = int(cfg["mask_warmup_fraction"] * max_steps)
    gen = torch.Generator().manual_seed(seed)
    log = []
    model.train()
    for step in range(1, max_steps + 1):
        total, parts = 0.0, {}
        for t in TASKS:
            d = train_data[t]
            idx = torch.randint(len(d["target_pos"]), (batch_size,), generator=gen)
            pos = d["target_pos"][idx].to(device)
            T = int(pos.max()) + 1
            ids = d["clean_ids"][idx, :T].to(device)
            cache = {}
            warm = step <= warm_steps and t in SPECIALIST
            logits = model(ids, active_mask=masks[t] if warm else None, patch_cache=cache, patch_mode="record")
            if t == "TASK_GENERAL":
                loss = cfg["general_loss_weight"] * F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1), ignore_index=0)
            else:
                ce = F.cross_entropy(logits[torch.arange(batch_size, device=device), pos], d["clean_target_id"][idx].to(device))
                if warm:
                    parts[t] = ce.item()
                    total = total + ce
                    continue
                valid = (torch.arange(T, device=device)[None, :] <= pos[:, None]).float()
                penalty = torch.stack([(cache[n].pow(2).sum(-1) * valid).sum() / valid.sum() for n in off[t]]).mean()
                loss = ce + lambda_off * penalty
                parts[f"{t}_penalty"] = penalty.item()
                if iit_weight:
                    source = {}
                    model(d["corrupt_ids"][idx, :T].to(device), patch_cache=source, patch_mode="record")
                    rows = torch.arange(batch_size, device=device)
                    sufficient = model(ids, patch_cache=source, patch_mode="patch", patch_nodes=set(off[t]))[rows, pos]
                    necessary = model(ids, patch_cache=source, patch_mode="patch", patch_nodes=set(layout["circuits"][t]))[rows, pos]
                    iit = F.cross_entropy(sufficient, d["clean_target_id"][idx].to(device)) + F.cross_entropy(necessary, d["corrupt_target_id"][idx].to(device))
                    loss = loss + iit_weight * iit
                    parts[f"{t}_iit"] = iit.item()
            parts[t] = loss.item()
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
            log.append({"step": step, "loss": total.item(), **parts, "val_acc": acc})
            print(f"seed {seed} step {step} loss {total.item():.4f} " + " ".join(f"{t[5:]}={a:.3f}" for t, a in acc.items()), flush=True)
        if step % cfg["checkpoint_every"] == 0 or step == max_steps:
            torch.save({"model": model.state_dict(), "vocab_size": len(load_vocab()), "step": step, "lambda_off": lambda_off, "iit_weight": iit_weight, "layout": layout, "log": log}, out_dir / "skeletal_model.pt")
    (out_dir / "train_log.json").write_text(json.dumps({"lambda_off": lambda_off, "iit_weight": iit_weight, "max_steps": max_steps, "batch_size": batch_size, "log": log}, indent=2))
    model.eval()
    return model, layout


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["vocab_size"]).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval(), ckpt["layout"], ckpt["lambda_off"]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=TRAIN_CONFIG["max_steps"])
    p.add_argument("--batch_size", type=int, default=TRAIN_CONFIG["batch_size"])
    p.add_argument("--lambda_off", type=float, default=TRAIN_CONFIG["lambda_off"])
    p.add_argument("--iit_weight", type=float, default=TRAIN_CONFIG["iit_weight"])
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    train(a.seed, Path(a.out) if a.out else ROOT / "results" / f"seed_{a.seed}", a.device, a.max_steps, a.batch_size, a.lambda_off, a.iit_weight)
