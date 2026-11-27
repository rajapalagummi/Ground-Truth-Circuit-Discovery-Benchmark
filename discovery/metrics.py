import torch
from torch import Tensor


def logit_diff(logits: Tensor, pos: Tensor, clean_id: Tensor, corrupt_id: Tensor) -> Tensor:
    idx = torch.arange(logits.shape[0], device=logits.device)
    last = logits[idx, pos]
    return last[idx, clean_id] - last[idx, corrupt_id]


def faithfulness_score(patched_ld: float, clean_ld: float, corrupt_ld: float) -> float:
    return (patched_ld - corrupt_ld) / (clean_ld - corrupt_ld + 1e-8)


def precision_recall_f1(pred: set[str], truth: set[str]) -> tuple[float, float, float]:
    tp = len(pred & truth)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(truth) if truth else 0.0
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def harmonic(a: float, b: float) -> float:
    return 2 * a * b / (a + b) if a + b else 0.0
