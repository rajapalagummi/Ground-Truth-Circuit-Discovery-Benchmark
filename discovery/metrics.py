import torch


def logit_diff(logits, pos, clean_id, corrupt_id):
    idx = torch.arange(logits.shape[0], device=logits.device)
    last = logits[idx, pos]
    return last[idx, clean_id] - last[idx, corrupt_id]


def faithfulness_score(patched_ld, clean_ld, corrupt_ld):
    return (patched_ld - corrupt_ld) / (clean_ld - corrupt_ld + 1e-8)


def precision_recall_f1(pred, truth):
    tp = len(set(pred) & set(truth))
    p = tp / len(pred) if pred else 0.0
    r = tp / len(truth) if truth else 0.0
    return p, r, 2 * p * r / (p + r) if p + r else 0.0
