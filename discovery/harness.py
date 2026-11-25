import torch

from discovery.metrics import faithfulness_score, logit_diff
from discovery.protocol import HookableModelProtocol


class CircuitBench:
    def __init__(self, model, task_data, device):
        assert isinstance(model, HookableModelProtocol)
        self.model = model
        self.nodes = list(model.node_names)
        self.node_layers = dict(model.node_layers)
        self.tasks = list(task_data)
        T = max(int(d["target_pos"].max()) + 1 for d in task_data.values())
        cat = lambda k: torch.cat([d[k] for d in task_data.values()]).to(device)
        self.clean = cat("clean_ids")[:, :T]
        self.corrupt = cat("corrupt_ids")[:, :T]
        self.pos, self.cid, self.xid = cat("target_pos"), cat("clean_target_id"), cat("corrupt_target_id")
        self.slices, start = {}, 0
        for t, d in task_data.items():
            self.slices[t] = slice(start, start + len(d["target_pos"]))
            start += len(d["target_pos"])
        self.full = slice(0, start)
        self.clean_cache, self.corrupt_cache = {}, {}
        with torch.no_grad():
            clean_logits = model(self.clean, patch_cache=self.clean_cache, patch_mode="record")
            corrupt_logits = model(self.corrupt, patch_cache=self.corrupt_cache, patch_mode="record")
        self.clean_ld = self._lds(clean_logits, None)
        self.corrupt_ld = self._lds(corrupt_logits, None)
        self.reset()

    def reset(self):
        self.calls = 0
        self._memo = {}

    def _lds(self, logits, task):
        offset = self.slices[task].start if task else 0
        out = {}
        for t in [task] if task else self.tasks:
            s = self.slices[t]
            local = slice(s.start - offset, s.stop - offset)
            out[t] = logit_diff(logits[local], self.pos[s], self.cid[s], self.xid[s]).mean().item()
        return out

    def faithfulness(self, patch_nodes, task=None):
        key = (frozenset(patch_nodes), task)
        if key not in self._memo:
            sl = self.slices[task] if task else self.full
            cache = {n: self.corrupt_cache[n][sl] for n in patch_nodes}
            with torch.no_grad():
                logits = self.model(self.clean[sl], patch_cache=cache, patch_mode="patch", patch_nodes=set(patch_nodes))
            self.calls += 1
            lds = self._lds(logits, task)
            self._memo[key] = {t: faithfulness_score(v, self.clean_ld[t], self.corrupt_ld[t]) for t, v in lds.items()}
        return self._memo[key]

    def keep(self, circuit, task=None):
        return self.faithfulness(set(self.nodes) - set(circuit), task)

    def ablate(self, nodes, task=None):
        return self.faithfulness(set(nodes), task)
