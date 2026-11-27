from collections.abc import Iterable

import torch
from torch import Tensor

from discovery.metrics import faithfulness_score, logit_diff
from discovery.protocol import HookableModelProtocol


class CircuitBench:
    def __init__(self, model: HookableModelProtocol, task_data: dict[str, dict[str, Tensor]], device: str) -> None:
        self.model = model
        self.nodes = list(model.node_names)
        self.node_layers = dict(model.node_layers)
        self.tasks = list(task_data)
        T = max(int(d["target_pos"].max()) + 1 for d in task_data.values())

        def cat(key: str) -> Tensor:
            return torch.cat([d[key] for d in task_data.values()]).to(device)

        self.clean = cat("clean_ids")[:, :T]
        self.corrupt = cat("corrupt_ids")[:, :T]
        self.pos = cat("target_pos")
        self.cid = cat("clean_target_id")
        self.xid = cat("corrupt_target_id")
        self.ranges: dict[str, tuple[int, int]] = {}
        start = 0
        for t, d in task_data.items():
            self.ranges[t] = (start, start + len(d["target_pos"]))
            start += len(d["target_pos"])
        self.total = start
        self.clean_cache: dict[str, Tensor] = {}
        self.corrupt_cache: dict[str, Tensor] = {}
        with torch.no_grad():
            clean_logits = model(self.clean, patch_cache=self.clean_cache, patch_mode="record")
            corrupt_logits = model(self.corrupt, patch_cache=self.corrupt_cache, patch_mode="record")
        self.clean_ld = self._lds(clean_logits, None)
        self.corrupt_ld = self._lds(corrupt_logits, None)
        self.calls = 0
        self._memo: dict[tuple[frozenset[str], str | None], dict[str, float]] = {}

    def reset(self) -> None:
        self.calls = 0
        self._memo = {}

    def rows(self, task: str | None) -> tuple[int, int]:
        return self.ranges[task] if task else (0, self.total)

    def _lds(self, logits: Tensor, task: str | None) -> dict[str, float]:
        offset = self.rows(task)[0]
        out: dict[str, float] = {}
        for t in [task] if task else self.tasks:
            a, b = self.ranges[t]
            ld = logit_diff(logits[a - offset : b - offset], self.pos[a:b], self.cid[a:b], self.xid[a:b])
            out[t] = float(ld.mean().item())
        return out

    def faithfulness(self, patch_nodes: Iterable[str], task: str | None = None) -> dict[str, float]:
        nodes = frozenset(patch_nodes)
        key = (nodes, task)
        if key not in self._memo:
            a, b = self.rows(task)
            cache = {n: self.corrupt_cache[n][a:b] for n in nodes}
            with torch.no_grad():
                logits = self.model(self.clean[a:b], patch_cache=cache, patch_mode="patch", patch_nodes=set(nodes))
            self.calls += 1
            lds = self._lds(logits, task)
            self._memo[key] = {t: faithfulness_score(v, self.clean_ld[t], self.corrupt_ld[t]) for t, v in lds.items()}
        return self._memo[key]

    def keep(self, circuit: Iterable[str], task: str | None = None) -> dict[str, float]:
        return self.faithfulness(set(self.nodes) - set(circuit), task)

    def ablate(self, nodes: Iterable[str], task: str | None = None) -> dict[str, float]:
        return self.faithfulness(nodes, task)
