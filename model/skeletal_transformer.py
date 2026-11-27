import json
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "model_config.json"


class Head(nn.Module):
    def __init__(self, d_model: int, d_head: int) -> None:
        super().__init__()
        self.q = nn.Linear(d_model, d_head)
        self.k = nn.Linear(d_model, d_head)
        self.v = nn.Linear(d_model, d_head)
        self.o = nn.Linear(d_head, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.o(F.scaled_dot_product_attention(self.q(x), self.k(x), self.v(x), is_causal=True))


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int) -> None:
        super().__init__()
        self.heads = nn.ModuleList(Head(d_model, d_head) for _ in range(n_heads))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_head: int, n_mlps: int, mlp_hidden: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads, d_head)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlps = nn.ModuleList(
            nn.Sequential(nn.Linear(d_model, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, d_model)) for _ in range(n_mlps)
        )


class SkeletalTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_head: int,
        mlp_modules_per_layer: int,
        mlp_hidden: int,
        max_seq_len: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_heads, d_head, mlp_modules_per_layer, mlp_hidden) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)
        self.drop = nn.Dropout(dropout)
        self.node_names: list[str] = []
        self.node_layers: dict[str, int] = {}
        for L in range(n_layers):
            heads = [f"blocks.{L}.attn.heads.{H}" for H in range(n_heads)]
            mlps = [f"blocks.{L}.mlps.{M}" for M in range(mlp_modules_per_layer)]
            for name in heads + mlps:
                self.node_names.append(name)
                self.node_layers[name] = L
        for name, p in self.named_parameters():
            if p.dim() == 2:
                nn.init.normal_(p, std=0.02)
            elif ".ln" not in name and not name.startswith("ln"):
                nn.init.zeros_(p)

    def embed(self, ids: Tensor) -> Tensor:
        return self.tok_embed(ids) + self.pos_embed(torch.arange(ids.shape[1], device=ids.device))

    def _node(
        self,
        name: str,
        x: Tensor,
        module: nn.Module,
        active_mask: dict[str, float] | None,
        cache: dict[str, Tensor],
        patch_mode: str,
        patch_nodes: set[str],
    ) -> Tensor:
        out: Tensor = cache[name] if patch_mode == "patch" and name in patch_nodes else module(x)
        if patch_mode == "record":
            cache[name] = out
        scale = 1.0 if active_mask is None else active_mask.get(name, 1.0)
        return scale * self.drop(out)

    def forward(
        self,
        ids: Tensor,
        active_mask: dict[str, float] | None = None,
        patch_cache: dict[str, Tensor] | None = None,
        patch_mode: str = "none",
        patch_nodes: set[str] | None = None,
    ) -> Tensor:
        cache: dict[str, Tensor] = patch_cache if patch_cache is not None else {}
        nodes: set[str] = patch_nodes if patch_nodes is not None else set()
        x: Tensor = self.drop(self.embed(ids))
        for L, module in enumerate(self.blocks):
            block = cast(Block, module)
            h: Tensor = block.ln1(x)
            for H, head in enumerate(block.attn.heads):
                x = x + self._node(f"blocks.{L}.attn.heads.{H}", h, head, active_mask, cache, patch_mode, nodes)
            h = block.ln2(x)
            for M, mlp in enumerate(block.mlps):
                x = x + self._node(f"blocks.{L}.mlps.{M}", h, mlp, active_mask, cache, patch_mode, nodes)
        return self.unembed(self.ln_f(x))


def build_model(vocab_size: int) -> SkeletalTransformer:
    return SkeletalTransformer(vocab_size=vocab_size, **json.loads(CONFIG.read_text()))
