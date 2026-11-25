import json
from pathlib import Path
from typing import Dict, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "model_config.json"


class Head(nn.Module):
    def __init__(self, d_model, d_head):
        super().__init__()
        self.q = nn.Linear(d_model, d_head)
        self.k = nn.Linear(d_model, d_head)
        self.v = nn.Linear(d_model, d_head)
        self.o = nn.Linear(d_head, d_model)

    def forward(self, x):
        return self.o(F.scaled_dot_product_attention(self.q(x), self.k(x), self.v(x), is_causal=True))


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, d_head):
        super().__init__()
        self.heads = nn.ModuleList(Head(d_model, d_head) for _ in range(n_heads))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_head, n_mlps, mlp_hidden):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads, d_head)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlps = nn.ModuleList(
            nn.Sequential(nn.Linear(d_model, mlp_hidden), nn.GELU(), nn.Linear(mlp_hidden, d_model)) for _ in range(n_mlps)
        )


class SkeletalTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_head, mlp_modules_per_layer, mlp_hidden, max_seq_len, dropout):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList(
            Block(d_model, n_heads, d_head, mlp_modules_per_layer, mlp_hidden) for _ in range(n_layers)
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)
        self.drop = nn.Dropout(dropout)
        self.node_names = []
        self.node_layers = {}
        for L in range(n_layers):
            for name in [f"blocks.{L}.attn.heads.{H}" for H in range(n_heads)] + [f"blocks.{L}.mlps.{M}" for M in range(mlp_modules_per_layer)]:
                self.node_names.append(name)
                self.node_layers[name] = L
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def embed(self, ids: Tensor) -> Tensor:
        return self.tok_embed(ids) + self.pos_embed(torch.arange(ids.shape[1], device=ids.device))

    def _node(self, name, x, module, active_mask, patch_cache, patch_mode, patch_nodes):
        if patch_mode == "patch" and name in patch_nodes:
            out = patch_cache[name]
        else:
            out = module(x)
        if patch_mode == "record":
            patch_cache[name] = out
        scale = 1.0 if active_mask is None else active_mask.get(name, 1.0)
        return scale * self.drop(out)

    def forward(
        self,
        ids: Tensor,
        active_mask: Optional[Dict[str, float]] = None,
        patch_cache: Optional[Dict[str, Tensor]] = None,
        patch_mode: str = "none",
        patch_nodes: Optional[Set[str]] = None,
    ) -> Tensor:
        args = (active_mask, patch_cache, patch_mode, patch_nodes or set())
        x = self.drop(self.embed(ids))
        for L, block in enumerate(self.blocks):
            h = block.ln1(x)
            x = x + sum(self._node(f"blocks.{L}.attn.heads.{H}", h, head, *args) for H, head in enumerate(block.attn.heads))
            h = block.ln2(x)
            x = x + sum(self._node(f"blocks.{L}.mlps.{M}", h, mlp, *args) for M, mlp in enumerate(block.mlps))
        return self.unembed(self.ln_f(x))


def build_model(vocab_size):
    return SkeletalTransformer(vocab_size=vocab_size, **json.loads(CONFIG.read_text()))
