import torch
from torch import Tensor

from discovery.protocol import HookableModelProtocol
from model.skeletal_transformer import SkeletalTransformer, build_model


def make() -> tuple[SkeletalTransformer, Tensor]:
    torch.manual_seed(0)
    model = build_model(241)
    model.eval()
    return model, torch.randint(1, 241, (4, 12))


def test_parameter_count_and_protocol() -> None:
    model, _ = make()
    n = sum(p.numel() for p in model.parameters())
    assert 4.5e6 < n < 5.5e6
    hookable: HookableModelProtocol = model
    assert len(hookable.node_names) == 72


def test_zero_mask_leaves_embeddings() -> None:
    model, ids = make()
    with torch.no_grad():
        logits: Tensor = model(ids, active_mask=dict.fromkeys(model.node_names, 0.0))
        expected: Tensor = model.unembed(model.ln_f(model.embed(ids)))
    assert torch.allclose(logits, expected, atol=1e-6)


def test_record_then_patch_reproduces_logits() -> None:
    model, ids = make()
    cache: dict[str, Tensor] = {}
    with torch.no_grad():
        clean: Tensor = model(ids, patch_cache=cache, patch_mode="record")
        patched: Tensor = model(ids, patch_cache=cache, patch_mode="patch", patch_nodes=set(model.node_names))
    assert float((clean - patched).abs().max().item()) < 1e-5


def test_patch_changes_output() -> None:
    model, ids = make()
    other = torch.randint(1, 241, ids.shape)
    cache: dict[str, Tensor] = {}
    with torch.no_grad():
        model(other, patch_cache=cache, patch_mode="record")
        clean: Tensor = model(ids)
        patched: Tensor = model(ids, patch_cache=cache, patch_mode="patch", patch_nodes={"blocks.0.attn.heads.0"})
    assert float((clean - patched).abs().max().item()) > 1e-6
