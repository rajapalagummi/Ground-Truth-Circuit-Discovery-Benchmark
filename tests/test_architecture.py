import torch

from discovery.protocol import HookableModelProtocol
from model.skeletal_transformer import build_model


def make():
    torch.manual_seed(0)
    return build_model(241).eval(), torch.randint(1, 241, (4, 12))


def test_parameter_count_and_nodes():
    model, _ = make()
    n = sum(p.numel() for p in model.parameters())
    assert 4.5e6 < n < 5.5e6
    assert len(model.node_names) == 72
    assert isinstance(model, HookableModelProtocol)


def test_zero_mask_leaves_embeddings():
    model, ids = make()
    with torch.no_grad():
        logits = model(ids, active_mask={n: 0.0 for n in model.node_names})
        expected = model.unembed(model.ln_f(model.embed(ids)))
    assert torch.allclose(logits, expected, atol=1e-6)


def test_record_then_patch_reproduces_logits():
    model, ids = make()
    cache = {}
    with torch.no_grad():
        clean = model(ids, patch_cache=cache, patch_mode="record")
        patched = model(ids, patch_cache=cache, patch_mode="patch", patch_nodes=set(model.node_names))
    assert (clean - patched).abs().max().item() < 1e-5


def test_patch_changes_output():
    model, ids = make()
    other = torch.randint(1, 241, ids.shape)
    cache = {}
    with torch.no_grad():
        model(other, patch_cache=cache, patch_mode="record")
        clean = model(ids)
        patched = model(ids, patch_cache=cache, patch_mode="patch", patch_nodes={"blocks.0.attn.heads.0"})
    assert (clean - patched).abs().max().item() > 1e-6
