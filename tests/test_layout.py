import random
from typing import Any

from benchmarks.layout import generate_layout, pick_noncontiguous
from data.build_datasets import SPECIALIST
from model.skeletal_transformer import build_model

MODEL = build_model(241)


def layouts() -> list[dict[str, Any]]:
    return [generate_layout(s, MODEL.node_names, MODEL.node_layers, 6, 8, 4, 0.12) for s in range(3)]


def test_layouts_differ_per_seed() -> None:
    keys = {tuple(tuple(lay["circuits"][t]) for t in SPECIALIST) for lay in layouts()}
    assert len(keys) == 3


def test_spans_are_staggered() -> None:
    for lay in layouts():
        assert sorted(lay["spans"][t][0] for t in SPECIALIST) == [0, 1, 2]


def test_hubs_shared_by_exactly_two_tasks() -> None:
    for lay in layouts():
        total = len(set[str]().union(*[set[str](lay["circuits"][t]) for t in SPECIALIST]))
        assert 0.08 <= len(lay["hubs"]) / total <= 0.16
        for h in lay["hubs"]:
            assert sum(h in lay["circuits"][t] for t in SPECIALIST) == 2


def test_pick_is_non_contiguous() -> None:
    rng = random.Random(0)
    for _ in range(200):
        heads = sorted(pick_noncontiguous(list(range(8)), rng.choice([2, 3]), rng))
        assert heads[-1] - heads[0] != len(heads) - 1


def test_circuits_are_not_vertical_columns() -> None:
    for lay in layouts():
        for t in SPECIALIST:
            per_layer: dict[int, set[int]] = {}
            for n in lay["circuits"][t]:
                if ".attn." in n:
                    per_layer.setdefault(MODEL.node_layers[n], set()).add(int(n.split(".")[-1]))
            assert len({tuple(sorted(h)) for h in per_layer.values()}) > 1
