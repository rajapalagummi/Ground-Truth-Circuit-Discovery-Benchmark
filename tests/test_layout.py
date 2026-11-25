import random

from benchmarks.layout import generate_layout, pick_noncontiguous
from data.build_datasets import SPECIALIST
from model.skeletal_transformer import build_model

MODEL = build_model(241)


def layouts():
    return [generate_layout(s, MODEL.node_names, MODEL.node_layers, 6, 8, 4, 0.12) for s in range(3)]


def test_layouts_differ_per_seed():
    a, b, c = [tuple(tuple(l["circuits"][t]) for t in SPECIALIST) for l in layouts()]
    assert len({a, b, c}) == 3


def test_spans_are_staggered():
    for l in layouts():
        starts = sorted(l["spans"][t][0] for t in SPECIALIST)
        assert starts == [0, 1, 2]


def test_hubs_shared_by_exactly_two_tasks():
    for l in layouts():
        total = len(set().union(*[set(l["circuits"][t]) for t in SPECIALIST]))
        assert 0.08 <= len(l["hubs"]) / total <= 0.16
        for h in l["hubs"]:
            assert sum(h in l["circuits"][t] for t in SPECIALIST) == 2


def test_pick_is_non_contiguous():
    rng = random.Random(0)
    for _ in range(200):
        heads = sorted(pick_noncontiguous(list(range(8)), rng.choice([2, 3]), rng))
        assert heads[-1] - heads[0] != len(heads) - 1


def test_circuits_are_not_vertical_columns():
    for l in layouts():
        for t in SPECIALIST:
            per_layer = {}
            for n in l["circuits"][t]:
                if ".attn." in n:
                    per_layer.setdefault(MODEL.node_layers[n], set()).add(int(n.split(".")[-1]))
            assert len({tuple(sorted(h)) for h in per_layer.values()}) > 1
