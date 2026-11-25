import pytest
import torch

from data.build_datasets import OUT, SPECIALIST, build, load_split
from discovery.baseline_acdc import gradient_attribution, greedy_acdc
from discovery.group_knn_discovery import group_knn, windows
from discovery.harness import CircuitBench
from discovery.metrics import faithfulness_score, precision_recall_f1
from model.skeletal_transformer import build_model


@pytest.fixture(scope="module")
def bench():
    if not (OUT / "task_3.pt").exists():
        build()
    torch.manual_seed(0)
    model = build_model(241).eval()
    data = {t: {k: v[:16] for k, v in d.items()} for t, d in load_split("test").items() if t in SPECIALIST}
    return CircuitBench(model, data, "cpu")


def test_metrics():
    assert precision_recall_f1({"a", "b"}, {"b", "c"}) == (0.5, 0.5, 0.5)
    assert precision_recall_f1(set(), {"a"}) == (0.0, 0.0, 0.0)
    assert abs(faithfulness_score(1.0, 1.0, -1.0) - 1.0) < 1e-6
    assert abs(faithfulness_score(-1.0, 1.0, -1.0)) < 1e-6


def test_faithfulness_endpoints(bench):
    for t in SPECIALIST:
        assert abs(bench.keep(bench.nodes, t)[t] - 1.0) < 1e-4
        assert abs(bench.keep([], t)[t]) < 1e-4


def test_multi_task_pass_matches_single_task(bench):
    group = set(bench.nodes[10:20])
    together = bench.ablate(group)
    for t in SPECIALIST:
        assert abs(together[t] - bench.ablate(group, t)[t]) < 1e-4


def test_windows_cover_all_nodes():
    nodes = list(range(72))
    w = windows(nodes, [8, 4, 2, 1])
    assert len(w) == 17 + 35 + 71 + 72
    assert set().union(*map(set, w)) == set(nodes)


def test_methods_return_node_subsets(bench):
    pred, records = group_knn(bench, [8, 4, 2, 1], 4, 5, 0.05, 0)
    assert set(pred) == set(SPECIALIST) and len(records) == 195
    for t in SPECIALIST:
        assert greedy_acdc(bench, t, 0.03) <= set(bench.nodes)
        assert gradient_attribution(bench, t, 0.85) <= set(bench.nodes)
        assert pred[t] <= set(bench.nodes)
