import json
from typing import Any

import pytest

from data.build_datasets import OUT, SPECIALIST, build, load_split, load_vocab


@pytest.fixture(scope="module", autouse=True)
def built() -> None:
    if not (OUT / "task_3.pt").exists():
        build()


def test_targets_differ_and_lengths_match() -> None:
    for split in ["train", "val", "test"]:
        data = load_split(split)
        for t in SPECIALIST:
            d = data[t]
            assert bool((d["clean_target_id"] != d["corrupt_target_id"]).all())
            assert bool(((d["clean_ids"] != 0).sum(1) == (d["corrupt_ids"] != 0).sum(1)).all())
            assert bool(((d["clean_ids"] != 0).sum(1) == d["target_pos"] + 1).all())


def test_arithmetic_is_correct() -> None:
    vocab = load_vocab()
    d = load_split("test")["TASK_ARITH"]
    rows: list[list[int]] = d["clean_ids"].tolist()
    positions: list[int] = d["target_pos"].tolist()
    targets: list[int] = d["clean_target_id"].tolist()
    for row, pos, target in zip(rows, positions, targets, strict=True):
        a, _, b, _ = [vocab[i] for i in row[: pos + 1]]
        assert int(a) + int(b) == int(vocab[target])


def test_rhyme_pairs_are_held_out() -> None:
    pairs: dict[str, list[Any]] = json.loads((OUT / "rhyme_pairs.json").read_text())
    train = {tuple(p) for p in pairs["train"]}
    assert len(train) == 140 and len(pairs["val"]) == 30 and len(pairs["test"]) == 30
    assert not train & {tuple(p) for p in pairs["val"] + pairs["test"]}


def test_rhyme_target_is_the_last_rime() -> None:
    vocab = load_vocab()
    d = load_split("test")["TASK_RHYME"]
    rows: list[list[int]] = d["clean_ids"][:50].tolist()
    positions: list[int] = d["target_pos"][:50].tolist()
    targets: list[int] = d["clean_target_id"][:50].tolist()
    for row, pos, target in zip(rows, positions, targets, strict=True):
        words = [vocab[i] for i in row[: pos + 1]]
        assert words[-3] == vocab[target] and words[3] != vocab[target]
