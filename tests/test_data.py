import json

import pytest

from data.build_datasets import OUT, SPECIALIST, build, load_split, load_vocab


@pytest.fixture(scope="module", autouse=True)
def built():
    if not (OUT / "task_3.pt").exists():
        build()


def test_targets_differ_and_lengths_match():
    for split in ["train", "val", "test"]:
        data = load_split(split)
        for t in SPECIALIST:
            d = data[t]
            assert bool((d["clean_target_id"] != d["corrupt_target_id"]).all())
            assert bool(((d["clean_ids"] != 0).sum(1) == (d["corrupt_ids"] != 0).sum(1)).all())
            assert bool(((d["clean_ids"] != 0).sum(1) == d["target_pos"] + 1).all())


def test_arithmetic_is_correct():
    vocab = load_vocab()
    d = load_split("test")["TASK_ARITH"]
    for row, pos, target in zip(d["clean_ids"], d["target_pos"], d["clean_target_id"]):
        a, _, b, _ = [vocab[i] for i in row[: pos + 1]]
        assert int(a) + int(b) == int(vocab[target])


def test_rhyme_pairs_are_held_out():
    pairs = json.loads((OUT / "rhyme_pairs.json").read_text())
    train = {tuple(p) for p in pairs["train"]}
    assert len(train) == 140 and len(pairs["val"]) == 30 and len(pairs["test"]) == 30
    assert not train & {tuple(p) for p in pairs["val"] + pairs["test"]}


def test_rhyme_target_is_the_last_rime():
    vocab = load_vocab()
    d = load_split("test")["TASK_RHYME"]
    for row, pos, target in zip(d["clean_ids"][:50], d["target_pos"][:50], d["clean_target_id"][:50]):
        words = [vocab[i] for i in row[: pos + 1]]
        assert words[-3] == vocab[target] and words[3] != vocab[target]
