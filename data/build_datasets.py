import json
import random
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "processed"
TASKS = ["TASK_SYNTAX", "TASK_ARITH", "TASK_RHYME", "TASK_GENERAL"]
SPECIALIST = TASKS[:3]
SPLITS = {"train": 10000, "val": 1000, "test": 500}
MAX_LEN = 64
PAD = "<pad>"

NOUNS = [
    ("key", "keys"), ("book", "books"), ("dog", "dogs"), ("car", "cars"), ("door", "doors"),
    ("tree", "trees"), ("cup", "cups"), ("chair", "chairs"), ("lamp", "lamps"), ("box", "boxes"),
    ("bird", "birds"), ("boat", "boats"), ("coat", "coats"), ("desk", "desks"), ("flower", "flowers"),
    ("girl", "girls"), ("boy", "boys"), ("horse", "horses"), ("house", "houses"), ("letter", "letters"),
    ("map", "maps"), ("pen", "pens"), ("river", "rivers"), ("road", "roads"), ("shoe", "shoes"),
    ("star", "stars"), ("table", "tables"), ("window", "windows"), ("wall", "walls"), ("teacher", "teachers"),
]
PREPS = ["to", "near", "behind", "beside", "above", "under", "with", "for"]
ADJS = ["old", "new", "red", "small", "large", "quiet", "bright", "dark", "green", "heavy"]
VERBS = ["saw", "found", "moved", "painted", "opened", "carried", "cleaned", "watched", "pushed", "lifted"]
CONNECTORS = ["and", "while", "because"]
RHYME = {
    "ight": ["l", "n", "s", "f", "m", "t", "r", "br"],
    "ake": ["l", "m", "t", "b", "c", "f", "r", "sh"],
    "ound": ["s", "r", "f", "h", "m", "p", "b", "gr"],
    "ear": ["d", "f", "g", "h", "n", "t", "y", "cl"],
    "oat": ["b", "c", "g", "m", "fl", "thr"],
    "ing": ["k", "r", "s", "th", "w", "br"],
    "ell": ["b", "d", "f", "s", "t", "w", "sh"],
    "ame": ["c", "f", "g", "n", "s", "t", "fl"],
    "ain": ["m", "p", "r", "g", "tr", "br", "pl"],
    "ill": ["b", "f", "h", "k", "m", "p", "w"],
    "ock": ["d", "l", "r", "s", "cl", "bl", "kn"],
    "ash": ["c", "d", "m", "r", "s", "cr", "fl"],
    "ate": ["d", "g", "h", "l", "m", "r", "pl"],
    "ay": ["d", "h", "m", "p", "r", "s", "w", "pl"],
    "ine": ["d", "f", "l", "m", "n", "p", "w", "sp"],
    "op": ["h", "m", "p", "s", "t", "sh", "st"],
    "un": ["b", "f", "g", "n", "r", "s", "sp"],
    "ore": ["c", "m", "s", "t", "w", "sh", "st"],
    "ack": ["b", "j", "p", "r", "s", "tr", "bl"],
    "ent": ["b", "d", "s", "t", "w", "sp"],
}


def syntax_sample(rng, _):
    i = rng.randrange(len(NOUNS))
    j = rng.choice([k for k in range(len(NOUNS)) if k != i])
    plural = rng.random() < 0.5
    dist = NOUNS[j][rng.random() < 0.5]
    prep = rng.choice(PREPS)
    make = lambda pl: (f"the {NOUNS[i][pl]} {prep} the {dist}", "are" if pl else "is")
    return (*make(plural), *make(not plural))


def arith_sample(rng, _):
    a, b = rng.randint(10, 49), rng.randint(10, 49)
    b2 = rng.choice([x for x in range(10, 50) if x != b])
    return f"{a} + {b} =", str(a + b), f"{a} + {b2} =", str(a + b2)


def rhyme_pairs(rng):
    pairs = []
    for suf, onsets in RHYME.items():
        ordered = [(o1, o2, suf) for o1 in onsets for o2 in onsets if o1 != o2]
        pairs += rng.sample(ordered, 10)
    rng.shuffle(pairs)
    return {"train": pairs[:140], "val": pairs[140:170], "test": pairs[170:]}


def rhyme_sampler(pairs):
    def sample(rng, split):
        o1, o2, suf = rng.choice(pairs[split])
        c1, c2, csuf = rng.choice([p for p in pairs["train"] if p[2] != suf])
        suf2 = rng.choice([s for s in RHYME if s not in (suf, csuf)])
        make = lambda s: f"rhyme : {c1} {csuf} -> {c2} {csuf} . {o1} {s} -> {o2}"
        return make(suf), suf, make(suf2), suf2
    return sample


def clause(rng):
    noun = lambda: rng.choice(NOUNS)[rng.random() < 0.5]
    return f"the {rng.choice(ADJS)} {noun()} {rng.choice(VERBS)} the {noun()} {rng.choice(PREPS)} the {noun()}"


def general_sample(rng, _):
    sentences = []
    for _ in range(rng.randint(1, 2)):
        parts = [clause(rng)]
        if rng.random() < 0.6:
            parts += [rng.choice(CONNECTORS), clause(rng)]
        sentences.append(" ".join(parts) + " .")
    text = " ".join(sentences)
    return text, PAD, text, PAD


def encode(rows, stoi):
    n = len(rows)
    clean = torch.zeros(n, MAX_LEN, dtype=torch.long)
    corrupt = torch.zeros(n, MAX_LEN, dtype=torch.long)
    pos = torch.zeros(n, dtype=torch.long)
    cid = torch.zeros(n, dtype=torch.long)
    xid = torch.zeros(n, dtype=torch.long)
    for r, (cp, ct, xp, xt) in enumerate(rows):
        c, x = [stoi[w] for w in cp.split()], [stoi[w] for w in xp.split()]
        assert len(c) == len(x) <= MAX_LEN
        clean[r, :len(c)] = torch.tensor(c)
        corrupt[r, :len(x)] = torch.tensor(x)
        pos[r] = len(c) - 1
        cid[r], xid[r] = stoi[ct], stoi[xt]
    return {"clean_ids": clean, "corrupt_ids": corrupt, "target_pos": pos, "clean_target_id": cid, "corrupt_target_id": xid}


def build(seed=0):
    rng = random.Random(seed)
    pairs = rhyme_pairs(rng)
    samplers = [syntax_sample, arith_sample, rhyme_sampler(pairs), general_sample]
    rows = {t: {s: [f(rng, s) for _ in range(n)] for s, n in SPLITS.items()} for t, f in zip(TASKS, samplers)}
    words = {w for t in rows.values() for split in t.values() for row in split for part in row for w in part.split()}
    vocab = [PAD] + sorted(words - {PAD})
    stoi = {w: i for i, w in enumerate(vocab)}
    OUT.mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(TASKS):
        torch.save({s: encode(rows[t][s], stoi) for s in SPLITS}, OUT / f"task_{i}.pt")
    (OUT / "vocab.json").write_text(json.dumps(vocab))
    (OUT / "rhyme_pairs.json").write_text(json.dumps(pairs))


def load_split(split):
    return {t: torch.load(OUT / f"task_{i}.pt")[split] for i, t in enumerate(TASKS)}


def load_vocab():
    return json.loads((OUT / "vocab.json").read_text())


def verify():
    vocab = load_vocab()
    for i, t in enumerate(TASKS):
        data = torch.load(OUT / f"task_{i}.pt")
        for s, n in SPLITS.items():
            d = data[s]
            assert d["clean_ids"].shape == (n, MAX_LEN) and d["corrupt_ids"].shape == (n, MAX_LEN)
            assert all(v.dtype == torch.long for v in d.values())
            assert int(d["clean_ids"].max()) < len(vocab)
            if t in SPECIALIST:
                assert bool((d["clean_target_id"] != d["corrupt_target_id"]).all())
        print(f"{t}: train/val/test = {', '.join(str(len(data[s]['target_pos'])) for s in SPLITS)}, max prompt len = {int(data['train']['target_pos'].max()) + 1}")
    pairs = json.loads((OUT / "rhyme_pairs.json").read_text())
    seen = {tuple(p) for p in pairs["train"]}
    assert not seen & {tuple(p) for p in pairs["val"] + pairs["test"]}
    print(f"vocab size = {len(vocab)}; rhyme pairs train/val/test = {len(pairs['train'])}/{len(pairs['val'])}/{len(pairs['test'])} (disjoint)")
    print("verification passed: 100% of specialist samples have clean_target != corrupt_target")


if __name__ == "__main__":
    build()
    verify()
