"""RULER-style multi-task battery.

Covers three capability families, in the spirit of RULER (Hsieh et al. 2024):
  - Retrieval / NIAH variants: single, multikey, multivalue, multiquery.
  - Reasoning: variable tracking (VT) — multi-hop chain over the context.
  - Extraction: frequent-word extraction (FWE).

Each builder returns (body, question, targets) where `targets` is a list of
strings that must ALL appear in the generation for a hit. The haystack body is
built by interleaving task sentences into neutral filler at sampled depths, up to
`context_len` tokens.
"""

import random

from needle import FILLER, _ADJS, _NOUNS


def _filler_ids(tok, n):
    ids = tok(FILLER * (n // 20 + 2), add_special_tokens=False).input_ids
    return ids[:max(n, 0)]


def _interleave(tok, context_len, sentences, depths, reserve_q=48):
    """Interleave `sentences` at fractional `depths` inside filler, return body text."""
    sent_ids = [tok(" " + s + " ", add_special_tokens=False).input_ids for s in sentences]
    total_sent = sum(len(x) for x in sent_ids)
    n_fill = max(context_len - total_sent - reserve_q, 0)
    filler = _filler_ids(tok, n_fill)
    items = sorted(zip(depths, sent_ids))
    pieces, prev = [], 0
    for d, sid in items:
        pos = int(len(filler) * d)
        pieces.append(filler[prev:pos])
        pieces.append(sid)
        prev = pos
    pieces.append(filler[prev:])
    body_ids = [t for p in pieces for t in p]
    return tok.decode(body_ids)


def build_niah_single(tok, ctx, rng):
    key = f"{rng.choice(_ADJS)} {rng.choice(_NOUNS)}"
    val = rng.randint(10000, 99999)
    body = _interleave(tok, ctx, [f"The access code for {key} is {val}."], [rng.random()])
    return body, f"What is the access code for {key}?", [str(val)]


def build_niah_multikey(tok, ctx, rng, n_facts=4):
    used, keys = set(), []
    while len(keys) < n_facts:
        k = f"{rng.choice(_ADJS)} {rng.choice(_NOUNS)}"
        if k not in used:
            used.add(k)
            keys.append(k)
    target = keys[0]
    vals = {k: rng.randint(10000, 99999) for k in keys}
    sents = [f"The access code for {k} is {vals[k]}." for k in keys]
    depths = sorted(rng.random() for _ in range(n_facts))
    body = _interleave(tok, ctx, sents, depths)
    return body, f"What is the access code for {target}?", [str(vals[target])]


def build_niah_multivalue(tok, ctx, rng, n_vals=3):
    key = f"{rng.choice(_ADJS)} {rng.choice(_NOUNS)}"
    vals = rng.sample(range(10000, 99999), n_vals)
    sents = [f"One access code for {key} is {v}." for v in vals]
    depths = sorted(rng.random() for _ in range(n_vals))
    body = _interleave(tok, ctx, sents, depths)
    # ask for the LAST stated code (requires tracking all mentions)
    return body, f"What is the last access code given for {key}?", [str(vals[-1])]


def build_niah_multiquery(tok, ctx, rng, n_query=3):
    used, keys = set(), []
    while len(keys) < n_query:
        k = f"{rng.choice(_ADJS)} {rng.choice(_NOUNS)}"
        if k not in used:
            used.add(k)
            keys.append(k)
    vals = {k: rng.randint(10000, 99999) for k in keys}
    sents = [f"The access code for {k} is {vals[k]}." for k in keys]
    depths = sorted(rng.random() for _ in range(n_query))
    body = _interleave(tok, ctx, sents, depths)
    q = "Give the access codes for " + ", ".join(keys) + " in order."
    return body, q, [str(vals[k]) for k in keys]


_CHAIN_NAMES = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]


def build_vt(tok, ctx, rng, chain=4, n_distract=2):
    """Variable tracking: final -> ... -> first -> number, scattered + direct distractors."""
    number = rng.randint(1000, 9999)
    names = _CHAIN_NAMES[:chain]
    sents = [f"The value of {names[0]} is {number}."]
    for i in range(1, chain):
        sents.append(f"The value of {names[i]} is the value of {names[i-1]}.")
    for j in range(n_distract):
        sents.append(f"The value of register_{j} is {rng.randint(1000, 9999)}.")
    depths = [rng.random() for _ in range(len(sents))]
    body = _interleave(tok, ctx, sents, depths)
    return body, f"What is the value of {names[-1]}?", [str(number)]


def build_fwe(tok, ctx, rng, n_cand=4):
    """Frequent-word extraction: one candidate word is injected far more often."""
    words = rng.sample(["quartz", "onyx", "zephyr", "juniper", "cinder", "vortex",
                        "bramble", "solstice"], n_cand)
    winner = words[0]
    hi = rng.randint(22, 30)
    lo = rng.sample(range(4, 12), n_cand - 1)           # clear gap vs the winner
    counts = [hi] + lo
    sents = []
    for w, c in zip(words, counts):
        for _ in range(c):
            sents.append(f"Remember the word {w}.")
    depths = [rng.random() for _ in range(len(sents))]
    body = _interleave(tok, ctx, sents, depths, reserve_q=60)
    q = ("Among the words " + ", ".join(words) +
         ", which one appears most frequently in the text above? Answer with a single word.")
    return body, q, [winner]


def build_niah_single_distract(tok, ctx, rng, n_distract=5):
    """single_2: one target needle plus distractor needles with different keys."""
    used, keys = set(), []
    while len(keys) < n_distract + 1:
        k = f"{rng.choice(_ADJS)} {rng.choice(_NOUNS)}"
        if k not in used:
            used.add(k)
            keys.append(k)
    target = keys[0]
    vals = {k: rng.randint(10000, 99999) for k in keys}
    sents = [f"The special magic number for {k} is {vals[k]}." for k in keys]
    depths = [rng.random() for _ in range(len(sents))]
    body = _interleave(tok, ctx, sents, depths)
    return body, f"What is the special magic number for {target}?", [str(vals[target])]


def build_niah_single_samekey(tok, ctx, rng, n_same=4):
    """single_3: distractors share the target key but carry different values."""
    key = f"{rng.choice(_ADJS)} {rng.choice(_NOUNS)}"
    vals = rng.sample(range(10000, 99999), n_same + 1)
    target_val = vals[0]
    sents = [f"The final access code for {key} is {target_val}."]
    for v in vals[1:]:
        sents.append(f"A provisional access code for {key} is {v}.")
    depths = [rng.random() for _ in range(len(sents))]
    body = _interleave(tok, ctx, sents, depths)
    return body, f"What is the final access code for {key}?", [str(target_val)]


def build_niah_multikey_shared(tok, ctx, rng, n_facts=5):
    """multikey_2: distractor keys share an adjective or noun with the target key."""
    adj_t, noun_t = rng.choice(_ADJS), rng.choice(_NOUNS)
    target = f"{adj_t} {noun_t}"
    used, keys = {(adj_t, noun_t)}, [target]
    while len(keys) < n_facts:
        mode = rng.choice(["adj", "noun", "fresh"])
        if mode == "adj":
            k = (adj_t, rng.choice([n for n in _NOUNS if n != noun_t]))
        elif mode == "noun":
            k = (rng.choice([a for a in _ADJS if a != adj_t]), noun_t)
        else:
            k = (rng.choice(_ADJS), rng.choice(_NOUNS))
        if k not in used:
            used.add(k)
            keys.append(f"{k[0]} {k[1]}")
    vals = {k: rng.randint(10000, 99999) for k in keys}
    sents = [f"The access code for {k} is {vals[k]}." for k in keys]
    depths = [rng.random() for _ in range(len(sents))]
    body = _interleave(tok, ctx, sents, depths)
    return body, f"What is the access code for {target}?", [str(vals[target])]


def build_niah_multikey_many(tok, ctx, rng):
    """multikey_3: many keys (8), query one."""
    return build_niah_multikey(tok, ctx, rng, n_facts=8)


def build_cwe(tok, ctx, rng, n_cand=4):
    """Common-word extraction: candidate words embedded in varied sentences."""
    words = rng.sample(["marble", "thistle", "lantern", "orchid", "bastion",
                        "rivulet", "gossamer", "parapet"], n_cand)
    winner = words[0]
    hi = rng.randint(28, 36)
    lo = rng.sample(range(3, 8), n_cand - 1)
    counts = [hi] + lo
    sents = []
    for w, c in zip(words, counts):
        for _ in range(c):
            sents.append(f"The catalog lists the word {w}.")
    depths = [rng.random() for _ in range(len(sents))]
    body = _interleave(tok, ctx, sents, depths, reserve_q=60)
    q = ("Among the words " + ", ".join(words) +
         ", which one is mentioned most often in the text above? Answer with a single word.")
    return body, q, [winner]


_SQUAD = None
_HOTPOT = None


def _load_squad():
    global _SQUAD
    if _SQUAD is None:
        from datasets import load_dataset
        _SQUAD = load_dataset("rajpurkar/squad", split="validation")
    return _SQUAD


def _load_hotpot():
    global _HOTPOT
    if _HOTPOT is None:
        from datasets import load_dataset
        _HOTPOT = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split="validation")
    return _HOTPOT


def _para_body_gold(tok, ctx, rng, gold_paras, distract_paras, reserve_q=48):
    """Fill ctx with distractors, guaranteeing the gold paragraph(s) are included."""
    budget = ctx - reserve_q

    def enc(paras):
        ids = []
        for p in paras:
            ids.extend(tok(" " + p.strip() + " ", add_special_tokens=False).input_ids)
        return ids

    gold_ids = enc(gold_paras)
    dist_ids = enc(distract_paras)
    dist_ids = dist_ids[: max(budget - len(gold_ids), 0)]
    pos = int(len(dist_ids) * rng.random())
    body_ids = dist_ids[:pos] + gold_ids + dist_ids[pos:]
    return tok.decode(body_ids)


def build_qa_squad(tok, ctx, rng):
    """qa_1 (SQuAD): gold paragraph among distractor paragraphs, answer a real question."""
    ds = _load_squad()
    ex = None
    for _ in range(30):
        cand = ds[rng.randrange(len(ds))]
        a = cand["answers"]["text"]
        if a and 1 <= len(a[0].split()) <= 5:
            ex = cand
            break
    if ex is None:
        ex = ds[0]
    answer = ex["answers"]["text"][0]
    distract = []
    seen = {ex["id"]}
    while len(distract) < 40:
        d = ds[rng.randrange(len(ds))]
        if d["id"] not in seen:
            seen.add(d["id"])
            distract.append(d["context"])
    rng.shuffle(distract)
    body = _para_body_gold(tok, ctx, rng, [ex["context"]], distract)
    return body, ex["question"], [answer]


def build_qa_hotpot(tok, ctx, rng):
    """qa_2 (HotpotQA): supporting paragraphs among distractors, multi-hop question."""
    ds = _load_hotpot()
    ex = None
    for _ in range(30):
        cand = ds[rng.randrange(len(ds))]
        if 1 <= len(cand["answer"].split()) <= 5:
            ex = cand
            break
    if ex is None:
        ex = ds[0]
    titles = ex["context"]["title"]
    sents = ex["context"]["sentences"]
    supporting = set(ex["supporting_facts"]["title"])
    gold, distract = [], []
    for t, ss in zip(titles, sents):
        (gold if t in supporting else distract).append(t + ". " + " ".join(ss))
    # top up distractors from other examples to reach ctx
    seen = {ex["id"]}
    while len(distract) < 40:
        d = ds[rng.randrange(len(ds))]
        if d["id"] in seen:
            continue
        seen.add(d["id"])
        for t, ss in zip(d["context"]["title"][:2], d["context"]["sentences"][:2]):
            distract.append(t + ". " + " ".join(ss))
    rng.shuffle(distract)
    body = _para_body_gold(tok, ctx, rng, gold, distract)
    return body, ex["question"], [ex["answer"]]


TASKS = {
    "niah_single": build_niah_single,
    "niah_single_distract": build_niah_single_distract,
    "niah_single_samekey": build_niah_single_samekey,
    "niah_multikey": build_niah_multikey,
    "niah_multikey_shared": build_niah_multikey_shared,
    "niah_multikey_many": build_niah_multikey_many,
    "niah_multivalue": build_niah_multivalue,
    "niah_multiquery": build_niah_multiquery,
    "vt": build_vt,
    "fwe": build_fwe,
    "cwe": build_cwe,
    "qa_squad": build_qa_squad,
    "qa_hotpot": build_qa_hotpot,
}
