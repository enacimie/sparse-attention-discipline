"""Needle-in-a-haystack: synthetic long-context retrieval evaluation.

Builds a long filler context with a "needle" fact inserted at a given depth,
then asks the model to retrieve it. This is the task most sensitive to full
attention, used to compare dense vs approximate attention.

The needle is a random passkey; success = the passkey appears in the generation.
"""

import random

import torch

FILLER = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
    "Bright vixens jump; dozy fowl quack. "
)


def build_needle_prompt(tokenizer, context_len: int, needle: str, depth: float,
                        question: str = None):
    """Insert needle at relative depth (0.0=start, 1.0=end) within context_len tokens.

    Returns (prompt_text, needle). The prompt is a plain text (no chat template)
    suitable for base models and for controlled retrieval measurement.
    """
    if question is None:
        question = "What is the special passkey mentioned in the text?"
    needle_sentence = f"The special passkey is {needle}."
    # Rough filler text long enough; we then truncate by tokens.
    n_rep = max(1, (context_len * 5) // len(FILLER) + 1)
    body = FILLER * n_rep
    # Insert needle at the depth position (character-level approximation).
    pos = int(len(body) * depth)
    text = body[:pos] + " " + needle_sentence + " " + body[pos:]
    text = text + "\n\nQuestion: " + question + "\nAnswer:"
    # Truncate to context_len tokens (keeping the tail question).
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    if ids.numel() > context_len:
        # keep head+needle region and the tail question
        head = ids[: int(context_len * 0.7)]
        tail = ids[-int(context_len * 0.3):]
        ids = torch.cat([head, tail])
    return tokenizer.decode(ids), needle


def make_needle(rng: random.Random) -> str:
    return " ".join(str(rng.randint(0, 9)) for _ in range(5))


def check_needle(generated: str, needle: str) -> bool:
    return needle in generated


_ADJS = ["crimson", "golden", "silver", "emerald", "cobalt", "amber", "violet", "scarlet"]
_NOUNS = ["falcon", "panther", "harbor", "meadow", "glacier", "lantern", "thicket", "summit"]


def build_multifact_haystack(tokenizer, context_len, n_facts, rng, target_depth=0.5):
    """Harder RULER-style fact lookup with lexical distractors.

    Inserts n_facts sentences 'The access code for <key> is <value>.' into neutral
    filler. The target key shares one token (adjective OR noun) with several distractor
    keys, so a retriever must discriminate the exact key, not just match a keyword.

    Returns (body_text, facts, target_key, target_value).
    """
    adj_t, noun_t = rng.choice(_ADJS), rng.choice(_NOUNS)
    target_key = f"{adj_t} {noun_t}"
    used = {(adj_t, noun_t)}
    keys = [target_key]
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
    values = rng.sample(range(10000, 99999), n_facts)
    target_value = values[0]
    facts = {keys[i]: values[i] for i in range(n_facts)}
    fact_sentences = [f"The access code for {k} is {v}." for k, v in facts.items()]

    other_depths = sorted(rng.random() for _ in range(n_facts - 1))
    items = sorted(zip([target_depth] + other_depths, fact_sentences))

    q_tokens = 40
    fact_tokens = len(tokenizer(" ".join(fact_sentences), add_special_tokens=False).input_ids)
    n_rep = context_len // 30 + 5          # FILLER ~= 43 tok; over-provision then truncate
    filler_ids = tokenizer(FILLER * n_rep, add_special_tokens=False).input_ids
    filler_ids = filler_ids[: max(context_len - fact_tokens - q_tokens, 0)]
    n_filler = len(filler_ids)

    pieces, prev = [], 0
    for d, s in items:
        pos = int(n_filler * d)
        pieces.append(filler_ids[prev:pos])
        pieces.append(tokenizer(" " + s + " ", add_special_tokens=False).input_ids)
        prev = pos
    pieces.append(filler_ids[prev:])
    body_ids = [t for p in pieces for t in p]
    return tokenizer.decode(body_ids), facts, target_key, target_value
