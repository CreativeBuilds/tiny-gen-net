"""Synthetic character-level corpus for Phase 0 baselines.

Generates short repeating-pattern strings over a tiny alphabet.
Easy to overfit, fast to train, reproducible.

Supports two prediction modes:
  - next_char (default): predict char[t+1] from char[t] — solvable by bigram table
  - skip_char: predict char[t+k] from char[t] — requires multi-step transitions
    that exercise the recurrent hidden state (where depth/width matter)
"""

import random
from dataclasses import dataclass

# Tiny 8-char alphabet (indices 0-7)
CHARS = "abcdefgh"
CHAR_TO_IDX = {c: i for i, c in enumerate(CHARS)}
IDX_TO_CHAR = {i: c for i, c in enumerate(CHARS)}
VOCAB_SIZE = len(CHARS)

# Base patterns — model learns local structure + repeats
BASE_PATTERNS = ["abcd", "abdc", "acbd", "abcdhgfe", "aabbc", "abcabc"]


@dataclass
class SyntheticConfig:
    num_sequences: int = 512
    min_len: int = 64
    max_len: int = 128
    seed: int = 42
    skip: int = 0  # 0 = next-char prediction, k>0 = predict char[t+k] from char[t]


def generate_corpus(cfg: SyntheticConfig | None = None) -> list[str]:
    """Build list of training strings from random pattern repeats."""
    cfg = cfg or SyntheticConfig()
    rng = random.Random(cfg.seed)
    corpus: list[str] = []
    for _ in range(cfg.num_sequences):
        pat = rng.choice(BASE_PATTERNS)
        target_len = rng.randint(cfg.min_len, cfg.max_len)
        s = []
        while len("".join(s)) < target_len:
            s.append(pat)
            if rng.random() < 0.3:
                s.append(rng.choice(BASE_PATTERNS))
        corpus.append("".join(s)[:target_len])
    return corpus


def encode_char(c: str) -> int:
    return CHAR_TO_IDX.get(c, 0)


def decode_char(i: int) -> str:
    return IDX_TO_CHAR.get(i % VOCAB_SIZE, "a")


def corpus_to_tensor_pairs(corpus: list[str], skip: int = 0):
    """Yield (input_idx, target_idx) prediction pairs.

    skip=0: predict char[t+1] from char[t] (next-char, bigram-solvable)
    skip=k: predict char[t+k] from char[t] (skip-char, requires multi-step)
    """
    import torch
    offset = skip + 1 if skip > 0 else 1
    for text in corpus:
        indices = [encode_char(c) for c in text]
        if len(indices) < offset + 1:
            continue
        for t in range(len(indices) - offset):
            yield torch.tensor(indices[t], dtype=torch.long), torch.tensor(indices[t + offset], dtype=torch.long)
