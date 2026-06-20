"""Task text -> NanoSpec with sentence embeddings (Phase 5a)."""

import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.arch_gen.nano_spec import N_EMBD_CHOICES, N_HEAD_CHOICES, N_LAYER_CHOICES, BLOCK_SIZE_CHOICES, NanoSpec
from src.arch_gen.sentence_embedder import SENT_DIM, SentenceTaskEmbedder

NANO_AUG = {"big": "large", "heavy": "capacity", "stacked": "deep", "many": "deep", "light": "lightweight", "mini": "small", "tiny": "small", "gpt": "transformer", "nano": "small"}

DEFAULT_NANO_EVAL_TASKS = [
    "small fast shakespeare char completion model",
    "medium balanced text autocompletion transformer",
    "large wide capacity shakespeare language model",
    "deep reasoning transformer for text prediction",
    "lightweight efficient char-level nano gpt",
    "large deep shakespeare transformer with long context",
    "compact small model for quick text completion",
    "wide medium transformer for character prediction",
]


def normalize_nano_task(text: str) -> str:
    t = text.lower().strip()
    for a, b in NANO_AUG.items(): t = re.sub(rf"\b{a}\b", b, t)
    if "shakespeare" not in t and "text" not in t: t += " shakespeare text"
    if "transformer" not in t and "gpt" not in t: t += " transformer"
    return t


def describe_nano_spec(spec: NanoSpec) -> str:
    parts = ["shakespeare", "text", "autocompletion", "transformer", "char", "nano"]
    e, l, h, b = spec.n_embd, spec.n_layer, spec.n_head, spec.block_size
    if e <= 128: parts += ["small", "compact", "efficient", "lightweight", "fast"]
    elif e <= 192: parts += ["medium", "balanced"]
    elif e <= 256: parts += ["medium", "wide", "capacity"]
    else: parts += ["large", "wide", "capacity", "reasoning"]
    if l <= 4: parts += ["shallow", "fast"]
    elif l >= 8: parts += ["deep", "reasoning"]
    else: parts += ["balanced", "deep"]
    if h >= 8 and e >= 192: parts.append("wide")
    if b >= 256: parts += ["long", "context"]
    return " ".join(dict.fromkeys(parts))


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", normalize_nano_task(text)))


def nano_spec_matches_task(spec: NanoSpec, text: str) -> float:
    toks = _tokens(text)
    score, n = 0.0, 0
    if toks & {"small", "compact", "lightweight", "efficient", "fast"}:
        score += 1.0 if spec.n_embd <= 128 else 0.0; n += 1
    if toks & {"large", "wide", "capacity"}:
        score += 1.0 if spec.n_embd >= 256 else 0.0; n += 1
    if toks & {"medium", "balanced"} and not toks & {"large", "small"}:
        score += 1.0 if 128 < spec.n_embd <= 256 else 0.0; n += 1
    if "shallow" in toks or "fast" in toks:
        score += 1.0 if spec.n_layer <= 4 else 0.0; n += 1
    if "deep" in toks or "reasoning" in toks:
        score += 1.0 if spec.n_layer >= 8 else 0.0; n += 1
    if "long" in toks or "context" in toks:
        score += 1.0 if spec.block_size >= 256 else 0.0; n += 1
    return score / max(n, 1)


class NanoMatchHead(nn.Module):
    def __init__(self, task_dim: int = SENT_DIM):
        super().__init__()
        self.d_score = nn.Linear(task_dim, 1)
        self.l_score = nn.Linear(task_dim, len(N_LAYER_CHOICES))
        self.e_score = nn.Linear(task_dim, len(N_EMBD_CHOICES))

    def loss(self, task_vecs: torch.Tensor, specs: list[NanoSpec]) -> torch.Tensor:
        li = torch.tensor([N_LAYER_CHOICES.index(s.n_layer) for s in specs], device=task_vecs.device)
        ei = torch.tensor([N_EMBD_CHOICES.index(s.n_embd) for s in specs], device=task_vecs.device)
        return F.cross_entropy(self.l_score(task_vecs), li) + F.cross_entropy(self.e_score(task_vecs), ei)


def augment_nano_training_pairs(specs: list[NanoSpec], texts: list[str], rng) -> tuple[list[NanoSpec], list[str]]:
    out_s, out_t = list(specs), list(texts)
    syns = [("small", "compact"), ("large", "wide capacity"), ("deep", "reasoning"), ("fast", "efficient"), ("text", "shakespeare")]
    for spec, text in zip(specs, texts):
        for a, b in syns:
            if a in text: out_s.append(spec); out_t.append(text.replace(a, b))
        out_s.append(spec); out_t.append(normalize_nano_task(text))
    paired = list(zip(out_s, out_t)); rng.shuffle(paired)
    return [p[0] for p in paired], [p[1] for p in paired]
