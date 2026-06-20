"""Task text -> TxSpec mapping with tx-calibrated tiers and training augmentation."""

import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.arch_gen.task_cond import TASK_DIM, TaskEmbedder, normalize_task_text
from src.arch_gen.tx_spec import D_MODEL_CHOICES, N_HEAD_CHOICES, N_LAYER_CHOICES, TxSpec

TX_AUG = {"big": "large", "heavy": "capacity", "stacked": "deep", "many": "deep", "light": "lightweight", "mini": "small", "tiny": "small"}


def enrich_tx_task(text: str) -> str:
    t = normalize_task_text(text)
    for a, b in TX_AUG.items(): t = re.sub(rf"\b{a}\b", b, t)
    if "capacity" in t or "reasoning" in t: t += " large wide"
    if "deep" in t and "shallow" not in t: t += " reasoning"
    if "large" in t or "capacity" in t: t += " wide"
    if "transformer" not in t: t += " transformer"
    return t


def describe_tx_spec(spec: TxSpec) -> str:
    parts = ["char", "predictor", "transformer"]
    d, l, h = spec.d_model, spec.n_layer, spec.n_head
    if d <= 32: parts += ["tiny", "small", "compact", "efficient", "lightweight"]
    elif d <= 48: parts += ["small", "compact", "efficient", "lightweight"]
    elif d <= 64: parts += ["medium", "balanced"]
    elif d <= 96: parts += ["medium", "wide", "capacity"]
    else: parts += ["large", "wide", "capacity", "reasoning"]
    if l <= 2: parts += ["shallow", "fast"]
    elif l >= 5: parts += ["deep", "reasoning"]
    elif l >= 4: parts += ["deep"]
    else: parts += ["balanced"]
    if h >= 8 and d >= 64: parts.append("wide")
    if l >= 5 and d >= 96: parts.append("capacity")
    return " ".join(dict.fromkeys(parts))


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", enrich_tx_task(text)))


def tx_spec_matches_task(spec: TxSpec, text: str) -> float:
    toks = _tokens(text)
    score, n = 0.0, 0
    if toks & {"small", "compact", "lightweight", "efficient", "tiny"}:
        score += 1.0 if spec.d_model <= 48 else 0.0; n += 1
    if toks & {"large", "wide", "capacity"}:
        score += 1.0 if spec.d_model >= 96 else 0.0; n += 1
    if toks & {"medium", "balanced"} and not toks & {"large", "small", "tiny"}:
        score += 1.0 if 48 < spec.d_model <= 96 else 0.0; n += 1
    if "shallow" in toks or "fast" in toks:
        score += 1.0 if spec.n_layer <= 2 else 0.0; n += 1
    if "deep" in toks or "reasoning" in toks:
        score += 1.0 if spec.n_layer >= 4 else 0.0; n += 1
    return score / max(n, 1)


def augment_tx_training_pairs(specs: list[TxSpec], texts: list[str]) -> tuple[list[TxSpec], list[str]]:
    out_s, out_t, seen = [], [], set()
    for spec, text in zip(specs, texts):
        variants = {describe_tx_spec(spec), enrich_tx_task(text)}
        d, l = spec.d_model, spec.n_layer
        if d <= 48: variants.add("small compact efficient lightweight transformer char predictor")
        if d >= 96: variants.add("large wide capacity transformer char predictor")
        if l >= 5: variants.add("deep reasoning capacity transformer char predictor")
        if l <= 2: variants.add("shallow fast lightweight transformer char predictor")
        if 48 < d <= 96 and 3 <= l <= 4: variants.add("medium balanced wide transformer char predictor")
        for v in variants:
            key = (spec.d_model, spec.n_layer, spec.n_head, v)
            if key in seen: continue
            seen.add(key); out_s.append(spec); out_t.append(v)
    return out_s, out_t


def tx_arch_summary(spec: TxSpec) -> str:
    return f"d={spec.d_model} L={spec.n_layer} H={spec.n_head} ctx={spec.ctx_len}"


class TxMatchHead(nn.Module):
    def __init__(self, task_dim: int = TASK_DIM):
        super().__init__()
        self.d = nn.Linear(task_dim, len(D_MODEL_CHOICES))
        self.l = nn.Linear(task_dim, len(N_LAYER_CHOICES))
        self.h = nn.Linear(task_dim, len(N_HEAD_CHOICES))

    def loss(self, task_vecs: torch.Tensor, specs: list[TxSpec]) -> torch.Tensor:
        device = task_vecs.device
        di = torch.tensor([D_MODEL_CHOICES.index(s.d_model) for s in specs], device=device)
        li = torch.tensor([N_LAYER_CHOICES.index(s.n_layer) for s in specs], device=device)
        hi = torch.tensor([N_HEAD_CHOICES.index(s.n_head) for s in specs], device=device)
        return (F.cross_entropy(self.d(task_vecs), di) + F.cross_entropy(self.l(task_vecs), li) + F.cross_entropy(self.h(task_vecs), hi)) / 3


DEFAULT_TX_EVAL_TASKS = [
    "small shallow fast char predictor transformer",
    "small efficient lightweight transformer char predictor",
    "large deep reasoning char predictor transformer",
    "large wide capacity transformer char predictor",
    "medium balanced char predictor transformer",
    "deep balanced reasoning transformer char predictor",
    "compact shallow char predictor transformer",
    "large deep char predictor transformer",
    "medium simple char predictor transformer",
    "efficient shallow transformer char predictor",
    "small balanced transformer char predictor",
    "large capacity reasoning transformer char predictor",
    "medium deep char predictor transformer",
    "wide deep transformer char predictor",
    "large simple transformer char predictor",
    "medium wide balanced transformer char predictor",
    "small deep reasoning transformer char predictor",
    "compact shallow simple transformer char predictor",
    "wide large transformer char predictor",
    "balanced medium transformer char predictor",
    "efficient medium transformer char predictor",
    "deep reasoning capacity transformer char predictor",
    "small fast char predictor transformer",
    "medium char predictor transformer",
    "large transformer char predictor",
]

TX_DEMO_EXAMPLES = DEFAULT_TX_EVAL_TASKS[:5]
