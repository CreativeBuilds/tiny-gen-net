"""Task text -> embedding for Phase 4 conditioning (keyword bag-of-words, learnable)."""

import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.arch_gen.spec import DEPTH_CHOICES, HIDDEN_CHOICES, ArchSpec

KEYWORDS = (
    "char", "predictor", "small", "medium", "large", "wide", "narrow",
    "shallow", "deep", "balanced", "skip", "connections", "simple",
    "efficient", "lightweight", "compact", "fast", "capacity", "reasoning",
)
TASK_DIM = 32
SYNONYMS = {"connection": "connections", "connections": "skip", "efficiency": "efficient", "light": "lightweight"}


def normalize_task_text(text: str) -> str:
    t = text.lower()
    t = t.replace("skip connection", "skip connections")
    for a, b in SYNONYMS.items():
        t = re.sub(rf"\b{a}\b", b, t)
    if "skip connections" in t: t += " skip connections"
    return t


def describe_spec(spec: ArchSpec) -> str:
    """Canonical text label for training pairs — richer vocabulary."""
    parts = ["char", "predictor"]
    h = spec.hidden_dim
    if h <= 32: parts += ["small", "narrow", "compact", "efficient", "lightweight"]
    elif h <= 48: parts += ["small", "efficient", "lightweight"]
    elif h >= 128: parts += ["large", "wide", "capacity"]
    elif h >= 96: parts += ["medium", "wide", "capacity"]
    else: parts += ["medium", "balanced"]
    if spec.depth == 1: parts += ["shallow", "fast"]
    elif spec.depth >= 3: parts += ["deep", "reasoning"]
    else: parts += ["balanced"]
    if spec.skip: parts += ["skip", "connections"]
    else: parts += ["simple"]
    if h <= 48 and spec.depth <= 2: parts.append("efficient")
    if h >= 96 and spec.depth >= 2: parts.append("reasoning")
    return " ".join(dict.fromkeys(parts))


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", normalize_task_text(text)))


def keyword_hits(text: str) -> list[int]:
    toks = _tokens(text)
    hits = [i for i, k in enumerate(KEYWORDS) if k in toks]
    if "skip" in toks and KEYWORDS.index("connections") not in hits: hits.append(KEYWORDS.index("connections"))
    return hits


class TaskEmbedder(nn.Module):
    def __init__(self, dim: int = TASK_DIM):
        super().__init__()
        self.dim = dim
        self.kw = nn.Embedding(len(KEYWORDS), dim)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.default = nn.Parameter(torch.zeros(dim))

    def encode_text(self, text: str, device: str) -> torch.Tensor:
        hits = keyword_hits(text)
        if not hits: return self.default.to(device)
        idx = torch.tensor(hits, device=device, dtype=torch.long)
        return self.proj(self.kw(idx).mean(dim=0))

    def encode_spec(self, spec: ArchSpec, device: str) -> torch.Tensor:
        return self.encode_text(describe_spec(spec), device)


class ArchMatchHead(nn.Module):
    """Auxiliary heads: task embedding -> arch attributes (match-aware training)."""

    def __init__(self, task_dim: int = TASK_DIM):
        super().__init__()
        self.h = nn.Linear(task_dim, len(HIDDEN_CHOICES))
        self.d = nn.Linear(task_dim, len(DEPTH_CHOICES))
        self.s = nn.Linear(task_dim, 2)

    def loss(self, task_vecs: torch.Tensor, specs: list[ArchSpec]) -> torch.Tensor:
        device = task_vecs.device
        hi = torch.tensor([HIDDEN_CHOICES.index(s.hidden_dim) for s in specs], device=device)
        di = torch.tensor([DEPTH_CHOICES.index(s.depth) for s in specs], device=device)
        si = torch.tensor([int(s.skip) for s in specs], device=device)
        return (F.cross_entropy(self.h(task_vecs), hi) + F.cross_entropy(self.d(task_vecs), di) + F.cross_entropy(self.s(task_vecs), si)) / 3


def spec_matches_task(spec: ArchSpec, text: str) -> float:
    toks = _tokens(text)
    score, n = 0.0, 0
    if toks & {"small", "narrow", "compact", "lightweight", "efficient"}:
        score += 1.0 if spec.hidden_dim <= 48 else 0.0; n += 1
    if toks & {"large", "wide", "capacity"}:
        score += 1.0 if spec.hidden_dim >= 96 else 0.0; n += 1
    if toks & {"medium", "balanced"} and not toks & {"large", "small"}:
        score += 1.0 if 48 < spec.hidden_dim < 96 else 0.0; n += 1
    if "shallow" in toks or "fast" in toks:
        score += 1.0 if spec.depth == 1 else 0.0; n += 1
    if "deep" in toks or "reasoning" in toks:
        score += 1.0 if spec.depth >= 3 else 0.0; n += 1
    if toks & {"skip", "connections"}:
        score += 1.0 if spec.skip else 0.0; n += 1
    if "simple" in toks:
        score += 1.0 if not spec.skip else 0.0; n += 1
    return score / max(n, 1)


def arch_summary(spec: ArchSpec) -> str:
    return f"h={spec.hidden_dim} depth={spec.depth} skip={spec.skip} params≈{spec.hidden_dim}"


DEFAULT_EVAL_TASKS = [
    "small shallow simple char predictor",
    "small efficient lightweight char predictor",
    "large deep skip connections char predictor",
    "large wide capacity char predictor",
    "medium balanced char predictor",
    "deep balanced reasoning char predictor",
    "shallow fast char predictor",
    "small narrow compact char predictor",
    "large deep char predictor",
    "medium simple char predictor",
    "efficient shallow char predictor",
    "wide deep skip connections char predictor",
    "small balanced char predictor",
    "large capacity reasoning char predictor",
    "medium deep char predictor",
    "lightweight fast char predictor",
    "narrow deep char predictor",
    "large simple char predictor",
    "medium wide balanced char predictor",
    "small deep reasoning char predictor",
    "compact shallow simple char predictor",
    "wide large skip connections char predictor",
    "balanced medium char predictor",
    "efficient medium char predictor",
    "deep reasoning capacity char predictor",
]

DEMO_EXAMPLES = [
    "small shallow simple char predictor",
    "large deep skip connections char predictor",
    "efficient lightweight char predictor",
    "deep reasoning capacity char predictor",
    "medium balanced char predictor",
]
