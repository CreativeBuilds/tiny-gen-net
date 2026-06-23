"""Shared training loop for TinyMLP on synthetic char task."""

import random
from dataclasses import dataclass

import torch
import torch.nn as nn

from data.synthetic_text import SyntheticConfig, corpus_to_tensor_pairs, generate_corpus
from src.models.tiny_mlp import TinyMLP
from src.utils.device import device_str


@dataclass
class TrainConfig:
    steps: int = 500
    lr: float = 1e-3
    hidden_dim: int = 64
    batch_size: int = 64
    seed: int = 42
    device: str = ""
    skip: int = 0  # 0=next-char, k>0=skip-char (predict char[t+k] from char[t])


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


def build_batch(pairs: list, batch_size: int, device: str):
    batch = random.sample(pairs, min(batch_size, len(pairs)))
    xs = torch.stack([b[0] for b in batch]).to(device)
    ys = torch.stack([b[1] for b in batch]).to(device)
    return xs, ys


def train_tiny_mlp(cfg: TrainConfig | None = None) -> tuple[TinyMLP, list[float], dict]:
    """Train one TinyMLP; returns model, loss history, final metrics."""
    cfg = cfg or TrainConfig()
    set_seed(cfg.seed)
    device = cfg.device or device_str()
    corpus = generate_corpus(SyntheticConfig(seed=cfg.seed))
    pairs = list(corpus_to_tensor_pairs(corpus, skip=cfg.skip))
    if not pairs: raise RuntimeError("Empty training pairs from corpus")

    model = TinyMLP(hidden_dim=cfg.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    crit = nn.CrossEntropyLoss()
    losses: list[float] = []

    model.train()
    for step in range(cfg.steps):
        xs, ys = build_batch(pairs, cfg.batch_size, device)
        logits, _ = model(xs)
        loss = crit(logits, ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    # Quick eval: average loss on full corpus (cheap for tiny data)
    model.eval()
    with torch.no_grad():
        eval_losses = []
        correct = 0
        total = 0
        for xs, ys in [build_batch(pairs, cfg.batch_size, device) for _ in range(min(20, len(pairs) // cfg.batch_size + 1))]:
            logits, _ = model(xs)
            eval_losses.append(crit(logits, ys).item())
            correct += (logits.argmax(-1) == ys).sum().item()
            total += ys.numel()
    metrics = {
        "final_train_loss": losses[-1],
        "eval_loss": sum(eval_losses) / max(len(eval_losses), 1),
        "eval_acc": correct / max(total, 1),
        "num_params": model.num_parameters(),
        "steps": cfg.steps,
        "seed": cfg.seed,
    }
    return model, losses, metrics


def eval_model_loss(model: TinyMLP, seed: int = 0, device: str = "cpu", batches: int = 20, skip: int = 0) -> dict:
    """Evaluate model on synthetic data (for comparing inits)."""
    set_seed(seed + 999)
    corpus = generate_corpus(SyntheticConfig(seed=seed + 999))
    pairs = list(corpus_to_tensor_pairs(corpus, skip=skip))
    crit = nn.CrossEntropyLoss()
    model.eval()
    losses, correct, total = [], 0, 0
    with torch.no_grad():
        for _ in range(batches):
            xs, ys = build_batch(pairs, 64, device)
            logits, _ = model(xs.to(device))
            losses.append(crit(logits, ys).item())
            correct += (logits.argmax(-1) == ys).sum().item()
            total += ys.numel()
    return {"eval_loss": sum(losses) / len(losses), "eval_acc": correct / max(total, 1)}


def finetune_steps(model: TinyMLP, steps: int, seed: int = 0, lr: float = 1e-3, device: str = "cpu", skip: int = 0) -> list[float]:
    """Short fine-tune from current weights; returns loss curve."""
    set_seed(seed + 1234)
    corpus = generate_corpus(SyntheticConfig(seed=seed + 1234))
    pairs = list(corpus_to_tensor_pairs(corpus, skip=skip))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    losses: list[float] = []
    model.train()
    for _ in range(steps):
        xs, ys = build_batch(pairs, 64, device)
        logits, _ = model(xs.to(device))
        loss = crit(logits, ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses
