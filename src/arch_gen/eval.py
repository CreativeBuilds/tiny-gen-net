"""Eval VariableTinyMLP on synthetic char task (mirrors train_loop helpers)."""

import random

import torch
import torch.nn as nn

from data.synthetic_text import SyntheticConfig, corpus_to_tensor_pairs, generate_corpus
from src.models.variable_mlp import VariableTinyMLP
from src.utils.train_loop import build_batch, set_seed


def eval_arch_model(model: VariableTinyMLP, seed: int = 0, device: str = "cpu", batches: int = 20) -> dict:
    set_seed(seed + 999)
    corpus = generate_corpus(SyntheticConfig(seed=seed + 999))
    pairs = list(corpus_to_tensor_pairs(corpus))
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


def finetune_arch(model: VariableTinyMLP, steps: int, seed: int = 0, lr: float = 1e-3, device: str = "cpu") -> list[float]:
    set_seed(seed + 1234)
    corpus = generate_corpus(SyntheticConfig(seed=seed + 1234))
    pairs = list(corpus_to_tensor_pairs(corpus))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    losses: list[float] = []
    model.train()
    for _ in range(steps):
        xs, ys = build_batch(pairs, 64, device)
        logits, _ = model(xs.to(device))
        loss = crit(logits, ys)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return losses


def eval_arch_row(model: VariableTinyMLP, device: str, ft_steps: list[int], seed: int) -> dict:
    out = {}
    z = eval_arch_model(model, seed=seed, device=device)
    out["zero_loss"], out["zero_acc"] = z["eval_loss"], z["eval_acc"]
    for n in ft_steps:
        if n <= 0: continue
        mc = VariableTinyMLP(model.spec)
        mc.load_state_dict(model.state_dict())
        mc.to(device)
        finetune_arch(mc, n, seed=seed, device=device)
        a = eval_arch_model(mc, seed=seed, device=device)
        out[f"ft{n}_loss"], out[f"ft{n}_acc"] = a["eval_loss"], a["eval_acc"]
    return out
