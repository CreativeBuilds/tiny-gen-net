"""Eval VariableTinyTransformer on synthetic char task (sequence windows)."""

import random

import torch
import torch.nn as nn

from data.synthetic_text import SyntheticConfig, encode_char, generate_corpus
from src.arch_gen.tx_spec import TxSpec
from src.models.variable_transformer import VariableTinyTransformer
from src.utils.train_loop import set_seed


def corpus_to_seq_pairs(corpus: list[str], ctx_len: int):
    for text in corpus:
        idx = [encode_char(c) for c in text]
        if len(idx) <= ctx_len: continue
        for t in range(len(idx) - ctx_len):
            yield idx[t : t + ctx_len], idx[t + ctx_len]


def build_seq_batch(pairs: list, batch_size: int, device: str):
    batch = random.sample(pairs, min(batch_size, len(pairs)))
    xs = torch.tensor([b[0] for b in batch], dtype=torch.long, device=device)
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
    return xs, ys


def eval_tx_model(model: VariableTinyTransformer, seed: int = 0, device: str = "cpu", batches: int = 20) -> dict:
    set_seed(seed + 999)
    corpus = generate_corpus(SyntheticConfig(seed=seed + 999))
    pairs = list(corpus_to_seq_pairs(corpus, model.ctx_len))
    crit = nn.CrossEntropyLoss()
    model.eval()
    losses, correct, total = [], 0, 0
    with torch.no_grad():
        for _ in range(batches):
            xs, ys = build_seq_batch(pairs, 64, device)
            logits, _ = model(xs)
            losses.append(crit(logits, ys).item())
            correct += (logits.argmax(-1) == ys).sum().item()
            total += ys.numel()
    return {"eval_loss": sum(losses) / len(losses), "eval_acc": correct / max(total, 1)}


def finetune_tx(model: VariableTinyTransformer, steps: int, seed: int = 0, lr: float = 1e-3, device: str = "cpu") -> list[float]:
    set_seed(seed + 1234)
    corpus = generate_corpus(SyntheticConfig(seed=seed + 1234))
    pairs = list(corpus_to_seq_pairs(corpus, model.ctx_len))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    losses: list[float] = []
    model.train()
    for _ in range(steps):
        xs, ys = build_seq_batch(pairs, 64, device)
        logits, _ = model(xs)
        loss = crit(logits, ys)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return losses


def eval_tx_flat(flat: torch.Tensor, spec: TxSpec, device: str, ft_steps: list[int], seed: int) -> dict:
    from src.models.variable_transformer import build_from_tx_spec
    from src.utils.weights import load_flat_into_model
    m = build_from_tx_spec(spec).to(device)
    load_flat_into_model(flat, m)
    return eval_tx_row(m, device, ft_steps, seed)


def eval_tx_row(model: VariableTinyTransformer, device: str, ft_steps: list[int], seed: int) -> dict:
    out = {}
    z = eval_tx_model(model, seed=seed, device=device)
    out["zero_loss"], out["zero_acc"] = z["eval_loss"], z["eval_acc"]
    for n in ft_steps:
        if n <= 0: continue
        mc = VariableTinyTransformer(model.spec)
        mc.load_state_dict(model.state_dict())
        mc.to(device)
        finetune_tx(mc, n, seed=seed, device=device)
        a = eval_tx_model(mc, seed=seed, device=device)
        out[f"ft{n}_loss"], out[f"ft{n}_acc"] = a["eval_loss"], a["eval_acc"]
    return out
