"""Eval NanoTransformer on Tiny Shakespeare val set."""

import random

import torch

from data.shakespeare import get_batch, load_shakespeare
from src.arch_gen.nano_spec import NanoSpec
from src.models.nano_transformer import NanoTransformer, build_from_nano_spec
from src.utils.train_loop import set_seed

_train_cache: dict = {}


def _get_data(seed: int):
    if seed not in _train_cache:
        train, val, tok = load_shakespeare(seed=seed)
        _train_cache[seed] = (train, val, tok)
    return _train_cache[seed]


def eval_nano_model(model: NanoTransformer, seed: int = 0, device: str = "cpu", batches: int = 20, batch_size: int = 32) -> dict:
    set_seed(seed + 999)
    _, val, tok = _get_data(seed)
    model.eval()
    losses, correct, total = [], 0, 0
    rng = random.Random(seed + 999)
    with torch.no_grad():
        for _ in range(batches):
            x, y = get_batch(val, tok, batch_size, model.block_size, device, rng)
            _, loss = model(x, y)
            losses.append(loss.item())
            logits, _ = model(x)
            correct += (logits.argmax(-1) == y).sum().item()
            total += y.numel()
    return {"eval_loss": sum(losses) / len(losses), "eval_acc": correct / max(total, 1)}


def finetune_nano(model: NanoTransformer, steps: int, seed: int = 0, lr: float = 3e-4, device: str = "cpu", batch_size: int = 32) -> list[float]:
    set_seed(seed + 1234)
    train, _, tok = _get_data(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses: list[float] = []
    rng = random.Random(seed + 1234)
    model.train()
    for _ in range(steps):
        x, y = get_batch(train, tok, batch_size, model.block_size, device, rng)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return losses


def eval_nano_row(model: NanoTransformer, device: str, ft_steps: list[int], seed: int) -> dict:
    out = {}
    z = eval_nano_model(model, seed=seed, device=device)
    out["zero_loss"], out["zero_acc"] = z["eval_loss"], z["eval_acc"]
    for n in ft_steps:
        if n <= 0: continue
        mc = build_from_nano_spec(model.spec, model.vocab_size)
        mc.load_state_dict(model.state_dict())
        mc.to(device)
        finetune_nano(mc, n, seed=seed, device=device)
        a = eval_nano_model(mc, seed=seed, device=device)
        out[f"ft{n}_loss"], out[f"ft{n}_acc"] = a["eval_loss"], a["eval_acc"]
    return out
