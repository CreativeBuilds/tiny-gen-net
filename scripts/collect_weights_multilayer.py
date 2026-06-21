#!/usr/bin/env python3
"""Train N VariableTinyMLP (depth=3) models; save raw weight vectors for alignment ablation."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from data.synthetic_text import SyntheticConfig, corpus_to_tensor_pairs, generate_corpus, VOCAB_SIZE
# Import arch_gen before variable_mlp to break circular import
from src.arch_gen.spec import ArchSpec  # noqa: F401 — forces package init
from src.models.variable_mlp import VariableTinyMLP
from src.utils.device import device_str
from src.utils.train_loop import build_batch, set_seed
from src.utils.weights import flatten_state_dict, normalize_weights, param_slices, save_weight_collection, weight_dim


def train_one_variable_mlp(spec: ArchSpec, steps: int, lr: float, batch_size: int, seed: int, device: str):
    """Train one VariableTinyMLP; returns model, final metrics."""
    set_seed(seed)
    corpus = generate_corpus(SyntheticConfig(seed=seed))
    pairs = list(corpus_to_tensor_pairs(corpus))
    if not pairs:
        raise RuntimeError("Empty training pairs from corpus")

    model = VariableTinyMLP(spec, vocab_size=VOCAB_SIZE).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    model.train()
    last_loss = 0.0
    for step in range(steps):
        xs, ys = build_batch(pairs, batch_size, device)
        logits, _ = model(xs)
        loss = crit(logits, ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
        last_loss = loss.item()

    model.eval()
    with torch.no_grad():
        eval_losses, correct, total = [], 0, 0
        for _ in range(min(20, len(pairs) // batch_size + 1)):
            xs, ys = build_batch(pairs, batch_size, device)
            logits, _ = model(xs)
            eval_losses.append(crit(logits, ys).item())
            correct += (logits.argmax(-1) == ys).sum().item()
            total += ys.numel()

    metrics = {
        "final_train_loss": last_loss,
        "eval_loss": sum(eval_losses) / max(len(eval_losses), 1),
        "eval_acc": correct / max(total, 1),
        "num_params": model.num_parameters(),
        "steps": steps,
        "seed": seed,
    }
    return model, metrics


def main():
    p = argparse.ArgumentParser(description="Collect VariableTinyMLP weight dataset")
    p.add_argument("--num_models", type=int, default=100)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--out", type=str, default="checkpoints/weights/align_pool_ml")
    p.add_argument("--norm", type=str, default="perdim", choices=["global", "layer", "perdim"])
    args = p.parse_args()

    device = device_str()
    spec = ArchSpec(args.hidden, args.depth, skip=False)
    probe = VariableTinyMLP(spec, vocab_size=VOCAB_SIZE)
    dim, slices = weight_dim(probe), [(s["start"], s["end"]) for s in param_slices(probe)]
    rows: list[torch.Tensor] = []
    all_metrics: list[dict] = []

    print(f"Collecting {args.num_models} VariableTinyMLP(depth={args.depth}, H={args.hidden}) | weight dim={dim} | norm={args.norm}")
    for i in range(args.num_models):
        seed = args.seed_start + i
        model, metrics = train_one_variable_mlp(spec, args.steps, 1e-3, 64, seed, device)
        rows.append(flatten_state_dict(model))
        all_metrics.append(metrics)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{args.num_models}] seed={seed} loss={metrics['eval_loss']:.4f} acc={metrics['eval_acc']:.3f}")

    raw = torch.stack(rows)
    norm_w, norm_meta = normalize_weights(raw, mode=args.norm, layer_slices=slices)
    meta = {
        "num_models": args.num_models,
        "weight_dim": dim,
        "hidden_dim": args.hidden,
        "depth": args.depth,
        "train_steps": args.steps,
        "layer_slices": [list(x) for x in slices],
        "raw_mean": raw.mean().item(),
        "raw_std": raw.std().item(),
        "metrics_summary": {
            "avg_eval_loss": sum(m["eval_loss"] for m in all_metrics) / len(all_metrics),
            "avg_eval_acc": sum(m["eval_acc"] for m in all_metrics) / len(all_metrics),
        },
        **norm_meta,
    }
    save_weight_collection(raw, {k: v for k, v in meta.items() if k not in ("mean", "std", "norm_mode")}, args.out, filename="weights_raw.pt")
    save_weight_collection(norm_w, meta, args.out)
    print(f"Saved raw {raw.shape} -> {args.out}/weights_raw.pt")
    print(f"Saved norm ({args.norm}) {norm_w.shape} -> {args.out}/weights.pt")


if __name__ == "__main__":
    main()
