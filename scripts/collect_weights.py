#!/usr/bin/env python3
"""Train N TinyMLPs; save raw + normalized weight matrices for diffusion."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.models.tiny_mlp import TinyMLP
from src.utils.device import device_str
from src.utils.train_loop import TrainConfig, train_tiny_mlp
from src.utils.weights import flatten_state_dict, normalize_weights, param_slices, save_weight_collection, weight_dim


def main():
    p = argparse.ArgumentParser(description="Collect weight dataset from trained TinyMLPs")
    p.add_argument("--num_models", type=int, default=50)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--out", type=str, default="checkpoints/weights/phase0")
    p.add_argument("--norm", type=str, default="perdim", choices=["global", "layer", "perdim"], help="Normalization for weights.pt")
    args = p.parse_args()

    device = device_str()
    probe = TinyMLP(hidden_dim=args.hidden)
    dim, slices = weight_dim(probe), [(s["start"], s["end"]) for s in param_slices(probe)]
    rows: list[torch.Tensor] = []
    all_metrics: list[dict] = []

    print(f"Collecting {args.num_models} models | weight dim={dim} | norm={args.norm}")
    for i in range(args.num_models):
        seed = args.seed_start + i
        cfg = TrainConfig(steps=args.steps, hidden_dim=args.hidden, seed=seed, device=device)
        model, _, metrics = train_tiny_mlp(cfg)
        rows.append(flatten_state_dict(model))
        all_metrics.append(metrics)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{args.num_models}] seed={seed} loss={metrics['final_train_loss']:.4f} acc={metrics['eval_acc']:.3f}")

    raw = torch.stack(rows)
    norm_w, norm_meta = normalize_weights(raw, mode=args.norm, layer_slices=slices)
    meta = {
        "num_models": args.num_models,
        "weight_dim": dim,
        "hidden_dim": args.hidden,
        "train_steps": args.steps,
        "layer_slices": [list(x) for x in slices],
        "raw_mean": raw.mean().item(),
        "raw_std": raw.std().item(),
        "metrics_summary": {
            "avg_final_loss": sum(m["final_train_loss"] for m in all_metrics) / len(all_metrics),
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
