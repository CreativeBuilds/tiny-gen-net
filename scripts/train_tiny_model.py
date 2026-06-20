#!/usr/bin/env python3
"""Train a single TinyMLP baseline and save checkpoint."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.utils.device import device_str
from src.utils.logging import log_run
from src.utils.train_loop import TrainConfig, train_tiny_mlp
from src.utils.viz import plot_loss_curve


def main():
    p = argparse.ArgumentParser(description="Train one TinyMLP")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--out", type=str, default="checkpoints/models")
    args = p.parse_args()

    device = device_str()
    cfg = TrainConfig(steps=args.steps, lr=args.lr, hidden_dim=args.hidden, seed=args.seed, device=device)
    model, losses, metrics = train_tiny_mlp(cfg)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"tiny_mlp_seed{args.seed}.pt"
    torch.save({"state_dict": model.state_dict(), "config": model.config_dict(), "metrics": metrics}, ckpt_path)
    plot_loss_curve(losses, out_path=Path("checkpoints/plots") / f"train_seed{args.seed}.png", title=f"TinyMLP seed={args.seed}")
    log_run("checkpoints/logs", f"train_seed{args.seed}", metrics, vars(args))
    print(f"Saved {ckpt_path} | final_loss={metrics['final_train_loss']:.4f} acc={metrics['eval_acc']:.3f}")


if __name__ == "__main__":
    main()
