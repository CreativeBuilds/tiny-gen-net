#!/usr/bin/env python3
"""Phase 0: batch baseline training across seeds + visualization."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.device import device_str, get_device
from src.utils.train_loop import TrainConfig, train_tiny_mlp
from src.utils.viz import plot_multi_loss_curves


def main():
    p = argparse.ArgumentParser(description="Phase 0 baseline training")
    p.add_argument("--num_seeds", type=int, default=10)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--plot_dir", type=str, default="checkpoints/plots")
    args = p.parse_args()

    device = device_str()
    curves: dict[str, list[float]] = {}
    summary: list[dict] = []

    print(f"Phase 0 | training {args.num_seeds} baselines on {device}")
    for i in range(args.num_seeds):
        seed = args.seed_start + i
        cfg = TrainConfig(steps=args.steps, hidden_dim=args.hidden, seed=seed, device=device)
        _, losses, metrics = train_tiny_mlp(cfg)
        curves[f"seed{seed}"] = losses
        summary.append(metrics)
        print(f"  seed={seed} final_loss={metrics['final_train_loss']:.4f} eval_acc={metrics['eval_acc']:.3f}")

    plot_dir = Path(args.plot_dir)
    plot_multi_loss_curves(curves, out_path=plot_dir / "phase0_baselines.png", title="Phase 0 — Baseline Loss Curves")

    avg_loss = sum(m["final_train_loss"] for m in summary) / len(summary)
    avg_acc = sum(m["eval_acc"] for m in summary) / len(summary)
    metrics = {"avg_final_loss": avg_loss, "avg_eval_acc": avg_acc, "num_seeds": args.num_seeds}
    log_run("checkpoints/logs", "phase0_baselines", metrics, vars(args))
    print(f"\nPhase 0 complete | avg_loss={avg_loss:.4f} avg_acc={avg_acc:.3f}")
    print(f"Plot -> {plot_dir / 'phase0_baselines.png'}")


if __name__ == "__main__":
    main()
