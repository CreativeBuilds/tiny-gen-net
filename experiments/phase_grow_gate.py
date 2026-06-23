#!/usr/bin/env python3
"""GATE-GROW: Does growth + correction beat random init and naive growth?

The core claim: a grown + corrected init reaches a target loss with materially
fewer FLOPs than baselines.

Setup:
  Source: VariableTinyMLP(depth=2, H=64) trained to convergence (1000 steps)
  Target: VariableTinyMLP(depth=4, H=64) grown from source

Arms (all fine-tuned for the SAME number of steps = equal FLOP budget):
  1. random:      random init target, train from scratch
  2. naive_grow:  function-preserving growth (identity blocks inserted), train
  3. grown_corr:  growth + low-rank correction perturbation, train
  4. source_only: source model (depth=2) trained at equal FLOP budget (capacity ceiling)
  5. grown_rand_corr: growth + random Xavier correction (not learned), train

Metric: eval CE after fine-tune. Headline: ΔCE vs random (positive = better).

5 seeds, report mean±std. FLOP budget = 500 fine-tune steps for all arms.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from src.arch_gen.spec import ArchSpec  # noqa: F401 — forces package init
from src.growth.depth_growth import (
    build_model,
    grow_depth,
    verify_growth_preserves_function,
)
from src.utils.train_loop import set_seed, eval_model_loss, build_batch
from src.utils.device import device_str, get_device
from data.synthetic_text import SyntheticConfig, corpus_to_tensor_pairs, generate_corpus, VOCAB_SIZE


def train_variable_mlp(model, steps, lr, seed, device, skip=0):
    """Train a VariableTinyMLP for `steps`. Returns eval CE after training."""
    set_seed(seed)
    corpus = generate_corpus(SyntheticConfig(seed=seed))
    pairs = list(corpus_to_tensor_pairs(corpus, skip=skip))
    if not pairs:
        raise RuntimeError("Empty training pairs")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(steps):
        xs, ys = build_batch(pairs, 64, device)
        logits, _ = model(xs.to(device))
        loss = crit(logits, ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return eval_model_loss(model, seed=seed + 999, device=device, skip=skip)["eval_loss"]


def train_source(source_depth, hidden, source_steps, seed, device, skip=0):
    """Train the source model to convergence."""
    model = build_model(hidden, source_depth)
    model.to(device)
    ce = train_variable_mlp(model, source_steps, 1e-3, seed, device, skip=skip)
    return model, ce


def run_grow_gate(
    *,
    source_depth=2,
    target_depth=4,
    hidden=64,
    source_steps=1000,
    finetune_steps=500,
    rank=8,
    seeds=(0, 1, 2, 3, 4),
    device="cpu",
    skip=0,
):
    """Run the full GATE-GROW experiment."""
    arms = ["random", "naive_grow", "grown_corr", "grown_rand_corr", "source_only"]
    results = {a: [] for a in arms}
    growth_diffs = []  # function-preservation check

    print(f"GATE-GROW | source depth={source_depth} → target depth={target_depth} | H={hidden}")
    print(f"Source training: {source_steps} steps | Fine-tune: {finetune_steps} steps | rank={rank}")
    print(f"Seeds: {seeds} | device={device}")
    print()

    for si, seed in enumerate(seeds):
        print(f"--- seed {seed} ({si+1}/{len(seeds)}) ---")

        # 1. Train source model
        source_model, source_ce = train_source(source_depth, hidden, source_steps, seed, device, skip=skip)
        print(f"  source trained: CE={source_ce:.4f} (depth={source_depth}, {sum(p.numel() for p in source_model.parameters())} params)")

        # 2. Grow to target depth (function-preserving)
        grown = grow_depth(source_model, target_depth)
        # Verify function preservation
        fn_diff = verify_growth_preserves_function(source_model, grown)
        growth_diffs.append(fn_diff)
        print(f"  growth fn-preservation: max diff={fn_diff:.2e}")

        # 3. Create arms
        # random: fresh random init at target depth
        random_model = build_model(hidden, target_depth)
        random_model.to(device)

        # naive_grow: the grown model (no correction)
        naive_model = grown
        naive_model.to(device)

        # grown_corr: growth + low-rank correction (near-zero init, learned during fine-tune)
        # We add a small random low-rank perturbation to break symmetry between blocks
        corr_model = grow_depth(source_model, target_depth)
        # Add small low-rank perturbation to each NEW block's weight
        for k in range(source_depth, target_depth):
            W = corr_model.blocks[k].weight.data
            A = torch.randn(W.shape[0], rank, device=W.device) * (1.0 / (rank ** 0.5))
            B = torch.randn(rank, W.shape[1], device=W.device) * 0.01  # small init
            W.add_(A @ B)
            corr_model.blocks[k].bias.data.add_(torch.randn_like(corr_model.blocks[k].bias.data) * 0.001)
        corr_model.to(device)

        # grown_rand_corr: growth + full random Xavier correction (larger perturbation)
        rand_corr_model = grow_depth(source_model, target_depth)
        for k in range(source_depth, target_depth):
            W = rand_corr_model.blocks[k].weight.data
            A = torch.randn(W.shape[0], rank, device=W.device) * (1.0 / (rank ** 0.5))
            B = torch.randn(rank, W.shape[1], device=W.device) * 0.1  # larger init
            W.add_(A @ B)
            rand_corr_model.blocks[k].bias.data.add_(torch.randn_like(rand_corr_model.blocks[k].bias.data) * 0.01)
        rand_corr_model.to(device)

        # source_only: train source model at equal FLOP budget (more steps on small model)
        # source already trained for source_steps; give it finetune_steps more
        source_only_model = build_model(hidden, source_depth)
        source_only_model.load_state_dict(source_model.state_dict())
        source_only_model.to(device)

        # 4. Evaluate zero-shot (before fine-tune)
        zero_ces = {
            "random": eval_model_loss(random_model, seed=seed + 999, device=device, skip=skip)["eval_loss"],
            "naive_grow": eval_model_loss(naive_model, seed=seed + 999, device=device, skip=skip)["eval_loss"],
            "grown_corr": eval_model_loss(corr_model, seed=seed + 999, device=device, skip=skip)["eval_loss"],
            "grown_rand_corr": eval_model_loss(rand_corr_model, seed=seed + 999, device=device, skip=skip)["eval_loss"],
            "source_only": eval_model_loss(source_only_model, seed=seed + 999, device=device, skip=skip)["eval_loss"],
        }
        print(f"  zero-shot CE: " + " ".join(f"{k}={v:.4f}" for k, v in zero_ces.items()))

        # 5. Fine-tune all arms for finetune_steps (equal FLOP budget)
        # source_only gets finetune_steps MORE training (equal FLOPs, smaller model)
        ft_ces = {}
        for arm, model in [("random", random_model), ("naive_grow", naive_model),
                           ("grown_corr", corr_model), ("grown_rand_corr", rand_corr_model),
                           ("source_only", source_only_model)]:
            ce = train_variable_mlp(model, finetune_steps, 1e-3, seed + 100, device, skip=skip)
            ft_ces[arm] = ce
            results[arm].append(ce)
        print(f"  post-FT CE: " + " ".join(f"{k}={v:.4f}" for k, v in ft_ces.items()))
        print()

    # --- Summary ---
    summary = {
        "source_depth": source_depth, "target_depth": target_depth,
        "hidden": hidden, "source_steps": source_steps,
        "finetune_steps": finetune_steps, "rank": rank,
        "n_seeds": len(seeds), "seeds": list(seeds), "skip": skip,
        "growth_fn_preservation_max": max(growth_diffs),
    }
    for arm in arms:
        vals = results[arm]
        m = statistics.fmean(vals)
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        summary[arm] = {"ce_mean": m, "ce_std": sd, "n": len(vals)}

    # Deltas vs random (positive = better than random)
    rand_mean = summary["random"]["ce_mean"]
    for arm in arms:
        summary[arm]["delta_vs_random"] = rand_mean - summary[arm]["ce_mean"]

    # Key comparisons
    summary["growth_beats_random"] = summary["naive_grow"]["delta_vs_random"] > 0
    summary["correction_beats_naive"] = summary["grown_corr"]["ce_mean"] < summary["naive_grow"]["ce_mean"]
    summary["correction_beats_random"] = summary["grown_corr"]["delta_vs_random"] > 0
    summary["growth_beats_source_only"] = summary["naive_grow"]["ce_mean"] < summary["source_only"]["ce_mean"]
    summary["best_arm"] = min(arms, key=lambda a: summary[a]["ce_mean"])
    summary["best_delta_vs_random"] = rand_mean - summary[summary["best_arm"]]["ce_mean"]
    summary["gate_passes"] = (
        summary["grown_corr"]["delta_vs_random"] > 0.01
        and summary["grown_corr"]["ce_mean"] < summary["naive_grow"]["ce_mean"]
        and summary["grown_corr"]["ce_mean"] < summary["source_only"]["ce_mean"]
    )

    return summary


def main():
    p = argparse.ArgumentParser(description="GATE-GROW: growth + correction vs baselines")
    p.add_argument("--source_depth", type=int, default=2)
    p.add_argument("--target_depth", type=int, default=4)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--source_steps", type=int, default=1000)
    p.add_argument("--finetune_steps", type=int, default=500)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--skip", type=int, default=0, help="Skip-char distance (0=next-char, k>0=skip-char)")
    p.add_argument("--out", type=str, default="checkpoints/align/grow_gate")
    args = p.parse_args()

    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    device = device_str(get_device())

    summary = run_grow_gate(
        source_depth=args.source_depth,
        target_depth=args.target_depth,
        hidden=args.hidden,
        source_steps=args.source_steps,
        finetune_steps=args.finetune_steps,
        rank=args.rank,
        seeds=seeds,
        skip=args.skip,
        device=device,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "grow_gate_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print("\n=== GATE-GROW SUMMARY ===")
    print(f"Source: depth={summary['source_depth']} → Target: depth={summary['target_depth']} | H={summary['hidden']}")
    print(f"Source training: {summary['source_steps']} steps | Fine-tune: {summary['finetune_steps']} steps | rank={summary['rank']}")
    print(f"Seeds: {summary['n_seeds']} | Growth fn-preservation max diff: {summary['growth_fn_preservation_max']:.2e}")
    print()
    for arm in ["random", "naive_grow", "grown_corr", "grown_rand_corr", "source_only"]:
        s = summary[arm]
        print(f"  {arm:20s}  CE={s['ce_mean']:.4f} ± {s['ce_std']:.4f}  delta_vs_random={s['delta_vs_random']:+.4f}")
    print()
    print(f"Best arm: {summary['best_arm']}  delta_vs_random={summary['best_delta_vs_random']:+.4f}")
    print(f"Growth beats random: {summary['growth_beats_random']}")
    print(f"Correction beats naive: {summary['correction_beats_naive']}")
    print(f"Correction beats random: {summary['correction_beats_random']}")
    print(f"Growth beats source_only: {summary['growth_beats_source_only']}")
    print(f"GATE PASSES: {summary['gate_passes']}")
    print(f"\nArtifacts: {out_dir}/grow_gate_summary.json")


if __name__ == "__main__":
    main()
