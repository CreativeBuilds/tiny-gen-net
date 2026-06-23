#!/usr/bin/env python3
"""GATE-GROW (width): Does width growth + correction beat random and naive growth?

Core claim: a grown + corrected init reaches a target loss with materially
fewer FLOPs than baselines, on a task where the target width has genuinely
more capacity than the source.

Setup:
  Source: VariableTinyMLP(depth=2, H=16) trained to convergence (2000 steps)
  Target: VariableTinyMLP(depth=2, H=64) grown from source (width growth)

Task: 16-char alphabet, 20 patterns, next-char prediction. This creates a
genuine width capacity gap (H=16: CE~1.59, H=64: CE~1.44 at convergence).

Arms (all fine-tuned for the SAME number of steps = equal FLOP budget):
  1. random:          random init target (H=64), train from scratch
  2. naive_grow:       function-preserving width growth (zero-padded), train
  3. grown_corr:       growth + small low-rank correction on new units, train
  4. grown_rand_corr: growth + larger random correction, train
  5. source_only:      source model (H=16) trained at equal FLOP budget

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
from src.growth.width_growth import (
    build_model,
    grow_width,
    add_width_correction,
    verify_growth_preserves_function,
)
from src.utils.train_loop import set_seed, eval_model_loss, build_batch
from src.utils.device import device_str, get_device
from data import synthetic_text as st
from data.synthetic_text import SyntheticConfig, corpus_to_tensor_pairs, generate_corpus


# --- Expanded task config: 16 chars, 20 patterns ---
EXPANDED_CHARS = "abcdefghijklmnop"
EXPANDED_PATTERNS = [
    "abcd", "abdc", "acbd", "badc", "bcad", "bdac", "cabd", "cbad", "cdab",
    "abcdef", "acedbf", "aabbcc", "abcabc", "ababcd", "aabbccdd",
    "acebdf", "aebcdf", "abcedf", "afbecd", "abcdefabcdef",
]
EXPANDED_VOCAB = len(EXPANDED_CHARS)


def setup_expanded_task():
    """Override the synthetic text module globals for the expanded task."""
    st.CHARS = EXPANDED_CHARS
    st.CHAR_TO_IDX = {c: i for i, c in enumerate(EXPANDED_CHARS)}
    st.IDX_TO_CHAR = {i: c for i, c in enumerate(EXPANDED_CHARS)}
    st.VOCAB_SIZE = EXPANDED_VOCAB
    st.BASE_PATTERNS = EXPANDED_PATTERNS


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


def run_grow_gate(
    *,
    source_hidden=16,
    target_hidden=64,
    depth=2,
    source_steps=2000,
    finetune_steps=500,
    rank=8,
    corr_scale=0.05,
    rand_corr_scale=0.1,
    seeds=(0, 1, 2, 3, 4),
    device="cpu",
):
    """Run the full width-growth GATE-GROW experiment."""
    arms = ["random", "naive_grow", "grown_corr", "grown_rand_corr", "source_only"]
    results = {a: [] for a in arms}
    growth_diffs = []

    print(f"GATE-GROW (width) | source H={source_hidden} -> target H={target_hidden} | depth={depth}")
    print(f"Source training: {source_steps} steps | Fine-tune: {finetune_steps} steps | rank={rank}")
    print(f"corr_scale={corr_scale} rand_corr_scale={rand_corr_scale}")
    print(f"Seeds: {seeds} | device={device} | vocab={EXPANDED_VOCAB}")
    print()

    for si, seed in enumerate(seeds):
        print(f"--- seed {seed} ({si+1}/{len(seeds)}) ---")

        # 1. Train source model (H=source_hidden)
        source_model = build_model(source_hidden, depth, vocab=EXPANDED_VOCAB).to(device)
        source_ce = train_variable_mlp(source_model, source_steps, 1e-3, seed, device)
        print(f"  source trained: CE={source_ce:.4f} (H={source_hidden}, {sum(p.numel() for p in source_model.parameters())} params)")

        # 2. Grow to target width (function-preserving)
        grown = grow_width(source_model, target_hidden)
        fn_diff = verify_growth_preserves_function(source_model, grown)
        growth_diffs.append(fn_diff)
        print(f"  growth fn-preservation: max diff={fn_diff:.2e}")

        # 3. Create arms
        # random: fresh random init at target width
        random_model = build_model(target_hidden, depth, vocab=EXPANDED_VOCAB).to(device)

        # naive_grow: the grown model (no correction) — clone so we don't share state
        naive_model = grow_width(source_model, target_hidden).to(device)

        # grown_corr: growth + small low-rank correction on new units
        corr_model = grow_width(source_model, target_hidden)
        add_width_correction(corr_model, source_hidden, rank=rank, init_scale=corr_scale)
        corr_model = corr_model.to(device)

        # grown_rand_corr: growth + larger random correction
        rand_corr_model = grow_width(source_model, target_hidden)
        add_width_correction(rand_corr_model, source_hidden, rank=rank, init_scale=rand_corr_scale)
        rand_corr_model = rand_corr_model.to(device)

        # source_only: source model given equal total FLOPs (source_steps + finetune_steps)
        source_only_model = build_model(source_hidden, depth, vocab=EXPANDED_VOCAB)
        source_only_model.load_state_dict(source_model.state_dict())
        source_only_model = source_only_model.to(device)

        # 4. Evaluate zero-shot (before fine-tune)
        zero_ces = {
            "random": eval_model_loss(random_model, seed=seed + 999, device=device)["eval_loss"],
            "naive_grow": eval_model_loss(naive_model, seed=seed + 999, device=device)["eval_loss"],
            "grown_corr": eval_model_loss(corr_model, seed=seed + 999, device=device)["eval_loss"],
            "grown_rand_corr": eval_model_loss(rand_corr_model, seed=seed + 999, device=device)["eval_loss"],
            "source_only": eval_model_loss(source_only_model, seed=seed + 999, device=device)["eval_loss"],
        }
        print(f"  zero-shot CE: " + " ".join(f"{k}={v:.4f}" for k, v in zero_ces.items()))

        # 5. Fine-tune all arms for finetune_steps (equal FLOP budget)
        ft_ces = {}
        for arm, model in [("random", random_model), ("naive_grow", naive_model),
                           ("grown_corr", corr_model), ("grown_rand_corr", rand_corr_model),
                           ("source_only", source_only_model)]:
            ce = train_variable_mlp(model, finetune_steps, 1e-3, seed + 100, device)
            ft_ces[arm] = ce
            results[arm].append(ce)
        print(f"  post-FT CE: " + " ".join(f"{k}={v:.4f}" for k, v in ft_ces.items()))
        print()

    # --- Summary ---
    summary = {
        "source_hidden": source_hidden, "target_hidden": target_hidden,
        "depth": depth, "source_steps": source_steps,
        "finetune_steps": finetune_steps, "rank": rank,
        "corr_scale": corr_scale, "rand_corr_scale": rand_corr_scale,
        "n_seeds": len(seeds), "seeds": list(seeds),
        "vocab_size": EXPANDED_VOCAB,
        "growth_fn_preservation_max": max(growth_diffs),
    }
    for arm in arms:
        vals = results[arm]
        m = statistics.fmean(vals)
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        summary[arm] = {"ce_mean": m, "ce_std": sd, "n": len(vals)}

    rand_mean = summary["random"]["ce_mean"]
    for arm in arms:
        summary[arm]["delta_vs_random"] = rand_mean - summary[arm]["ce_mean"]

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
    p = argparse.ArgumentParser(description="GATE-GROW (width): width growth + correction vs baselines")
    p.add_argument("--source_hidden", type=int, default=16)
    p.add_argument("--target_hidden", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--source_steps", type=int, default=2000)
    p.add_argument("--finetune_steps", type=int, default=500)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--corr_scale", type=float, default=0.05)
    p.add_argument("--rand_corr_scale", type=float, default=0.1)
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--out", type=str, default="checkpoints/align/grow_gate_width")
    args = p.parse_args()

    setup_expanded_task()

    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    device = device_str(get_device())

    summary = run_grow_gate(
        source_hidden=args.source_hidden,
        target_hidden=args.target_hidden,
        depth=args.depth,
        source_steps=args.source_steps,
        finetune_steps=args.finetune_steps,
        rank=args.rank,
        corr_scale=args.corr_scale,
        rand_corr_scale=args.rand_corr_scale,
        seeds=seeds,
        device=device,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "grow_gate_width_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print("\n=== GATE-GROW (width) SUMMARY ===")
    print(f"Source: H={summary['source_hidden']} -> Target: H={summary['target_hidden']} | depth={summary['depth']}")
    print(f"Source training: {summary['source_steps']} steps | Fine-tune: {summary['finetune_steps']} steps | rank={summary['rank']}")
    print(f"Seeds: {summary['n_seeds']} | Growth fn-preservation max diff: {summary['growth_fn_preservation_max']:.2e}")
    print()
    for arm in ["random", "naive_grow", "grown_corr", "grown_rand_corr", "source_only"]:
        s = summary[arm]
        print(f"  {arm:20s}  CE={s['ce_mean']:.4f} +/- {s['ce_std']:.4f}  delta_vs_random={s['delta_vs_random']:+.4f}")
    print()
    print(f"Best arm: {summary['best_arm']}  delta_vs_random={summary['best_delta_vs_random']:+.4f}")
    print(f"Growth beats random: {summary['growth_beats_random']}")
    print(f"Correction beats naive: {summary['correction_beats_naive']}")
    print(f"Correction beats random: {summary['correction_beats_random']}")
    print(f"Growth beats source_only: {summary['growth_beats_source_only']}")
    print(f"GATE PASSES: {summary['gate_passes']}")
    print(f"\nArtifacts: {out_dir}/grow_gate_width_summary.json")


if __name__ == "__main__":
    main()
