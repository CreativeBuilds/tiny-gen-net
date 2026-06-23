#!/usr/bin/env python3
"""GATE-GROW (tx depth): Does transformer depth growth + correction beat baselines?

Core claim: inserting zero-init blocks (function-preserving via residual) plus
a low-rank correction on the new blocks reaches lower CE than random init or
naive growth at equal FLOP budget.

Setup:
  Source: VariableTinyTransformer(d_model=64, n_layer=2, n_head=4) ~50K params
  Target: VariableTinyTransformer(d_model=64, n_layer=4, n_head=4) ~80K params
  Task: 16-char alphabet, 20 patterns, next-char with context length 8

Arms (all fine-tuned for the SAME number of steps = equal FLOP budget):
  1. random:          random init target, train from scratch
  2. naive_grow:       zero-init block insertion (identity via residual), train
  3. grown_corr:       growth + small low-rank correction on new blocks, train
  4. grown_rand_corr: growth + larger random correction, train
  5. source_only:      source model trained at equal total FLOPs

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
from src.growth.tx_depth_growth import (
    build_tx_model,
    grow_tx_depth,
    add_tx_correction,
    verify_tx_growth_preserves_function,
)
from src.arch_gen.tx_eval import (
    eval_tx_model,
    finetune_tx,
    corpus_to_seq_pairs,
    build_seq_batch,
)
from src.utils.train_loop import set_seed
from src.utils.device import device_str, get_device
from data import synthetic_text as st
from data.synthetic_text import SyntheticConfig, generate_corpus


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


def train_tx_model(model, steps, lr, seed, device):
    """Train a VariableTinyTransformer for `steps`. Returns eval CE after training."""
    set_seed(seed)
    corpus = generate_corpus(SyntheticConfig(seed=seed))
    pairs = list(corpus_to_seq_pairs(corpus, model.ctx_len))
    if not pairs:
        raise RuntimeError("Empty training pairs")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(steps):
        xs, ys = build_seq_batch(pairs, 64, device)
        logits, _ = model(xs.to(device))
        loss = crit(logits, ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return eval_tx_model(model, seed=seed + 999, device=device)["eval_loss"]


def run_grow_gate_tx(
    *,
    d_model=64,
    source_n_layer=2,
    target_n_layer=4,
    n_head=4,
    ctx_len=8,
    source_steps=2000,
    finetune_steps=500,
    rank=8,
    corr_scale=0.05,
    rand_corr_scale=0.1,
    seeds=(0, 1, 2, 3, 4),
    device="cpu",
):
    """Run the full transformer depth-growth GATE-GROW experiment."""
    arms = ["random", "naive_grow", "grown_corr", "grown_rand_corr", "source_only"]
    results = {a: [] for a in arms}
    growth_diffs = []

    print(f"GATE-GROW (tx depth) | source L={source_n_layer} -> target L={target_n_layer} | d={d_model} h={n_head}")
    print(f"Source training: {source_steps} steps | Fine-tune: {finetune_steps} steps | rank={rank}")
    print(f"corr_scale={corr_scale} rand_corr_scale={rand_corr_scale}")
    print(f"Seeds: {seeds} | device={device} | vocab={EXPANDED_VOCAB}")
    print()

    for si, seed in enumerate(seeds):
        print(f"--- seed {seed} ({si+1}/{len(seeds)}) ---")

        # 1. Train source model (n_layer=source_n_layer)
        source_model = build_tx_model(d_model, source_n_layer, n_head, ctx_len, vocab=EXPANDED_VOCAB).to(device)
        source_ce = train_tx_model(source_model, source_steps, 1e-3, seed, device)
        print(f"  source trained: CE={source_ce:.4f} (L={source_n_layer}, {sum(p.numel() for p in source_model.parameters())} params)")

        # 2. Grow to target depth (function-preserving)
        grown = grow_tx_depth(source_model, target_n_layer)
        fn_diff = verify_tx_growth_preserves_function(source_model, grown)
        growth_diffs.append(fn_diff)
        print(f"  growth fn-preservation: max diff={fn_diff:.2e}")

        # 3. Create arms
        # random: fresh random init at target depth
        random_model = build_tx_model(d_model, target_n_layer, n_head, ctx_len, vocab=EXPANDED_VOCAB).to(device)

        # naive_grow: grown model (no correction)
        naive_model = grow_tx_depth(source_model, target_n_layer).to(device)

        # grown_corr: growth + small low-rank correction on new blocks
        corr_model = grow_tx_depth(source_model, target_n_layer)
        add_tx_correction(corr_model, source_n_layer, rank=rank, init_scale=corr_scale)
        corr_model = corr_model.to(device)

        # grown_rand_corr: growth + larger random correction
        rand_corr_model = grow_tx_depth(source_model, target_n_layer)
        add_tx_correction(rand_corr_model, source_n_layer, rank=rank, init_scale=rand_corr_scale)
        rand_corr_model = rand_corr_model.to(device)

        # source_only: source model given equal total FLOPs
        source_only_model = build_tx_model(d_model, source_n_layer, n_head, ctx_len, vocab=EXPANDED_VOCAB)
        source_only_model.load_state_dict(source_model.state_dict())
        source_only_model = source_only_model.to(device)

        # 4. Evaluate zero-shot (before fine-tune)
        zero_ces = {
            "random": eval_tx_model(random_model, seed=seed + 999, device=device)["eval_loss"],
            "naive_grow": eval_tx_model(naive_model, seed=seed + 999, device=device)["eval_loss"],
            "grown_corr": eval_tx_model(corr_model, seed=seed + 999, device=device)["eval_loss"],
            "grown_rand_corr": eval_tx_model(rand_corr_model, seed=seed + 999, device=device)["eval_loss"],
            "source_only": eval_tx_model(source_only_model, seed=seed + 999, device=device)["eval_loss"],
        }
        print(f"  zero-shot CE: " + " ".join(f"{k}={v:.4f}" for k, v in zero_ces.items()))

        # 5. Fine-tune all arms for finetune_steps (equal FLOP budget)
        ft_ces = {}
        for arm, model in [("random", random_model), ("naive_grow", naive_model),
                           ("grown_corr", corr_model), ("grown_rand_corr", rand_corr_model),
                           ("source_only", source_only_model)]:
            ce = train_tx_model(model, finetune_steps, 1e-3, seed + 100, device)
            ft_ces[arm] = ce
            results[arm].append(ce)
        print(f"  post-FT CE: " + " ".join(f"{k}={v:.4f}" for k, v in ft_ces.items()))
        print()

    # --- Summary ---
    summary = {
        "d_model": d_model, "source_n_layer": source_n_layer, "target_n_layer": target_n_layer,
        "n_head": n_head, "ctx_len": ctx_len,
        "source_steps": source_steps, "finetune_steps": finetune_steps, "rank": rank,
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
    p = argparse.ArgumentParser(description="GATE-GROW (tx depth): transformer depth growth + correction")
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--source_n_layer", type=int, default=2)
    p.add_argument("--target_n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--ctx_len", type=int, default=8)
    p.add_argument("--source_steps", type=int, default=2000)
    p.add_argument("--finetune_steps", type=int, default=500)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--corr_scale", type=float, default=0.05)
    p.add_argument("--rand_corr_scale", type=float, default=0.1)
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--out", type=str, default="checkpoints/align/grow_gate_tx")
    args = p.parse_args()

    setup_expanded_task()

    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    device = device_str(get_device())

    summary = run_grow_gate_tx(
        d_model=args.d_model,
        source_n_layer=args.source_n_layer,
        target_n_layer=args.target_n_layer,
        n_head=args.n_head,
        ctx_len=args.ctx_len,
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
    (out_dir / "grow_gate_tx_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print("\n=== GATE-GROW (tx depth) SUMMARY ===")
    print(f"Source: L={summary['source_n_layer']} -> Target: L={summary['target_n_layer']} | d={summary['d_model']} h={summary['n_head']}")
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
    print(f"\nArtifacts: {out_dir}/grow_gate_tx_summary.json")


if __name__ == "__main__":
    main()
