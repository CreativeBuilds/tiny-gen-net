#!/usr/bin/env python3
"""Collect N VariableTinyMLP training trajectories with checkpoints.

Each trajectory = one model trained from its own seed, with checkpoints saved
at regular intervals. All checkpoints within a trajectory share the same
permutation (neurons don't permute during training), so alignment needs only
one weight-match per trajectory.

Output: trajectories.pt (N, T, D) + meta.json
"""

import argparse
import json
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
from src.utils.weights import flatten_state_dict, param_slices, weight_dim


def train_with_checkpoints(spec, steps, ckpt_interval, lr, batch_size, seed, device):
    """Train one model, return stack of checkpoint weight vectors (T, D)."""
    set_seed(seed)
    corpus = generate_corpus(SyntheticConfig(seed=seed))
    pairs = list(corpus_to_tensor_pairs(corpus))
    if not pairs:
        raise RuntimeError("Empty training pairs from corpus")

    model = VariableTinyMLP(spec, vocab_size=VOCAB_SIZE).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    checkpoints: list[torch.Tensor] = []
    # Save initial weights (step 0)
    checkpoints.append(flatten_state_dict(model).cpu())

    model.train()
    for step in range(steps):
        xs, ys = build_batch(pairs, batch_size, device)
        logits, _ = model(xs)
        loss = crit(logits, ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % ckpt_interval == 0:
            checkpoints.append(flatten_state_dict(model).cpu())

    return torch.stack(checkpoints)


def main():
    p = argparse.ArgumentParser(description="Collect VariableTinyMLP training trajectories")
    p.add_argument("--num_models", type=int, default=100)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--ckpt_interval", type=int, default=100)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--out", type=str, default="checkpoints/trajectories")
    args = p.parse_args()

    device = device_str()
    spec = ArchSpec(args.hidden, args.depth, skip=False)
    probe = VariableTinyMLP(spec, vocab_size=VOCAB_SIZE)
    dim, slices = weight_dim(probe), [(s["start"], s["end"]) for s in param_slices(probe)]

    T = args.steps // args.ckpt_interval + 1  # +1 for step 0
    print(f"Collecting {args.num_models} trajectories × {T} checkpoints | dim={dim} | depth={args.depth} | H={args.hidden}")
    print(f"  steps={args.steps} ckpt_interval={args.ckpt_interval}")

    all_trajs: list[torch.Tensor] = []
    for i in range(args.num_models):
        seed = args.seed_start + i
        ckpts = train_with_checkpoints(spec, args.steps, args.ckpt_interval, 1e-3, 64, seed, device)
        all_trajs.append(ckpts)
        if (i + 1) % 10 == 0 or i == 0:
            # Quick eval of last checkpoint
            m = VariableTinyMLP(spec, vocab_size=VOCAB_SIZE)
            from src.utils.weights import load_flat_into_model
            load_flat_into_model(ckpts[-1], m)
            from src.utils.train_loop import eval_model_loss
            el = eval_model_loss(m, seed=seed, device="cpu")
            print(f"  [{i+1}/{args.num_models}] seed={seed} done ({ckpts.shape}) final_eval_loss={el['eval_loss']:.4f}")

    trajs = torch.stack(all_trajs)  # (N, T, D)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(trajs, out_dir / "trajectories.pt")

    meta = {
        "num_models": args.num_models,
        "steps": args.steps,
        "ckpt_interval": args.ckpt_interval,
        "num_ckpts": T,
        "hidden_dim": args.hidden,
        "depth": args.depth,
        "weight_dim": dim,
        "layer_slices": [list(x) for x in slices],
        "seed_start": args.seed_start,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved {tuple(trajs.shape)} -> {out_dir}/trajectories.pt")


if __name__ == "__main__":
    main()
