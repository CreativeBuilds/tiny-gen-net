#!/usr/bin/env python3
"""Collect trained weights for every ArchSpec in the grid (multi-arch dataset)."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.arch_gen.spec import all_valid_specs
from src.models.variable_mlp import build_from_spec
from src.utils.device import device_str
from src.utils.weights import flatten_state_dict, param_slices


def train_variable_mlp(spec, steps: int, seed: int, device: str):
    """Train VariableTinyMLP via hidden_dim hack — reuse train loop on matching TinyMLP path."""
    from src.arch_gen.eval import finetune_arch
    from src.models.variable_mlp import VariableTinyMLP
    m = VariableTinyMLP(spec).to(device)
    finetune_arch(m, steps, seed=seed, device=device)
    return m, flatten_state_dict(m)


def main():
    p = argparse.ArgumentParser(description="Collect multi-arch weight dataset")
    p.add_argument("--num_per_spec", type=int, default=10)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seed_start", type=int, default=0)
    p.add_argument("--out", default="checkpoints/weights/multi_arch")
    args = p.parse_args()
    device = device_str()
    specs = all_valid_specs()
    rows, labels = [], []
    print(f"Collecting {args.num_per_spec} models × {len(specs)} archs | device={device}")
    for si, spec in enumerate(specs):
        for j in range(args.num_per_spec):
            seed = args.seed_start + si * 1000 + j
            _, flat = train_variable_mlp(spec, args.steps, seed, device)
            rows.append(flat)
            labels.append([spec.hidden_dim, spec.depth, int(spec.skip)])
            if (j + 1) % 5 == 0: print(f"  spec h={spec.hidden_dim} d={spec.depth} [{j+1}/{args.num_per_spec}]")
    probe = build_from_spec(specs[0])
    meta = {
        "num_models": len(rows), "num_per_spec": args.num_per_spec, "train_steps": args.steps,
        "specs": labels, "arch_grid": [[s.hidden_dim, s.depth] for s in specs],
        "layer_slices": [(s["start"], s["end"]) for s in param_slices(probe)],
    }
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.save({"weights": rows, "meta": meta}, out / "weights_raw.pt")
    (out / "meta.json").write_text(json.dumps({**meta, "weight_dims": [build_from_spec(s).num_parameters() for s in specs]}, indent=2))
    print(f"Saved {len(rows)} weight vectors -> {out}/weights_raw.pt")


if __name__ == "__main__":
    main()
