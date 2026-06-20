#!/usr/bin/env python3
"""Phase 1: train diffusion on collected weights; evaluate sampled vs random vs collected."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.logging import log_run
from src.utils.weights import NORM_MODES, load_raw_weights, load_weight_collection, normalize_weights, norm_meta_to_tensors
from experiments.phase1_lib import prepare_normalized_weights, run_phase1_experiment


def main():
    p = argparse.ArgumentParser(description="Phase 1 weight diffusion")
    p.add_argument("--weights", type=str, required=True, help="weights.pt or weights_raw.pt")
    p.add_argument("--norm", type=str, default=None, choices=list(NORM_MODES), help="Norm mode (required if --weights is raw)")
    p.add_argument("--diffusion_steps", type=int, default=2000)
    p.add_argument("--timesteps", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_samples", type=int, default=30)
    p.add_argument("--num_collected_eval", type=int, default=20)
    p.add_argument("--ft_steps", type=str, default="100,200")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot_dir", type=str, default="checkpoints/plots")
    p.add_argument("--tag", type=str, default="phase1_full")
    args = p.parse_args()
    ft_steps = [int(x.strip()) for x in args.ft_steps.split(",") if x.strip()]

    path = Path(args.weights)
    coll_meta = json.loads((path.parent / "meta.json").read_text()) if (path.parent / "meta.json").exists() else {}
    hidden = coll_meta.get("hidden_dim", 64)

    if path.name == "weights_raw.pt" or args.norm:
        if not args.norm: raise SystemExit("--norm required when using weights_raw.pt")
        slices = [tuple(x) for x in coll_meta.get("layer_slices", [])]
        if not slices: raise SystemExit("meta.json missing layer_slices; re-run collect_weights.py")
        weights_norm, norm_meta = prepare_normalized_weights(path, args.norm, slices)
    else:
        weights_norm, norm_meta = load_weight_collection(path)
        norm_meta = norm_meta_to_tensors(norm_meta)

    metrics = run_phase1_experiment(
        weights_norm, norm_meta, tag=args.tag, hidden=hidden,
        diffusion_steps=args.diffusion_steps, timesteps=args.timesteps, batch_size=args.batch_size, lr=args.lr,
        num_samples=args.num_samples, num_collected_eval=args.num_collected_eval, ft_steps=ft_steps,
        seed=args.seed, plot_dir=args.plot_dir,
    )
    ckpt_dir = Path("checkpoints/diffusion")
    out_json = ckpt_dir / f"metrics_{args.tag}.json"
    out_json.write_text(json.dumps(metrics, indent=2))
    log_run("checkpoints/logs", f"phase1_{args.tag}", metrics, vars(args))

    print(f"\n=== Phase 1 Summary [{args.tag}] norm={metrics.get('norm_mode')} ===")
    for k in sorted(metrics):
        v = metrics[k]
        if isinstance(v, float): print(f"  {k}: {v:.4f}")
    print(f"  zero_shot_delta (positive=gen better): {metrics['zero_shot_delta_vs_random']:.4f}")
    for n in ft_steps:
        if n <= 0: continue
        print(f"  ft{n}_delta (positive=gen better): {metrics[f'ft{n}_delta_vs_random']:.4f}")
    print(f"Artifacts: {args.plot_dir}/{args.tag}_*.png, {out_json}")


if __name__ == "__main__":
    main()
