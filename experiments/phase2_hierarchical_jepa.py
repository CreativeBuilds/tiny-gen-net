#!/usr/bin/env python3
"""Phase 2: train H-JEPA, compare vs flat JEPA, H-JEPA+diffusion hybrid, diffusion."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from tqdm import tqdm

from experiments.phase1_lib import avg_metric, eval_flat, prepare_normalized_weights
from experiments.phase2_jepa_vs_diffusion import load_diffusion_samples, metrics_from_rows
from src.hybrid.jepa_diffusion import JEPADiffusionHybrid, load_diffusion, load_hjepa, load_jepa
from src.jepa.hierarchical_jepa import HierarchicalWeightJEPA, HierarchicalWeightJEPATrainer
from src.models.tiny_mlp import TinyMLP
from src.utils.device import device_str, get_device
from src.utils.train_loop import set_seed
from src.utils.viz import plot_bar_comparison, plot_loss_curve, plot_weight_histograms
from src.utils.weights import denormalize_weights, flatten_state_dict, norm_meta_to_tensors


def main():
    p = argparse.ArgumentParser(description="Phase 2 hierarchical JEPA comparison")
    p.add_argument("--raw_weights", default="checkpoints/weights/phase0/weights_raw.pt")
    p.add_argument("--meta", default="checkpoints/weights/phase0/meta.json")
    p.add_argument("--flat_jepa_ckpt", default="checkpoints/jepa/weight_jepa_phase2_global.pt")
    p.add_argument("--diffusion_ckpt", default="checkpoints/diffusion/weight_denoiser_norm_global.pt")
    p.add_argument("--train_steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--refine_t", type=int, default=50)
    p.add_argument("--num_samples", type=int, default=30)
    p.add_argument("--ft_steps", default="100,200")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot_dir", default="checkpoints/plots/phase2_hjepa")
    p.add_argument("--tag", default="hjepa_global")
    p.add_argument("--hjepa_ckpt", default="", help="Skip training; load this H-JEPA checkpoint for eval-only")
    p.add_argument("--eval_only", action="store_true", help="Load --hjepa_ckpt (or default path) and run sampling+eval only")
    args = p.parse_args()
    ft_steps = [int(x) for x in args.ft_steps.split(",") if x.strip()]

    set_seed(args.seed)
    device = device_str()
    base_meta = json.loads(Path(args.meta).read_text())
    hidden, bounds = base_meta.get("hidden_dim", 64), [tuple(x) for x in base_meta["layer_slices"]]
    weights_norm, norm_meta = prepare_normalized_weights(args.raw_weights, "global", bounds)
    norm_meta = norm_meta_to_tensors(norm_meta)
    dim = weights_norm.shape[1]
    plot_dir = Path(args.plot_dir); plot_dir.mkdir(parents=True, exist_ok=True)
    bs = min(args.batch_size, weights_norm.shape[0])

    ckpt_dir = Path("checkpoints/jepa"); ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.hjepa_ckpt or str(ckpt_dir / f"weight_hjepa_{args.tag}.pt")
    train_losses: list[float] = []

    if args.eval_only:
        print(f"Eval-only: loading H-JEPA from {ckpt_path}")
        hjepa = load_hjepa(ckpt_path, device)
    else:
        hjepa = HierarchicalWeightJEPA(bounds, latent_dim=args.latent_dim)
        trainer = HierarchicalWeightJEPATrainer(hjepa, device=device, lr=args.lr)
        print(f"Phase 2 H-JEPA | weights {weights_norm.shape} | {hjepa.num_chunks} chunks | device={device}")
        for _ in tqdm(range(args.train_steps), desc="hjepa train"):
            idx = torch.randint(0, weights_norm.shape[0], (bs,))
            train_losses.append(trainer.step_batch(weights_norm[idx])["total"])
        hjepa.eval(); hjepa.fit_latent_stats(weights_norm.to(device))
        hjepa.save(ckpt_path)
        plot_loss_curve(train_losses[:: max(1, len(train_losses) // 500)], out_path=plot_dir / f"{args.tag}_train.png", title="H-JEPA train loss")

    print("Loading baselines + building hybrids...")
    flat_jepa = load_jepa(args.flat_jepa_ckpt, device)
    diff = load_diffusion(args.diffusion_ckpt, dim, device)
    h_hybrid = JEPADiffusionHybrid(hjepa, diff, refine_t=args.refine_t)
    flat_hybrid = JEPADiffusionHybrid(flat_jepa, diff, refine_t=args.refine_t)

    print(f"Sampling {args.num_samples} weights per method (hjepa, flat, diff, 2 hybrids)...")
    for name, fn in [("hjepa", lambda: hjepa.sample(n=args.num_samples, device=device).cpu()),
                     ("flat_jepa", lambda: flat_jepa.sample(n=args.num_samples, device=device).cpu()),
                     ("diffusion", lambda: load_diffusion_samples(args.diffusion_ckpt, dim, args.num_samples, device)),
                     ("h_hybrid", lambda: h_hybrid.sample(n=args.num_samples, device=device)),
                     ("flat_hybrid", lambda: flat_hybrid.sample(n=args.num_samples, device=device))]:
        print(f"  sampling {name}...", flush=True)
        out = fn()
        if name == "hjepa": hj_norm = out
        elif name == "flat_jepa": flat_norm = out
        elif name == "diffusion": diff_norm = out
        elif name == "h_hybrid": hh_norm = out
        else: fh_norm = out

    hj_raw = denormalize_weights(hj_norm, norm_meta)
    flat_raw = denormalize_weights(flat_norm, norm_meta)
    diff_raw = denormalize_weights(diff_norm, norm_meta)
    hh_raw = denormalize_weights(hh_norm, norm_meta)
    fh_raw = denormalize_weights(fh_norm, norm_meta)
    random_rows = torch.stack([flatten_state_dict(TinyMLP(hidden_dim=hidden)) for _ in range(args.num_samples)])

    plot_weight_histograms({"hjepa": hj_raw, "flat_jepa": flat_raw, "diffusion": diff_raw, "random": random_rows},
        out_path=plot_dir / f"{args.tag}_weight_hist.png", title="H-JEPA vs flat JEPA vs diffusion")

    rnd_rows = []
    print(f"Evaluating random baseline ({args.num_samples} samples)...", flush=True)
    for i in tqdm(range(args.num_samples), desc="random eval"):
        set_seed(1000 + i); rnd_rows.append(eval_flat(random_rows[i], hidden, device, ft_steps, args.seed + i))

    def run_eval(raw, prefix: str, seed_off: int) -> dict:
        rows = []
        for i in tqdm(range(args.num_samples), desc=f"{prefix} eval"):
            rows.append(eval_flat(raw[i], hidden, device, ft_steps, seed_off + i))
        return metrics_from_rows(rows, rnd_rows, ft_steps, prefix)

    print("Evaluating all methods (this is slow on CPU — ~1-2 min per sample)...", flush=True)
    hj_m = run_eval(hj_raw, "hjepa", args.seed)
    flat_m = run_eval(flat_raw, "flat_jepa", args.seed + 2000)
    diff_m = run_eval(diff_raw, "diffusion", args.seed + 5000)
    hh_m = run_eval(hh_raw, "h_hybrid", args.seed + 8000)
    fh_m = run_eval(fh_raw, "flat_hybrid", args.seed + 9000)

    metrics = {"tag": args.tag, "train_steps": args.train_steps, "refine_t": args.refine_t,
               "final_train_loss": train_losses[-1] if train_losses else None,
               **hj_m, **flat_m, **diff_m, **hh_m, **fh_m}
    metrics["hjepa_minus_flat_zero"] = flat_m["flat_jepa_zero_loss"] - hj_m["hjepa_zero_loss"]
    for n in ft_steps:
        if n <= 0: continue
        metrics[f"hjepa_minus_flat_ft{n}"] = flat_m[f"flat_jepa_ft{n}_loss"] - hj_m[f"hjepa_ft{n}_loss"]

    z_labels = ["hj_z", "flat_z", "diff_z", "hh_z", "fh_z", "rnd_z"]
    z_vals = [hj_m["hjepa_zero_loss"], flat_m["flat_jepa_zero_loss"], diff_m["diffusion_zero_loss"],
              hh_m["h_hybrid_zero_loss"], fh_m["flat_hybrid_zero_loss"], hj_m["random_zero_loss"]]
    for n in ft_steps:
        if n <= 0: continue
        z_labels += [f"hj_ft{n}", f"flat_ft{n}", f"diff_ft{n}", f"hh_ft{n}"]
        z_vals += [hj_m[f"hjepa_ft{n}_loss"], flat_m[f"flat_jepa_ft{n}_loss"], diff_m[f"diffusion_ft{n}_loss"], hh_m[f"h_hybrid_ft{n}_loss"]]
    plot_bar_comparison(z_labels, z_vals, out_path=plot_dir / f"{args.tag}_comparison_loss.png", title="H-JEPA comparison", ylabel="Loss")

    d_labels = ["hj_zΔ", "flat_zΔ", "diff_zΔ", "hh_zΔ", "fh_zΔ"]
    d_vals = [hj_m["hjepa_zero_delta_vs_random"], flat_m["flat_jepa_zero_delta_vs_random"], diff_m["diffusion_zero_delta_vs_random"],
              hh_m["h_hybrid_zero_delta_vs_random"], fh_m["flat_hybrid_zero_delta_vs_random"]]
    plot_bar_comparison(d_labels, d_vals, out_path=plot_dir / f"{args.tag}_delta.png", title="Delta vs random (+ better)", ylabel="Delta")

    out_json = ckpt_dir / f"metrics_{args.tag}.json"
    out_json.write_text(json.dumps(metrics, indent=2))
    log_run("checkpoints/logs", f"phase2_{args.tag}", metrics, vars(args))

    print(f"\n=== H-JEPA Summary [{args.tag}] ===")
    print(f"  zero: hjepa={hj_m['hjepa_zero_loss']:.4f} flat={flat_m['flat_jepa_zero_loss']:.4f} diff={diff_m['diffusion_zero_loss']:.4f}")
    print(f"  zero Δ: hj={hj_m['hjepa_zero_delta_vs_random']:.4f} flat={flat_m['flat_jepa_zero_delta_vs_random']:.4f}")
    print(f"  hybrid zero: h_hybrid={hh_m['h_hybrid_zero_loss']:.4f} flat_hybrid={fh_m['flat_hybrid_zero_loss']:.4f}")
    for n in ft_steps:
        if n <= 0: continue
        print(f"  ft{n}: hj={hj_m[f'hjepa_ft{n}_loss']:.4f} flat={flat_m[f'flat_jepa_ft{n}_loss']:.4f} h_hyb={hh_m[f'h_hybrid_ft{n}_loss']:.4f}")
    print(f"Artifacts: {plot_dir}/, {out_json}")


if __name__ == "__main__":
    main()
