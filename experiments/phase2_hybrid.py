#!/usr/bin/env python3
"""Phase 2 hybrid: JEPA init + diffusion refine vs pure JEPA, pure diffusion, random."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from experiments.phase1_lib import avg_metric, eval_flat, prepare_normalized_weights
from experiments.phase2_jepa_vs_diffusion import eval_samples, load_diffusion_samples, metrics_from_rows
from src.hybrid.jepa_diffusion import JEPADiffusionHybrid, load_diffusion, load_jepa
from src.models.tiny_mlp import TinyMLP
from src.utils.device import device_str, get_device
from src.utils.train_loop import set_seed
from src.utils.viz import plot_bar_comparison, plot_weight_histograms
from src.utils.weights import denormalize_weights, flatten_state_dict, norm_meta_to_tensors


def main():
    p = argparse.ArgumentParser(description="Phase 2 JEPA+diffusion hybrid eval")
    p.add_argument("--raw_weights", default="checkpoints/weights/phase0/weights_raw.pt")
    p.add_argument("--meta", default="checkpoints/weights/phase0/meta.json")
    p.add_argument("--jepa_ckpt", default="checkpoints/jepa/weight_jepa_phase2_global.pt")
    p.add_argument("--diffusion_ckpt", default="checkpoints/diffusion/weight_denoiser_norm_global.pt")
    p.add_argument("--refine_t", type=int, default=50, help="Diffusion start timestep for JEPA refine")
    p.add_argument("--num_samples", type=int, default=30)
    p.add_argument("--ft_steps", default="100,200")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot_dir", default="checkpoints/plots/phase2_hybrid")
    p.add_argument("--tag", default="hybrid_t50")
    args = p.parse_args()
    ft_steps = [int(x) for x in args.ft_steps.split(",") if x.strip()]

    set_seed(args.seed)
    device = device_str()
    base_meta = json.loads(Path(args.meta).read_text())
    hidden = base_meta.get("hidden_dim", 64)
    slices = [tuple(x) for x in base_meta["layer_slices"]]
    weights_norm, norm_meta = prepare_normalized_weights(args.raw_weights, "global", slices)
    norm_meta = norm_meta_to_tensors(norm_meta)
    dim = weights_norm.shape[1]
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"Phase 2 hybrid | refine_t={args.refine_t} | device={device}")
    jepa = load_jepa(args.jepa_ckpt, device)
    diff = load_diffusion(args.diffusion_ckpt, dim, device)
    hybrid = JEPADiffusionHybrid(jepa, diff, refine_t=args.refine_t)

    jepa_norm = jepa.sample(n=args.num_samples, device=device).cpu()
    diff_norm = load_diffusion_samples(args.diffusion_ckpt, dim, args.num_samples, device)
    hybrid_norm = hybrid.sample(n=args.num_samples, device=device)

    jepa_raw = denormalize_weights(jepa_norm, norm_meta)
    diff_raw = denormalize_weights(diff_norm, norm_meta)
    hybrid_raw = denormalize_weights(hybrid_norm, norm_meta)
    random_rows = torch.stack([flatten_state_dict(TinyMLP(hidden_dim=hidden)) for _ in range(args.num_samples)])

    plot_weight_histograms(
        {"jepa": jepa_raw, "diffusion": diff_raw, "hybrid": hybrid_raw, "random": random_rows},
        out_path=plot_dir / f"{args.tag}_weight_hist.png",
        title=f"Hybrid (refine_t={args.refine_t}) vs baselines",
    )

    rnd_rows = []
    for i in range(args.num_samples):
        set_seed(1000 + i)
        rnd_rows.append(eval_flat(random_rows[i], hidden, device, ft_steps, args.seed + i))

    jepa_m = metrics_from_rows(eval_samples(jepa_raw, hidden, device, ft_steps, args.seed, args.num_samples), rnd_rows, ft_steps, "jepa")
    diff_m = metrics_from_rows(eval_samples(diff_raw, hidden, device, ft_steps, args.seed + 5000, args.num_samples), rnd_rows, ft_steps, "diffusion")
    hyb_m = metrics_from_rows(eval_samples(hybrid_raw, hidden, device, ft_steps, args.seed + 9000, args.num_samples), rnd_rows, ft_steps, "hybrid")

    metrics = {"tag": args.tag, "refine_t": args.refine_t, **jepa_m, **diff_m, **hyb_m}
    for n in ft_steps:
        if n <= 0: continue
        metrics[f"hybrid_minus_jepa_ft{n}"] = jepa_m[f"jepa_ft{n}_loss"] - hyb_m[f"hybrid_ft{n}_loss"]
        metrics[f"hybrid_minus_diffusion_ft{n}"] = diff_m[f"diffusion_ft{n}_loss"] - hyb_m[f"hybrid_ft{n}_loss"]

    z_labels = ["jepa_z", "diff_z", "hyb_z", "rnd_z"]
    z_vals = [jepa_m["jepa_zero_loss"], diff_m["diffusion_zero_loss"], hyb_m["hybrid_zero_loss"], jepa_m["random_zero_loss"]]
    for n in ft_steps:
        if n <= 0: continue
        z_labels += [f"jepa_ft{n}", f"diff_ft{n}", f"hyb_ft{n}", f"rnd_ft{n}"]
        z_vals += [jepa_m[f"jepa_ft{n}_loss"], diff_m[f"diffusion_ft{n}_loss"], hyb_m[f"hybrid_ft{n}_loss"], avg_metric(rnd_rows, f"ft{n}_loss")]
    plot_bar_comparison(z_labels, z_vals, out_path=plot_dir / f"{args.tag}_comparison_loss.png", title="JEPA vs Diffusion vs Hybrid", ylabel="Loss")

    d_labels = ["jepa_zΔ", "diff_zΔ", "hyb_zΔ"]
    d_vals = [jepa_m["jepa_zero_delta_vs_random"], diff_m["diffusion_zero_delta_vs_random"], hyb_m["hybrid_zero_delta_vs_random"]]
    for n in ft_steps:
        if n <= 0: continue
        d_labels += [f"jepa_ft{n}Δ", f"diff_ft{n}Δ", f"hyb_ft{n}Δ"]
        d_vals += [jepa_m[f"jepa_ft{n}_delta_vs_random"], diff_m[f"diffusion_ft{n}_delta_vs_random"], hyb_m[f"hybrid_ft{n}_delta_vs_random"]]
    plot_bar_comparison(d_labels, d_vals, out_path=plot_dir / f"{args.tag}_delta.png", title="Delta vs random (+ better)", ylabel="Delta")

    out_json = Path("checkpoints/hybrid") / f"metrics_{args.tag}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2))
    log_run("checkpoints/logs", f"phase2_hybrid_{args.tag}", metrics, vars(args))

    print(f"\n=== Hybrid Summary [{args.tag}] refine_t={args.refine_t} ===")
    print(f"  zero: jepa={jepa_m['jepa_zero_loss']:.4f} diff={diff_m['diffusion_zero_loss']:.4f} hybrid={hyb_m['hybrid_zero_loss']:.4f}")
    print(f"  zero Δ: jepa={jepa_m['jepa_zero_delta_vs_random']:.4f} diff={diff_m['diffusion_zero_delta_vs_random']:.4f} hybrid={hyb_m['hybrid_zero_delta_vs_random']:.4f}")
    for n in ft_steps:
        if n <= 0: continue
        print(f"  ft{n}: jepa={jepa_m[f'jepa_ft{n}_loss']:.4f} diff={diff_m[f'diffusion_ft{n}_loss']:.4f} hybrid={hyb_m[f'hybrid_ft{n}_loss']:.4f}")
    print(f"Artifacts: {plot_dir}/, {out_json}")


if __name__ == "__main__":
    main()
