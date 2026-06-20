#!/usr/bin/env python3
"""Phase 2: train minimal JEPA on global-normalized weights; compare vs diffusion baseline."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from tqdm import tqdm

from experiments.phase1_lib import avg_metric, eval_flat, prepare_normalized_weights
from src.diffusion.simple_diffusion import WeightDiffusion
from src.jepa.weight_jepa import WeightJEPA, WeightJEPATrainer
from src.models.tiny_mlp import TinyMLP
from src.utils.device import device_str, get_device
from src.utils.train_loop import set_seed
from src.utils.viz import plot_bar_comparison, plot_loss_curve, plot_weight_histograms
from src.utils.weights import denormalize_weights, flatten_state_dict, norm_meta_to_tensors


def eval_samples(raw_weights: torch.Tensor, hidden: int, device: str, ft_steps: list[int], seed: int, num: int) -> list[dict]:
    rows = []
    for i in range(num):
        rows.append(eval_flat(raw_weights[i], hidden, device, ft_steps, seed + i))
    return rows


def metrics_from_rows(gen_rows: list[dict], rnd_rows: list[dict], ft_steps: list[int], prefix: str) -> dict:
    m = {
        f"{prefix}_zero_loss": avg_metric(gen_rows, "zero_loss"),
        f"{prefix}_zero_acc": avg_metric(gen_rows, "zero_acc"),
        "random_zero_loss": avg_metric(rnd_rows, "zero_loss"),
    }
    m[f"{prefix}_zero_delta_vs_random"] = m["random_zero_loss"] - m[f"{prefix}_zero_loss"]
    for n in ft_steps:
        if n <= 0: continue
        m[f"{prefix}_ft{n}_loss"] = avg_metric(gen_rows, f"ft{n}_loss")
        m[f"{prefix}_ft{n}_delta_vs_random"] = avg_metric(rnd_rows, f"ft{n}_loss") - m[f"{prefix}_ft{n}_loss"]
    return m


def load_diffusion_samples(path: str, dim: int, n: int, device: str, timesteps: int = 200) -> torch.Tensor:
    diff = WeightDiffusion(dim=dim, timesteps=timesteps, device=device)
    diff.load(path)
    diff.model.eval()
    return diff.sample(n=n).cpu()


def main():
    p = argparse.ArgumentParser(description="Phase 2 JEPA vs diffusion")
    p.add_argument("--raw_weights", default="checkpoints/weights/phase0/weights_raw.pt")
    p.add_argument("--meta", default="checkpoints/weights/phase0/meta.json")
    p.add_argument("--diffusion_ckpt", default="checkpoints/diffusion/weight_denoiser_norm_global.pt")
    p.add_argument("--train_steps", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--num_samples", type=int, default=30)
    p.add_argument("--ft_steps", default="100,200")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot_dir", default="checkpoints/plots/phase2_jepa")
    p.add_argument("--tag", default="phase2_global")
    args = p.parse_args()
    ft_steps = [int(x) for x in args.ft_steps.split(",") if x.strip()]

    set_seed(args.seed)
    device = device_str()
    base_meta = json.loads(Path(args.meta).read_text())
    hidden = base_meta.get("hidden_dim", 64)
    slices = [tuple(x) for x in base_meta["layer_slices"]]
    weights_norm, norm_meta = prepare_normalized_weights(args.raw_weights, "global", slices)
    norm_meta = norm_meta_to_tensors(norm_meta)
    norm_meta["hidden_dim"] = hidden
    bounds = slices
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    jepa = WeightJEPA(bounds, latent_dim=args.latent_dim)
    trainer = WeightJEPATrainer(jepa, device=device, lr=args.lr)
    train_losses: list[float] = []
    bs = min(args.batch_size, weights_norm.shape[0])
    print(f"Phase 2 JEPA | weights {weights_norm.shape} | {jepa.num_chunks} chunks | device={device}")

    for _ in tqdm(range(args.train_steps), desc="jepa train"):
        idx = torch.randint(0, weights_norm.shape[0], (bs,))
        losses = trainer.step_batch(weights_norm[idx])
        train_losses.append(losses["total"])

    jepa.eval()
    jepa.fit_latent_stats(weights_norm.to(device))
    ckpt_dir = Path("checkpoints/jepa")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    jepa.save(str(ckpt_dir / f"weight_jepa_{args.tag}.pt"))
    plot_loss_curve(train_losses[:: max(1, len(train_losses) // 500)], out_path=plot_dir / f"{args.tag}_train.png", title="JEPA train loss")

    jepa_sampled_norm = jepa.sample(n=args.num_samples, device=device).cpu()
    diff_sampled_norm = load_diffusion_samples(args.diffusion_ckpt, weights_norm.shape[1], args.num_samples, device)
    jepa_raw = denormalize_weights(jepa_sampled_norm, norm_meta)
    diff_raw = denormalize_weights(diff_sampled_norm, norm_meta)
    random_rows = torch.stack([flatten_state_dict(TinyMLP(hidden_dim=hidden)) for _ in range(args.num_samples)])

    plot_weight_histograms(
        {"jepa": jepa_raw, "diffusion": diff_raw, "random": random_rows},
        out_path=plot_dir / f"{args.tag}_weight_hist.png",
        title="JEPA vs diffusion vs random",
    )

    jepa_rows = eval_samples(jepa_raw, hidden, device, ft_steps, args.seed, args.num_samples)
    diff_rows = eval_samples(diff_raw, hidden, device, ft_steps, args.seed + 5000, args.num_samples)
    rnd_rows = []
    for i in range(args.num_samples):
        set_seed(1000 + i)
        rnd_rows.append(eval_flat(random_rows[i], hidden, device, ft_steps, args.seed + i))

    jepa_m = metrics_from_rows(jepa_rows, rnd_rows, ft_steps, "jepa")
    diff_m = metrics_from_rows(diff_rows, rnd_rows, ft_steps, "diffusion")
    metrics = {"tag": args.tag, "train_steps": args.train_steps, "final_train_loss": train_losses[-1], **jepa_m, **diff_m}
    for n in ft_steps:
        if n <= 0: continue
        metrics[f"jepa_minus_diffusion_ft{n}"] = diff_m[f"diffusion_ft{n}_loss"] - jepa_m[f"jepa_ft{n}_loss"]

    z_labels = ["jepa_zero", "diff_zero", "rnd_zero"]
    z_vals = [jepa_m["jepa_zero_loss"], diff_m["diffusion_zero_loss"], jepa_m["random_zero_loss"]]
    for n in ft_steps:
        if n <= 0: continue
        z_labels += [f"jepa_ft{n}", f"diff_ft{n}", f"rnd_ft{n}"]
        z_vals += [jepa_m[f"jepa_ft{n}_loss"], diff_m[f"diffusion_ft{n}_loss"], avg_metric(rnd_rows, f"ft{n}_loss")]
    plot_bar_comparison(z_labels, z_vals, out_path=plot_dir / f"{args.tag}_comparison_loss.png", title="JEPA vs Diffusion", ylabel="Loss")

    delta_labels, delta_vals = ["jepa_zero_Δ", "diff_zero_Δ"], [jepa_m["jepa_zero_delta_vs_random"], diff_m["diffusion_zero_delta_vs_random"]]
    for n in ft_steps:
        if n <= 0: continue
        delta_labels += [f"jepa_ft{n}_Δ", f"diff_ft{n}_Δ"]
        delta_vals += [jepa_m[f"jepa_ft{n}_delta_vs_random"], diff_m[f"diffusion_ft{n}_delta_vs_random"]]
    plot_bar_comparison(delta_labels, delta_vals, out_path=plot_dir / f"{args.tag}_delta.png", title="Delta vs random (+ better)", ylabel="Delta")

    out_json = ckpt_dir / f"metrics_{args.tag}.json"
    out_json.write_text(json.dumps(metrics, indent=2))
    log_run("checkpoints/logs", f"phase2_{args.tag}", metrics, vars(args))

    print(f"\n=== Phase 2 Summary [{args.tag}] ===")
    print(f"  jepa_zero_loss={jepa_m['jepa_zero_loss']:.4f}  diffusion_zero_loss={diff_m['diffusion_zero_loss']:.4f}")
    print(f"  jepa_zero_Δ={jepa_m['jepa_zero_delta_vs_random']:.4f}  diff_zero_Δ={diff_m['diffusion_zero_delta_vs_random']:.4f}")
    for n in ft_steps:
        if n <= 0: continue
        print(f"  ft{n}: jepa={jepa_m[f'jepa_ft{n}_loss']:.4f} diff={diff_m[f'diffusion_ft{n}_loss']:.4f} jepa-diff={metrics[f'jepa_minus_diffusion_ft{n}']:.4f}")
    print(f"Artifacts: {plot_dir}/, {out_json}")


if __name__ == "__main__":
    main()
