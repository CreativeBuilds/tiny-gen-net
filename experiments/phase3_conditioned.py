#!/usr/bin/env python3
"""Phase 3 conditioned: arch embed -> CondWeightGenerator vs unconditioned Phase 3 v1."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from tqdm import tqdm

from experiments.phase1_lib import prepare_normalized_weights
from experiments.phase2_jepa_vs_diffusion import metrics_from_rows
from src.arch_gen.conditioning import (
    CondWeightGenerator, CondWeightTrainer, normalize_multi_arch, ref_similarity, save_cond_checkpoint,
)
from src.arch_gen.eval import eval_arch_row
from src.arch_gen.generator import ArchGenerator, ArchGeneratorTrainer
from src.arch_gen.spec import ArchSpec, REF_DEPTH, REF_HIDDEN
from src.arch_gen.weight_init import init_model_weights, load_hybrid
from src.models.variable_mlp import build_from_spec
from src.utils.device import device_str
from src.utils.logging import log_run
from src.utils.train_loop import set_seed
from src.utils.viz import plot_bar_comparison, plot_loss_curve, plot_scatter
from src.utils.weights import norm_meta_to_tensors


def load_multi_arch(path: Path) -> tuple[list[ArchSpec], list[torch.Tensor]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    meta = ckpt["meta"]
    specs = [ArchSpec.from_label(lbl) for lbl in ckpt["meta"]["specs"]]
    weights = ckpt["weights"]
    if isinstance(weights, torch.Tensor): weights = [weights[i] for i in range(weights.shape[0])]
    return specs, weights


def main():
    p = argparse.ArgumentParser(description="Phase 3 arch-conditioned weight generation")
    p.add_argument("--multi_arch", default="checkpoints/weights/multi_arch/weights_raw.pt")
    p.add_argument("--raw_weights", default="checkpoints/weights/phase0/weights_raw.pt")
    p.add_argument("--meta", default="checkpoints/weights/phase0/meta.json")
    p.add_argument("--jepa_ckpt", default="checkpoints/jepa/weight_jepa_phase2_global.pt")
    p.add_argument("--diffusion_ckpt", default="checkpoints/diffusion/weight_denoiser_norm_global.pt")
    p.add_argument("--refine_t", type=int, default=50)
    p.add_argument("--cond_train_steps", type=int, default=2000)
    p.add_argument("--arch_train_steps", type=int, default=500)
    p.add_argument("--num_samples", type=int, default=30)
    p.add_argument("--ft_steps", default="100")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot_dir", default="checkpoints/plots/phase3_conditioned")
    p.add_argument("--tag", default="phase3_cond_v1")
    p.add_argument("--skip_collect", action="store_true")
    args = p.parse_args()
    ft_steps = [int(x) for x in args.ft_steps.split(",") if x.strip()]

    set_seed(args.seed)
    device = device_str()
    ma_path = Path(args.multi_arch)
    if not ma_path.exists() and not args.skip_collect:
        print("Multi-arch weights missing — collecting (10 specs × 10 models)...")
        import subprocess
        subprocess.run([sys.executable, "scripts/collect_multi_arch_weights.py", "--num_per_spec", "10"], check=True, cwd=ROOT)

    train_specs, raw_w = load_multi_arch(ma_path)
    norm_w, cond_norm = normalize_multi_arch(raw_w)
    base_meta = json.loads(Path(args.meta).read_text())
    slices = [tuple(x) for x in base_meta["layer_slices"]]
    _, norm_meta = prepare_normalized_weights(args.raw_weights, "global", slices)
    norm_meta = norm_meta_to_tensors(norm_meta)
    ref_dim = base_meta["weight_dim"]
    plot_dir = Path(args.plot_dir); plot_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("checkpoints/arch_gen"); ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Phase 3 conditioned | device={device} | train N={len(train_specs)}")
    cond = CondWeightGenerator()
    cond_losses = CondWeightTrainer(cond, device=device).fit(train_specs, norm_w, steps=args.cond_train_steps)
    save_cond_checkpoint(str(ckpt_dir / f"cond_weights_{args.tag}.pt"), cond, cond_norm)
    plot_loss_curve(cond_losses[:: max(1, len(cond_losses) // 500)], out_path=plot_dir / f"{args.tag}_cond_train.png", title="Cond weight generator loss")

    gen = ArchGenerator()
    ArchGeneratorTrainer(gen, device=device).fit(steps=args.arch_train_steps)
    specs = gen.sample(n=args.num_samples, device=device)
    hybrid = load_hybrid(args.jepa_ckpt, args.diffusion_ckpt, ref_dim, device, args.refine_t)
    ref_spec = ArchSpec(REF_HIDDEN, REF_DEPTH)

    cond_rows, uncond_rows, sims, methods = [], [], [], []
    for i, spec in enumerate(tqdm(specs, desc="conditioned eval")):
        sims.append(ref_similarity(spec))
        mc = build_from_spec(spec).to(device)
        methods.append(init_model_weights(mc, spec, cond_gen=cond, cond_norm=cond_norm, device=device, method="conditioned"))
        cond_rows.append(eval_arch_row(mc, device, ft_steps, args.seed + i))
        mu = build_from_spec(spec).to(device)
        init_model_weights(mu, spec, hybrid=hybrid, norm_meta=norm_meta, device=device, method="auto")
        uncond_rows.append(eval_arch_row(mu, device, ft_steps, args.seed + 5000 + i))

    rnd_baseline = []
    for i in range(args.num_samples):
        m = build_from_spec(ref_spec).to(device)
        init_model_weights(m, ref_spec, method="random")
        set_seed(1000 + i)
        rnd_baseline.append(eval_arch_row(m, device, ft_steps, args.seed + i))

    hyb_ref = build_from_spec(ref_spec).to(device)
    init_model_weights(hyb_ref, ref_spec, hybrid=hybrid, norm_meta=norm_meta, device=device)
    hyb_rows_ref = [eval_arch_row(hyb_ref, device, ft_steps, args.seed + 9000)]

    cond_m = metrics_from_rows(cond_rows, rnd_baseline, ft_steps, "cond")
    uncond_m = metrics_from_rows(uncond_rows, rnd_baseline, ft_steps, "uncond")
    ref_hyb_m = metrics_from_rows(hyb_rows_ref, rnd_baseline, ft_steps, "ref_hybrid")

    v1_path = ckpt_dir / "metrics_phase3_v1.json"
    v1_zero = json.loads(v1_path.read_text())["gen_zero_loss"] if v1_path.exists() else None

    metrics = {
        "tag": args.tag, "cond_train_steps": args.cond_train_steps, "num_samples": args.num_samples,
        "conditioned_init_count": sum(1 for m in methods if m == "conditioned"),
        "hidden_dim_dist": dict(Counter(s.hidden_dim for s in specs)),
        "depth_dist": dict(Counter(s.depth for s in specs)),
        "phase3_v1_zero_loss": v1_zero, **cond_m, **uncond_m, **ref_hyb_m,
    }
    metrics["cond_minus_uncond_zero"] = uncond_m["uncond_zero_loss"] - cond_m["cond_zero_loss"]
    metrics["cond_minus_v1_zero"] = (v1_zero - cond_m["cond_zero_loss"]) if v1_zero else None
    for n in ft_steps:
        if n <= 0: continue
        metrics[f"cond_minus_uncond_ft{n}"] = uncond_m[f"uncond_ft{n}_loss"] - cond_m[f"cond_ft{n}_loss"]

    zero_losses = [r["zero_loss"] for r in cond_rows]
    plot_scatter(sims, zero_losses, out_path=plot_dir / f"{args.tag}_sim_vs_zero.png",
                 title="Zero-shot loss vs ref similarity", xlabel="Ref similarity", ylabel="Zero loss")

    z_labels = ["cond_zero", "uncond_zero", "v1_zero", "ref_hyb_zero", "rnd_zero"]
    z_vals = [cond_m["cond_zero_loss"], uncond_m["uncond_zero_loss"], v1_zero or float("nan"),
              ref_hyb_m["ref_hybrid_zero_loss"], cond_m["random_zero_loss"]]
    for n in ft_steps:
        if n <= 0: continue
        z_labels += [f"cond_ft{n}", f"uncond_ft{n}", f"ref_hyb_ft{n}"]
        z_vals += [cond_m[f"cond_ft{n}_loss"], uncond_m[f"uncond_ft{n}_loss"], ref_hyb_m[f"ref_hybrid_ft{n}_loss"]]
    plot_bar_comparison(z_labels, z_vals, out_path=plot_dir / f"{args.tag}_comparison_loss.png", title="Conditioned vs unconditioned", ylabel="Loss")

    d_labels = ["cond_zΔ", "uncond_zΔ", "ref_hyb_zΔ"]
    d_vals = [cond_m["cond_zero_delta_vs_random"], uncond_m["uncond_zero_delta_vs_random"], ref_hyb_m["ref_hybrid_zero_delta_vs_random"]]
    plot_bar_comparison(d_labels, d_vals, out_path=plot_dir / f"{args.tag}_delta.png", title="Delta vs random (+ better)", ylabel="Delta")

    out_json = ckpt_dir / f"metrics_{args.tag}.json"
    out_json.write_text(json.dumps(metrics, indent=2))
    log_run("checkpoints/logs", f"phase3_{args.tag}", metrics, vars(args))

    print(f"\n=== Phase 3 Conditioned [{args.tag}] ===")
    print(f"  zero: cond={cond_m['cond_zero_loss']:.4f} uncond={uncond_m['uncond_zero_loss']:.4f} v1={v1_zero} ref_hyb={ref_hyb_m['ref_hybrid_zero_loss']:.4f}")
    print(f"  zero Δ: cond={cond_m['cond_zero_delta_vs_random']:.4f} uncond={uncond_m['uncond_zero_delta_vs_random']:.4f}")
    for n in ft_steps:
        if n <= 0: continue
        print(f"  ft{n}: cond={cond_m[f'cond_ft{n}_loss']:.4f} uncond={uncond_m[f'uncond_ft{n}_loss']:.4f}")
    print(f"Artifacts: {plot_dir}/, {out_json}")


if __name__ == "__main__":
    main()
