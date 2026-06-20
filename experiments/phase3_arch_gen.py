#!/usr/bin/env python3
"""Phase 3: generate variable architectures + JEPA/hybrid weights -> eval vs fixed baselines."""

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
from src.arch_gen.eval import eval_arch_row
from src.arch_gen.generator import ArchGenerator, ArchGeneratorTrainer
from src.arch_gen.spec import ArchSpec, REF_DEPTH, REF_HIDDEN
from src.arch_gen.weight_init import init_model_weights, load_hybrid
from src.models.variable_mlp import build_from_spec
from src.utils.device import device_str
from src.utils.logging import log_run
from src.utils.train_loop import set_seed
from src.utils.viz import plot_bar_comparison, plot_loss_curve
from src.utils.weights import norm_meta_to_tensors


def main():
    p = argparse.ArgumentParser(description="Phase 3 architecture generation")
    p.add_argument("--raw_weights", default="checkpoints/weights/phase0/weights_raw.pt")
    p.add_argument("--meta", default="checkpoints/weights/phase0/meta.json")
    p.add_argument("--jepa_ckpt", default="checkpoints/jepa/weight_jepa_phase2_global.pt")
    p.add_argument("--diffusion_ckpt", default="checkpoints/diffusion/weight_denoiser_norm_global.pt")
    p.add_argument("--refine_t", type=int, default=50)
    p.add_argument("--arch_train_steps", type=int, default=500)
    p.add_argument("--num_samples", type=int, default=30)
    p.add_argument("--ft_steps", default="100")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot_dir", default="checkpoints/plots/phase3_arch")
    p.add_argument("--tag", default="phase3_v1")
    args = p.parse_args()
    ft_steps = [int(x) for x in args.ft_steps.split(",") if x.strip()]

    set_seed(args.seed)
    device = device_str()
    base_meta = json.loads(Path(args.meta).read_text())
    slices = [tuple(x) for x in base_meta["layer_slices"]]
    _, norm_meta = prepare_normalized_weights(args.raw_weights, "global", slices)
    norm_meta = norm_meta_to_tensors(norm_meta)
    ref_dim = base_meta["weight_dim"]
    plot_dir = Path(args.plot_dir); plot_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("checkpoints/arch_gen"); ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Phase 3 arch gen | device={device} | ref h={REF_HIDDEN} d={REF_DEPTH}")
    gen = ArchGenerator()
    arch_losses = ArchGeneratorTrainer(gen, device=device).fit(steps=args.arch_train_steps)
    torch.save(gen.state_dict(), ckpt_dir / f"arch_generator_{args.tag}.pt")
    plot_loss_curve(arch_losses, out_path=plot_dir / f"{args.tag}_arch_train.png", title="Arch generator train loss")

    specs = gen.sample(n=args.num_samples, device=device)
    hybrid = load_hybrid(args.jepa_ckpt, args.diffusion_ckpt, ref_dim, device, args.refine_t)
    ref_spec = ArchSpec(REF_HIDDEN, REF_DEPTH)

    diagrams = []
    gen_rows, init_methods = [], []
    for i, spec in enumerate(tqdm(specs, desc="gen arch eval")):
        model = build_from_spec(spec).to(device)
        method = init_model_weights(model, spec, hybrid=hybrid, norm_meta=norm_meta, device=device)
        init_methods.append(method)
        gen_rows.append(eval_arch_row(model, device, ft_steps, args.seed + i))
        if i < 5: diagrams.append(f"--- sample {i} ({method}) ---\n{spec.diagram()}\nparams={model.num_parameters()}")

    rnd_ref = build_from_spec(ref_spec).to(device)
    init_model_weights(rnd_ref, ref_spec, method="random")
    rnd_rows_ref = [eval_arch_row(rnd_ref, device, ft_steps, args.seed + 1000)]

    hyb_ref = build_from_spec(ref_spec).to(device)
    init_model_weights(hyb_ref, ref_spec, hybrid=hybrid, norm_meta=norm_meta, device=device)
    hyb_rows_ref = [eval_arch_row(hyb_ref, device, ft_steps, args.seed + 2000)]

    rnd_baseline = []
    for i in range(args.num_samples):
        m = build_from_spec(ref_spec).to(device)
        init_model_weights(m, ref_spec, method="random")
        set_seed(1000 + i)
        rnd_baseline.append(eval_arch_row(m, device, ft_steps, args.seed + i))

    gen_m = metrics_from_rows(gen_rows, rnd_baseline, ft_steps, "gen")
    ref_rnd_m = metrics_from_rows(rnd_rows_ref, rnd_baseline, ft_steps, "ref_random")
    ref_hyb_m = metrics_from_rows(hyb_rows_ref, rnd_baseline, ft_steps, "ref_hybrid")

    jepa_count = sum(1 for m in init_methods if m == "hybrid")
    h_dist = Counter(s.hidden_dim for s in specs)
    d_dist = Counter(s.depth for s in specs)
    unique_specs = len({(s.hidden_dim, s.depth) for s in specs})

    metrics = {
        "tag": args.tag, "num_samples": args.num_samples, "arch_train_steps": args.arch_train_steps,
        "final_arch_train_loss": arch_losses[-1], "jepa_compatible_count": jepa_count,
        "unique_arch_specs": unique_specs, "hidden_dim_dist": dict(h_dist), "depth_dist": dict(d_dist),
        **gen_m, **ref_rnd_m, **ref_hyb_m,
    }
    for n in ft_steps:
        if n <= 0: continue
        metrics[f"gen_minus_ref_hybrid_ft{n}"] = ref_hyb_m[f"ref_hybrid_ft{n}_loss"] - gen_m[f"gen_ft{n}_loss"]

    (plot_dir / f"{args.tag}_sample_archs.txt").write_text("\n\n".join(diagrams))
    z_labels = ["gen_zero", "ref_hyb_zero", "ref_rnd_zero", "rnd_zero"]
    z_vals = [gen_m["gen_zero_loss"], ref_hyb_m["ref_hybrid_zero_loss"], ref_rnd_m["ref_random_zero_loss"], gen_m["random_zero_loss"]]
    for n in ft_steps:
        if n <= 0: continue
        z_labels += [f"gen_ft{n}", f"ref_hyb_ft{n}", f"ref_rnd_ft{n}"]
        z_vals += [gen_m[f"gen_ft{n}_loss"], ref_hyb_m[f"ref_hybrid_ft{n}_loss"], ref_rnd_m[f"ref_random_ft{n}_loss"]]
    plot_bar_comparison(z_labels, z_vals, out_path=plot_dir / f"{args.tag}_comparison_loss.png", title="Phase 3 arch+weights", ylabel="Loss")

    out_json = ckpt_dir / f"metrics_{args.tag}.json"
    out_json.write_text(json.dumps(metrics, indent=2))
    log_run("checkpoints/logs", f"phase3_{args.tag}", metrics, vars(args))

    print(f"\n=== Phase 3 Summary [{args.tag}] ===")
    print(f"  generated {args.num_samples} archs | unique={unique_specs} | jepa-compatible={jepa_count}/{args.num_samples}")
    print(f"  zero: gen={gen_m['gen_zero_loss']:.4f} ref_hybrid={ref_hyb_m['ref_hybrid_zero_loss']:.4f} ref_rnd={ref_rnd_m['ref_random_zero_loss']:.4f}")
    print(f"  zero Δ: gen={gen_m['gen_zero_delta_vs_random']:.4f} ref_hyb={ref_hyb_m['ref_hybrid_zero_delta_vs_random']:.4f}")
    for n in ft_steps:
        if n <= 0: continue
        print(f"  ft{n}: gen={gen_m[f'gen_ft{n}_loss']:.4f} ref_hyb={ref_hyb_m[f'ref_hybrid_ft{n}_loss']:.4f}")
    print(f"Artifacts: {plot_dir}/, {out_json}")


if __name__ == "__main__":
    main()
