#!/usr/bin/env python3
"""Phase 3 refine: expanded arch diversity + conditioned eval + noise-scale sweep."""

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
    sample_denorm,
)
from src.arch_gen.eval import eval_arch_row
from src.arch_gen.generator import ArchGenerator, ArchGeneratorTrainer
from src.arch_gen.spec import ArchSpec, REF_DEPTH, REF_HIDDEN, all_valid_specs
from src.arch_gen.weight_init import init_model_weights, load_hybrid
from src.models.variable_mlp import build_from_spec
from src.utils.device import device_str
from src.utils.logging import log_run
from src.utils.train_loop import set_seed
from src.utils.viz import plot_bar_comparison, plot_loss_curve, plot_scatter
from src.utils.weights import norm_meta_to_tensors


def load_multi_arch(path: Path) -> tuple[list[ArchSpec], list[torch.Tensor]]:
    from src.arch_gen.spec import ArchSpec
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    specs = [ArchSpec.from_label(lbl) for lbl in ckpt["meta"]["specs"]]
    weights = ckpt["weights"]
    if isinstance(weights, torch.Tensor): weights = [weights[i] for i in range(weights.shape[0])]
    return specs, weights


def diversity_stats(specs: list[ArchSpec]) -> dict:
    return {
        "unique": len({(s.hidden_dim, s.depth, s.skip) for s in specs}),
        "hidden_dim": dict(Counter(s.hidden_dim for s in specs)),
        "depth": dict(Counter(s.depth for s in specs)),
        "skip": dict(Counter(s.skip for s in specs)),
        "grid_size": len(all_valid_specs()),
    }


def eval_specs(specs: list[ArchSpec], cond, cond_norm, device, ft_steps, seed: int, noise_scale: float = 1.0) -> list[dict]:
    rows = []
    for i, spec in enumerate(specs):
        m = build_from_spec(spec).to(device)
        init_model_weights(m, spec, cond_gen=cond, cond_norm=cond_norm, device=device, method="conditioned", noise_scale=noise_scale)
        rows.append(eval_arch_row(m, device, ft_steps, seed + i))
    return rows


def main():
    p = argparse.ArgumentParser(description="Phase 3 conditioned refine — expanded diversity")
    p.add_argument("--multi_arch", default="checkpoints/weights/multi_arch_v2/weights_raw.pt")
    p.add_argument("--raw_weights", default="checkpoints/weights/phase0/weights_raw.pt")
    p.add_argument("--meta", default="checkpoints/weights/phase0/meta.json")
    p.add_argument("--jepa_ckpt", default="checkpoints/jepa/weight_jepa_phase2_global.pt")
    p.add_argument("--diffusion_ckpt", default="checkpoints/diffusion/weight_denoiser_norm_global.pt")
    p.add_argument("--cond_train_steps", type=int, default=2000)
    p.add_argument("--arch_train_steps", type=int, default=800)
    p.add_argument("--num_gen", type=int, default=80)
    p.add_argument("--num_eval", type=int, default=50)
    p.add_argument("--ft_steps", default="100")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot_dir", default="checkpoints/plots/phase3_refine")
    p.add_argument("--tag", default="phase3_refine_v1")
    p.add_argument("--skip_collect", action="store_true")
    args = p.parse_args()
    ft_steps = [int(x) for x in args.ft_steps.split(",") if x.strip()]

    set_seed(args.seed)
    device = device_str()
    ma_path = Path(args.multi_arch)
    if not ma_path.exists() and not args.skip_collect:
        print(f"Collecting expanded grid ({len(all_valid_specs())} specs × 4 models)...")
        import subprocess
        subprocess.run([sys.executable, "scripts/collect_multi_arch_weights.py", "--num_per_spec", "4", "--out", str(ma_path.parent)], check=True, cwd=ROOT)

    train_specs, raw_w = load_multi_arch(ma_path)
    norm_w, cond_norm = normalize_multi_arch(raw_w)
    base_meta = json.loads(Path(args.meta).read_text())
    _, norm_meta = prepare_normalized_weights(args.raw_weights, "global", [tuple(x) for x in base_meta["layer_slices"]])
    norm_meta = norm_meta_to_tensors(norm_meta)
    ref_dim = base_meta["weight_dim"]
    plot_dir = Path(args.plot_dir); plot_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("checkpoints/arch_gen"); ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Phase 3 refine | grid={len(all_valid_specs())} specs | train N={len(train_specs)} | device={device}")
    cond = CondWeightGenerator()
    cond_losses = CondWeightTrainer(cond, device=device).fit(train_specs, norm_w, steps=args.cond_train_steps)
    save_cond_checkpoint(str(ckpt_dir / f"cond_weights_{args.tag}.pt"), cond, cond_norm)
    plot_loss_curve(cond_losses[:: max(1, len(cond_losses) // 500)], out_path=plot_dir / f"{args.tag}_cond_train.png", title="Cond weight train (v2 grid)")

    gen = ArchGenerator()
    ArchGeneratorTrainer(gen, device=device).fit(steps=args.arch_train_steps)
    torch.save(gen.state_dict(), ckpt_dir / f"arch_generator_{args.tag}.pt")
    all_specs = gen.sample(n=args.num_gen, device=device)
    eval_specs_list = all_specs[: args.num_eval]
    div = diversity_stats(eval_specs_list)
    hybrid = load_hybrid(args.jepa_ckpt, args.diffusion_ckpt, ref_dim, device)
    ref_spec = ArchSpec(REF_HIDDEN, REF_DEPTH)

    cond_rows = eval_specs(eval_specs_list, cond, cond_norm, device, ft_steps, args.seed)
    sims = [ref_similarity(s) for s in eval_specs_list]
    zero_losses = [r["zero_loss"] for r in cond_rows]

    rnd_baseline = []
    for i in range(args.num_eval):
        m = build_from_spec(ref_spec).to(device)
        init_model_weights(m, ref_spec, method="random")
        set_seed(1000 + i)
        rnd_baseline.append(eval_arch_row(m, device, ft_steps, args.seed + i))

    hyb_ref = build_from_spec(ref_spec).to(device)
    init_model_weights(hyb_ref, ref_spec, hybrid=hybrid, norm_meta=norm_meta, device=device)
    hyb_rows = [eval_arch_row(hyb_ref, device, ft_steps, args.seed + 9000)]

    cond_m = metrics_from_rows(cond_rows, rnd_baseline, ft_steps, "cond")
    ref_hyb_m = metrics_from_rows(hyb_rows, rnd_baseline, ft_steps, "ref_hybrid")

    v1_path = ckpt_dir / "metrics_phase3_cond_v1.json"
    v1 = json.loads(v1_path.read_text()) if v1_path.exists() else {}

    # noise-scale sweep on 10 held-out random grid specs
    sweep_specs = all_valid_specs()[:10]
    sweep_results = {}
    for scale in [0.25, 0.5, 1.0, 2.0]:
        rows = eval_specs(sweep_specs, cond, cond_norm, device, ft_steps, args.seed + 7000, noise_scale=scale)
        sweep_results[str(scale)] = sum(r["zero_loss"] for r in rows) / len(rows)

    metrics = {"tag": args.tag, "cond_train_steps": args.cond_train_steps, "num_eval": args.num_eval, "diversity": div,
               "noise_sweep_zero_loss": sweep_results, "prev_cond_v1_zero": v1.get("cond_zero_loss"), **cond_m, **ref_hyb_m}
    metrics["cond_minus_v1_zero"] = (v1.get("cond_zero_loss") - cond_m["cond_zero_loss"]) if v1.get("cond_zero_loss") else None

    plot_scatter(sims, zero_losses, out_path=plot_dir / f"{args.tag}_sim_vs_zero.png", title="Zero loss vs ref similarity", xlabel="Ref similarity", ylabel="Zero loss")
    plot_scatter([s.hidden_dim for s in eval_specs_list], zero_losses, out_path=plot_dir / f"{args.tag}_hidden_vs_zero.png", title="Zero loss vs hidden dim", xlabel="Hidden", ylabel="Zero loss")
    plot_scatter([s.depth for s in eval_specs_list], zero_losses, out_path=plot_dir / f"{args.tag}_depth_vs_zero.png", title="Zero loss vs depth", xlabel="Depth", ylabel="Zero loss")

    z_labels = ["cond_v2", "cond_v1", "ref_hyb", "rnd"]
    z_vals = [cond_m["cond_zero_loss"], v1.get("cond_zero_loss", float("nan")), ref_hyb_m["ref_hybrid_zero_loss"], cond_m["random_zero_loss"]]
    plot_bar_comparison(z_labels, z_vals, out_path=plot_dir / f"{args.tag}_zero_compare.png", title="Zero-shot comparison", ylabel="Loss")

    sweep_labels, sweep_vals = list(sweep_results.keys()), list(sweep_results.values())
    plot_bar_comparison(sweep_labels, sweep_vals, out_path=plot_dir / f"{args.tag}_noise_sweep.png", title="Noise scale sweep (10 specs)", ylabel="Avg zero loss")

    out_json = ckpt_dir / f"metrics_{args.tag}.json"
    out_json.write_text(json.dumps(metrics, indent=2))
    log_run("checkpoints/logs", f"phase3_{args.tag}", metrics, vars(args))

    print(f"\n=== Phase 3 Refine [{args.tag}] ===")
    print(f"  diversity: unique={div['unique']}/{args.num_eval} | depth={div['depth']} | skip={div['skip']}")
    print(f"  zero: cond_v2={cond_m['cond_zero_loss']:.4f} cond_v1={v1.get('cond_zero_loss')} ref_hyb={ref_hyb_m['ref_hybrid_zero_loss']:.4f}")
    print(f"  zero Δ: {cond_m['cond_zero_delta_vs_random']:.4f} | noise sweep: {sweep_results}")
    for n in ft_steps:
        if n <= 0: continue
        print(f"  ft{n}: cond={cond_m[f'cond_ft{n}_loss']:.4f} ref_hyb={ref_hyb_m[f'ref_hybrid_ft{n}_loss']:.4f}")
    print(f"Artifacts: {plot_dir}/, {out_json}")


if __name__ == "__main__":
    main()
