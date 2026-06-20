#!/usr/bin/env python3
"""Phase 1 normalization ablation: global vs layer vs perdim on same raw weight collection."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.logging import log_run
from src.utils.viz import plot_bar_comparison
from src.utils.weights import NORM_MODES
from experiments.phase1_lib import prepare_normalized_weights, run_phase1_experiment


def main():
    p = argparse.ArgumentParser(description="Phase 1 normalization ablation")
    p.add_argument("--raw_weights", type=str, default="checkpoints/weights/phase0/weights_raw.pt")
    p.add_argument("--meta", type=str, default="checkpoints/weights/phase0/meta.json")
    p.add_argument("--diffusion_steps", type=int, default=2000)
    p.add_argument("--num_samples", type=int, default=30)
    p.add_argument("--ft_steps", type=str, default="100,200")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot_dir", type=str, default="checkpoints/plots/phase1_norm_ablation")
    args = p.parse_args()
    ft_steps = [int(x.strip()) for x in args.ft_steps.split(",") if x.strip()]

    base_meta = json.loads(Path(args.meta).read_text())
    hidden = base_meta.get("hidden_dim", 64)
    slices = [tuple(x) for x in base_meta["layer_slices"]]
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: dict[str, dict] = {}
    for mode in NORM_MODES:
        tag = f"norm_{mode}"
        norm_w, norm_meta = prepare_normalized_weights(args.raw_weights, mode, slices)
        norm_meta["hidden_dim"] = hidden
        m = run_phase1_experiment(
            norm_w, norm_meta, tag=tag, hidden=hidden, diffusion_steps=args.diffusion_steps,
            num_samples=args.num_samples, ft_steps=ft_steps, seed=args.seed, plot_dir=plot_dir,
        )
        all_metrics[mode] = m
        out = Path("checkpoints/diffusion") / f"metrics_{tag}.json"
        out.write_text(json.dumps(m, indent=2))

    # Cross-norm comparison plots (generated metrics only)
    z_labels = list(NORM_MODES)
    plot_bar_comparison(z_labels, [all_metrics[m]["generated_zero_loss"] for m in NORM_MODES],
        out_path=plot_dir / "ablation_zero_shot_loss.png", title="Zero-shot loss by norm (generated)", ylabel="Loss")
    plot_bar_comparison(z_labels, [all_metrics[m]["zero_shot_delta_vs_random"] for m in NORM_MODES],
        out_path=plot_dir / "ablation_zero_shot_delta.png", title="Zero-shot delta vs random (+ = gen better)", ylabel="Delta")
    for n in ft_steps:
        if n <= 0: continue
        plot_bar_comparison(z_labels, [all_metrics[m][f"ft{n}_delta_vs_random"] for m in NORM_MODES],
            out_path=plot_dir / f"ablation_ft{n}_delta.png", title=f"FT{n} delta vs random by norm", ylabel="Delta")

    summary = {"modes": all_metrics, "ft_steps": ft_steps, "diffusion_steps": args.diffusion_steps}
    summary_path = plot_dir / "ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log_run("checkpoints/logs", "phase1_norm_ablation", {f"{m}_zero_delta": all_metrics[m]["zero_shot_delta_vs_random"] for m in NORM_MODES}, vars(args))

    print("\n=== Normalization Ablation Summary ===")
    print(f"{'mode':<8} {'zero_loss':>10} {'zero_Δ':>10} ", end="")
    for n in ft_steps:
        if n > 0: print(f"{'ft'+str(n)+'_Δ':>10} ", end="")
    print()
    for mode in NORM_MODES:
        m = all_metrics[mode]
        print(f"{mode:<8} {m['generated_zero_loss']:>10.4f} {m['zero_shot_delta_vs_random']:>10.4f} ", end="")
        for n in ft_steps:
            if n > 0: print(f"{m[f'ft{n}_delta_vs_random']:>10.4f} ", end="")
        print()
    print(f"\nSummary -> {summary_path}")


if __name__ == "__main__":
    main()
