#!/usr/bin/env python3
"""Alignment ablation: does symmetry-canonicalizing weights make them learnable?

Two arms, IDENTICAL generator hyperparameters, only difference is whether
Git Re-Basin alignment is applied before normalization:

  ARM A (raw):     scrambled-coordinate weights  -> normalize -> diffusion -> eval
  ARM B (aligned): canonicalized weights         -> normalize -> diffusion -> eval

Reuses run_phase1_experiment verbatim for both arms so the comparison is clean.
Reports dataset variance pre/post alignment, and zero-shot / FT100 Delta vs
random (mean +/- std over N seeds) for each arm.

GATE 0 (hard): alignment must be function-preserving (verified via a recurrence
rollout inside align_collection); summary records the max logit diff.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.align.rebasin import align_collection
from src.utils.weights import dataset_variance, load_raw_weights, normalize_weights
from experiments.phase1_lib import run_phase1_experiment


def _make_model_factory(hidden: int, depth: int):
    """Returns a factory that creates the right model for the given depth."""
    if depth <= 1:
        from src.models.tiny_mlp import TinyMLP
        return lambda: TinyMLP(hidden_dim=hidden)
    else:
        from src.arch_gen.spec import ArchSpec  # noqa: F401 — forces package init
        from src.models.variable_mlp import VariableTinyMLP
        from data.synthetic_text import VOCAB_SIZE
        spec = ArchSpec(hidden, depth, skip=False)
        return lambda: VariableTinyMLP(spec, vocab_size=VOCAB_SIZE)


def _agg(rows: list[dict], key: str) -> tuple[float, float]:
    vals = [r[key] for r in rows]
    mean = statistics.fmean(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def run_align_ablation(
    raw_path: str,
    *,
    hidden: int,
    depth: int = 1,
    norm: str = "perdim",
    method: str = "weight_match",
    align_iters: int = 3,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    diffusion_steps: int = 2000,
    timesteps: int = 200,
    num_samples: int = 30,
    ft_steps: tuple[int, ...] = (100,),
    out: str = "checkpoints/align",
    plot_dir: str = "checkpoints/plots",
) -> dict:
    rp = Path(raw_path)
    meta_path = rp.parent / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    if "layer_slices" not in meta:
        raise SystemExit(f"meta.json missing layer_slices at {meta_path}; re-run collect_weights.py")
    H = hidden or meta.get("hidden_dim", 64)
    d = depth or meta.get("depth", 1)
    slices = [tuple(x) for x in meta["layer_slices"]]

    model_factory = _make_model_factory(H, d)

    raw = load_raw_weights(rp)
    print(f"Loaded raw weights {tuple(raw.shape)} | hidden={H} | depth={d} | norm={norm} | method={method}")

    # --- alignment (GATE 0 enforced inside) ---
    aligned, info = align_collection(raw, meta, H, d, model_factory, method=method, iters=align_iters, verify=True)
    fn_diff = info["function_preservation_max_diff"]
    print(f"GATE 0 function-preservation max abs logit diff: {fn_diff:.3e} (must be < 1e-4)")

    var_raw = dataset_variance(raw)
    var_aln = dataset_variance(aligned)
    ratio = var_aln["total_var"] / max(var_raw["total_var"], 1e-12)
    print(f"dataset variance  raw={var_raw['total_var']:.4f}  aligned={var_aln['total_var']:.4f}  ratio={ratio:.3f}")

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, list[dict]] = {"raw": [], "aligned": []}
    for s in seeds:
        for arm, W in (("raw", raw), ("aligned", aligned)):
            norm_w, norm_meta = normalize_weights(W, mode=norm, layer_slices=slices)
            norm_meta = {"layer_slices": [list(x) for x in slices], **norm_meta}
            m = run_phase1_experiment(
                norm_w,
                norm_meta,
                tag=f"align_{arm}_s{s}",
                hidden=H,
                diffusion_steps=diffusion_steps,
                timesteps=timesteps,
                num_samples=num_samples,
                ft_steps=list(ft_steps),
                seed=s,
                plot_dir=plot_dir,
                model_factory=model_factory,
            )
            results[arm].append(m)
            print(
                f"  [{arm} s{s}] zero_delta={m['zero_shot_delta_vs_random']:+.4f} "
                + " ".join(f"ft{n}_delta={m[f'ft{n}_delta_vs_random']:+.4f}" for n in ft_steps if n > 0)
            )

    summary = {
        "raw_weights_path": str(rp),
        "hidden": H,
        "norm": norm,
        "method": method,
        "align_iters": align_iters,
        "n_seeds": len(seeds),
        "seeds": list(seeds),
        "diffusion_steps": diffusion_steps,
        "num_samples": num_samples,
        "ft_steps": list(ft_steps),
        "function_preservation_max_diff": fn_diff,
        "dataset_variance": {
            "raw": var_raw["total_var"],
            "aligned": var_aln["total_var"],
            "ratio": ratio,
            "raw_mean_pairwise_l2": var_raw["mean_pairwise_l2"],
            "aligned_mean_pairwise_l2": var_aln["mean_pairwise_l2"],
        },
    }
    for arm in ("raw", "aligned"):
        zm, zs = _agg(results[arm], "zero_shot_delta_vs_random")
        entry = {"zero_delta_mean": zm, "zero_delta_std": zs}
        for n in ft_steps:
            if n <= 0:
                continue
            fm, fsd = _agg(results[arm], f"ft{n}_delta_vs_random")
            entry[f"ft{n}_delta_mean"] = fm
            entry[f"ft{n}_delta_std"] = fsd
        summary[arm] = entry

    (out_dir / "align_ablation_summary.json").write_text(json.dumps(summary, indent=2))

    # --- plots ---
    _plot_variance(var_raw["total_var"], var_aln["total_var"], out_dir / "align_variance_collapse.png")
    _plot_deltas(summary, list(ft_steps), out_dir / "align_delta_comparison.png")

    print("\n=== ALIGNMENT ABLATION SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nArtifacts: {out_dir}/align_ablation_summary.json, "
          f"{out_dir}/align_variance_collapse.png, {out_dir}/align_delta_comparison.png")
    return summary


def _plot_variance(raw_v: float, aln_v: float, path: Path):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(["raw", "aligned"], [raw_v, aln_v], color=["#c44", "#4a8"])
    ax.set_ylabel("dataset total variance")
    ax.set_title(f"Variance collapse (ratio={aln_v / max(raw_v, 1e-12):.3f})")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_deltas(summary: dict, ft_steps: list[int], path: Path):
    metrics = ["zero"] + [f"ft{n}" for n in ft_steps if n > 0]
    labels = ["zero-shot"] + [f"FT{n}" for n in ft_steps if n > 0]
    raw_means = [summary["raw"]["zero_delta_mean"]] + [summary["raw"][f"ft{n}_delta_mean"] for n in ft_steps if n > 0]
    raw_stds = [summary["raw"]["zero_delta_std"]] + [summary["raw"][f"ft{n}_delta_std"] for n in ft_steps if n > 0]
    aln_means = [summary["aligned"]["zero_delta_mean"]] + [summary["aligned"][f"ft{n}_delta_mean"] for n in ft_steps if n > 0]
    aln_stds = [summary["aligned"]["zero_delta_std"]] + [summary["aligned"][f"ft{n}_delta_std"] for n in ft_steps if n > 0]

    import numpy as np

    x = np.arange(len(metrics))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - w / 2, raw_means, w, yerr=raw_stds, capsize=4, label="raw", color="#c44")
    ax.bar(x + w / 2, aln_means, w, yerr=aln_stds, capsize=4, label="aligned", color="#4a8")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Delta vs random (positive = generated better)")
    ax.set_title("Generated init quality: raw vs aligned")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Alignment ablation (raw vs Git Re-Basin aligned)")
    p.add_argument("--weights", type=str, required=True, help="weights_raw.pt (raw, un-normalized)")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--depth", type=int, default=1, help="Model depth (1=TinyMLP, >1=VariableTinyMLP)")
    p.add_argument("--norm", type=str, default="perdim", choices=["global", "layer", "perdim"])
    p.add_argument("--method", type=str, default="weight_match", choices=["weight_match", "canonical_sort"])
    p.add_argument("--align_iters", type=int, default=3)
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--diffusion_steps", type=int, default=2000)
    p.add_argument("--timesteps", type=int, default=200)
    p.add_argument("--num_samples", type=int, default=30)
    p.add_argument("--ft_steps", type=str, default="100")
    p.add_argument("--out", type=str, default="checkpoints/align")
    p.add_argument("--plot_dir", type=str, default="checkpoints/plots")
    args = p.parse_args()

    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip() != "")
    ft_steps = tuple(int(x) for x in args.ft_steps.split(",") if x.strip() != "")

    run_align_ablation(
        args.weights,
        hidden=args.hidden,
        depth=args.depth,
        norm=args.norm,
        method=args.method,
        align_iters=args.align_iters,
        seeds=seeds,
        diffusion_steps=args.diffusion_steps,
        timesteps=args.timesteps,
        num_samples=args.num_samples,
        ft_steps=ft_steps,
        out=args.out,
        plot_dir=args.plot_dir,
    )


if __name__ == "__main__":
    main()
