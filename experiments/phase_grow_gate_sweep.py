#!/usr/bin/env python3
"""GATE-GROW sweep: run width-growth gate across multiple FT step counts.

Produces the headline convergence figure: CE vs FLOPs for each arm.
Answers: does the correction advantage persist or wash out with more training?
"""

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def run_single_gate(ft_steps, args, device):
    """Run one GATE-GROW with a specific FT step count."""
    out_dir = f"{args.out_base}_ft{ft_steps}"
    cmd = [
        sys.executable, str(ROOT / "experiments" / "phase_grow_gate.py"),
        "--source_hidden", str(args.source_hidden),
        "--target_hidden", str(args.target_hidden),
        "--depth", str(args.depth),
        "--source_steps", str(args.source_steps),
        "--finetune_steps", str(ft_steps),
        "--rank", str(args.rank),
        "--corr_scale", str(args.corr_scale),
        "--rand_corr_scale", str(args.rand_corr_scale),
        "--seeds", args.seeds,
        "--out", out_dir,
    ]
    print(f"\n{'='*60}")
    print(f"Running GATE-GROW with FT={ft_steps}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(ROOT), capture_output=False)
    if result.returncode != 0:
        print(f"ERROR: GATE-GROW FT={ft_steps} failed with code {result.returncode}")
        return None, out_dir

    summary_path = Path(out_dir) / "grow_gate_width_summary.json"
    if not summary_path.exists():
        print(f"ERROR: summary not found at {summary_path}")
        return None, out_dir
    return json.loads(summary_path.read_text()), out_dir


def aggregate(summaries, ft_steps_list, out_dir):
    """Aggregate all runs into convergence plot + FLOP-savings analysis."""
    arms = ["random", "naive_grow", "grown_corr", "grown_rand_corr", "source_only"]
    colors = {"random": "#c44", "naive_grow": "#888", "grown_corr": "#4a8",
              "grown_rand_corr": "#48a", "source_only": "#c84"}
    labels = {"random": "Random init", "naive_grow": "Naive growth",
              "grown_corr": "Growth + correction", "grown_rand_corr": "Growth + random corr.",
              "source_only": "Source only (equal FLOP)"}

    # Collect CE means and stds per arm per FT step
    data = {arm: {"means": [], "stds": []} for arm in arms}
    for ft, s in zip(ft_steps_list, summaries):
        if s is None:
            for arm in arms:
                data[arm]["means"].append(None)
                data[arm]["stds"].append(None)
            continue
        for arm in arms:
            data[arm]["means"].append(s[arm]["ce_mean"])
            data[arm]["stds"].append(s[arm]["ce_std"])

    # --- Convergence plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for arm in arms:
        means = data[arm]["means"]
        stds = data[arm]["stds"]
        # Filter out None values
        xs = [ft for ft, m in zip(ft_steps_list, means) if m is not None]
        ys = [m for m in means if m is not None]
        es = [s for s in stds if s is not None]
        ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=labels[arm],
                    color=colors[arm], linewidth=2, markersize=6)

    ax.set_xlabel("Fine-tune steps (equal FLOP budget)", fontsize=12)
    ax.set_ylabel("Eval CE (lower = better)", fontsize=12)
    ax.set_title("Growth + correction vs baselines: convergence vs compute", fontsize=14)
    ax.legend(fontsize=10, loc="best")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = Path(out_dir) / "grow_gate_sweep_convergence.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Convergence plot: {plot_path}")

    # --- FLOP-savings analysis ---
    # Find grown_corr CE at FT=500
    ref_idx = ft_steps_list.index(500) if 500 in ft_steps_list else len(ft_steps_list) // 2
    grown_corr_ce_500 = summaries[ref_idx]["grown_corr"]["ce_mean"] if summaries[ref_idx] else None

    # Find at what FT step random reaches that CE
    flop_savings = None
    if grown_corr_ce_500 is not None:
        for i, ft in enumerate(ft_steps_list):
            if summaries[i] is None:
                continue
            random_ce = summaries[i]["random"]["ce_mean"]
            if random_ce <= grown_corr_ce_500:
                flop_savings = ft / 500.0
                break

    # --- Gap analysis at longest FT ---
    last_valid = None
    for i in range(len(summaries) - 1, -1, -1):
        if summaries[i] is not None:
            last_valid = i
            break

    gap_long = None
    verdict = "UNKNOWN"
    if last_valid is not None:
        gap_long = summaries[last_valid]["naive_grow"]["ce_mean"] - summaries[last_valid]["grown_corr"]["ce_mean"]
        if gap_long > 0.05:
            verdict = "PERSISTS"
        elif gap_long < 0.01:
            verdict = "WASHES_OUT"
        else:
            verdict = "PARTIAL"

    # --- Summary JSON ---
    summary = {
        "ft_steps_list": ft_steps_list,
        "ref_ft": 500,
        "grown_corr_ce_at_500": grown_corr_ce_500,
        "flop_savings": flop_savings,
        "flop_savings_str": f"{flop_savings:.1f}x" if flop_savings else "random never reaches grown_corr@500",
        "gap_at_longest_ft": gap_long,
        "longest_ft": ft_steps_list[last_valid] if last_valid is not None else None,
        "verdict": verdict,
        "per_run": {},
    }
    for ft, s in zip(ft_steps_list, summaries):
        if s is None:
            continue
        summary["per_run"][str(ft)] = {
            arm: {"ce_mean": s[arm]["ce_mean"], "ce_std": s[arm]["ce_std"],
                  "delta_vs_random": s[arm]["delta_vs_random"]}
            for arm in arms
        }
        # Key gap
        summary["per_run"][str(ft)]["gap_corr_vs_naive"] = (
            s["naive_grow"]["ce_mean"] - s["grown_corr"]["ce_mean"]
        )

    return summary, plot_path


def main():
    p = argparse.ArgumentParser(description="GATE-GROW sweep across FT step counts")
    p.add_argument("--source_hidden", type=int, default=16)
    p.add_argument("--target_hidden", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--source_steps", type=int, default=2000)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--corr_scale", type=float, default=0.05)
    p.add_argument("--rand_corr_scale", type=float, default=0.1)
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--ft_steps", type=str, default="100,250,500,1000,2000,5000")
    p.add_argument("--out_base", type=str, default="checkpoints/align/grow_gate_width")
    p.add_argument("--out", type=str, default="checkpoints/align/grow_gate_sweep")
    args = p.parse_args()

    ft_steps_list = [int(x) for x in args.ft_steps.split(",")]
    print(f"GATE-GROW sweep: FT steps = {ft_steps_list}")
    print(f"Seeds: {args.seeds} | Source: H={args.source_hidden} -> H={args.target_hidden}")

    summaries = []
    for ft in ft_steps_list:
        s, out_dir = run_single_gate(ft, args, device=None)
        summaries.append(s)

    # Aggregate
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary, plot_path = aggregate(summaries, ft_steps_list, out_dir)
    (out_dir / "grow_gate_sweep_summary.json").write_text(json.dumps(summary, indent=2))

    # Print results
    print(f"\n{'='*70}")
    print(f"GATE-GROW SWEEP RESULTS")
    print(f"{'='*70}")
    print(f"\n{'FT':>6s}  {'random':>8s} {'naive':>8s} {'corr':>8s} {'rand_c':>8s} {'src_only':>8s}  {'gap(c-n)':>9s}")
    for ft, s in zip(ft_steps_list, summaries):
        if s is None:
            print(f"{ft:>6d}  FAILED")
            continue
        print(f"{ft:>6d}  {s['random']['ce_mean']:>8.4f} {s['naive_grow']['ce_mean']:>8.4f} "
              f"{s['grown_corr']['ce_mean']:>8.4f} {s['grown_rand_corr']['ce_mean']:>8.4f} "
              f"{s['source_only']['ce_mean']:>8.4f}  "
              f"{s['naive_grow']['ce_mean'] - s['grown_corr']['ce_mean']:>+9.4f}")

    print(f"\nVerdict: {summary['verdict']}")
    print(f"Gap at longest FT ({summary['longest_ft']}): {summary['gap_at_longest_ft']:+.4f}")
    print(f"FLOP savings: {summary['flop_savings_str']}")
    print(f"Plot: {plot_path}")
    print(f"Summary: {out_dir}/grow_gate_sweep_summary.json")


if __name__ == "__main__":
    main()
