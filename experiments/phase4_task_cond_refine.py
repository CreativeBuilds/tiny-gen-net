#!/usr/bin/env python3
"""Phase 4 refine: expanded task vocab + match-aware loss vs phase4_v1."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from tqdm import tqdm

from experiments.phase2_jepa_vs_diffusion import metrics_from_rows
from src.arch_gen.conditioning import load_cond_checkpoint, normalize_multi_arch
from src.arch_gen.eval import eval_arch_row
from src.arch_gen.generator import ArchGenerator
from src.arch_gen.spec import ArchSpec, REF_DEPTH, REF_HIDDEN
from src.arch_gen.spec import ArchSpec as AS
from src.arch_gen.task_cond import DEFAULT_EVAL_TASKS, describe_spec, spec_matches_task
from src.arch_gen.task_generator import TaskArchGenerator, TaskArchTrainer
from src.arch_gen.task_weights import TaskCondWeightGenerator, TaskCondWeightTrainer, save_task_cond_ckpt
from src.arch_gen.weight_init import init_model_weights, init_task_weights
from src.models.variable_mlp import build_from_spec
from src.utils.device import device_str
from src.utils.logging import log_run
from src.utils.train_loop import set_seed
from src.utils.viz import plot_bar_comparison, plot_loss_curve, plot_scatter


def load_multi_arch(path: Path) -> tuple[list[ArchSpec], list[torch.Tensor]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    specs = [AS.from_label(lbl) for lbl in ckpt["meta"]["specs"]]
    weights = ckpt["weights"]
    if isinstance(weights, torch.Tensor): weights = [weights[i] for i in range(weights.shape[0])]
    return specs, weights


def main():
    p = argparse.ArgumentParser(description="Phase 4 task conditioning refine")
    p.add_argument("--multi_arch", default="checkpoints/weights/multi_arch_v2/weights_raw.pt")
    p.add_argument("--arch_train_steps", type=int, default=1200)
    p.add_argument("--weight_train_steps", type=int, default=2000)
    p.add_argument("--match_weight", type=float, default=0.5)
    p.add_argument("--weight_match_weight", type=float, default=0.3)
    p.add_argument("--num_tasks", type=int, default=25)
    p.add_argument("--ft_steps", default="100")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot_dir", default="checkpoints/plots/phase4_refine")
    p.add_argument("--tag", default="phase4_v2")
    args = p.parse_args()
    ft_steps = [int(x) for x in args.ft_steps.split(",") if x.strip()]
    tasks = DEFAULT_EVAL_TASKS[: args.num_tasks]

    set_seed(args.seed)
    device = device_str()
    specs, raw_w = load_multi_arch(Path(args.multi_arch))
    texts = [describe_spec(s) for s in specs]
    norm_w, norm_meta = normalize_multi_arch(raw_w)
    plot_dir = Path(args.plot_dir); plot_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("checkpoints/arch_gen"); ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"Phase 4 refine | match_w={args.match_weight} | tasks={len(tasks)} | device={device}")
    arch_gen = TaskArchGenerator()
    w_gen = TaskCondWeightGenerator()
    arch_losses = TaskArchTrainer(arch_gen, device=device, match_weight=args.match_weight).fit_paired(specs, texts, steps=args.arch_train_steps)
    w_losses = TaskCondWeightTrainer(w_gen, device=device, match_weight=args.weight_match_weight).fit(specs, texts, norm_w, steps=args.weight_train_steps)
    save_task_cond_ckpt(str(ckpt_dir / f"task_cond_{args.tag}.pt"), arch_gen, w_gen, norm_meta)
    plot_loss_curve(arch_losses, plot_dir / f"{args.tag}_arch_train.png", title="Task arch (match-aware)")
    plot_loss_curve(w_losses[:: max(1, len(w_losses) // 500)], plot_dir / f"{args.tag}_weight_train.png", title="Task weights (match-aware)")

    uncond_arch = ArchGenerator().to(device)
    uncond_arch.load_state_dict(torch.load(ckpt_dir / "arch_generator_phase3_refine_v1.pt", map_location=device, weights_only=True))
    p3, p3_norm = load_cond_checkpoint(str(ckpt_dir / "cond_weights_phase3_refine_v1.pt"), device)

    task_rows, uncond_rows, match_scores, examples = [], [], [], []
    for i, task in enumerate(tqdm(tasks, desc="eval")):
        spec_t = arch_gen.sample(1, task, device)[0]
        spec_u = uncond_arch.sample(1, device)[0]
        mt = build_from_spec(spec_t).to(device)
        mu = build_from_spec(spec_u).to(device)
        init_task_weights(mt, spec_t, w_gen, task, norm_meta, device)
        init_model_weights(mu, spec_u, cond_gen=p3, cond_norm=p3_norm, device=device, method="conditioned", noise_scale=0.5)
        task_rows.append(eval_arch_row(mt, device, ft_steps, args.seed + i))
        uncond_rows.append(eval_arch_row(mu, device, ft_steps, args.seed + 5000 + i))
        ms = spec_matches_task(spec_t, task)
        match_scores.append(ms)
        examples.append({"task": task, "spec": [spec_t.hidden_dim, spec_t.depth, spec_t.skip], "match": ms})

    rnd = []
    ref = ArchSpec(REF_HIDDEN, REF_DEPTH)
    for i in range(len(tasks)):
        m = build_from_spec(ref).to(device)
        init_model_weights(m, ref, method="random")
        set_seed(1000 + i)
        rnd.append(eval_arch_row(m, device, ft_steps, args.seed + i))

    task_m = metrics_from_rows(task_rows, rnd, ft_steps, "task")
    uncond_m = metrics_from_rows(uncond_rows, rnd, ft_steps, "uncond")
    v1_path = ckpt_dir / "metrics_phase4_v1.json"
    v1 = json.loads(v1_path.read_text()) if v1_path.exists() else {}

    metrics = {"tag": args.tag, "num_tasks": len(tasks), "match_weight": args.match_weight,
               "avg_task_match": sum(match_scores) / len(match_scores), "examples": examples,
               "prev_v1_match": v1.get("avg_task_match"), "prev_v1_zero": v1.get("task_zero_loss"), **task_m, **uncond_m}
    metrics["match_gain_vs_v1"] = metrics["avg_task_match"] - (v1.get("avg_task_match") or 0)
    metrics["zero_gain_vs_v1"] = (v1.get("task_zero_loss") or 0) - task_m["task_zero_loss"]

    plot_bar_comparison(["v2_task", "v1_task", "uncond", "rnd"],
        [task_m["task_zero_loss"], v1.get("task_zero_loss", 0), uncond_m["uncond_zero_loss"], task_m["random_zero_loss"]],
        plot_dir / f"{args.tag}_zero_compare.png", title="Zero-shot v2 vs v1", ylabel="Loss")
    plot_bar_comparison(["v2_match", "v1_match"],
        [metrics["avg_task_match"], v1.get("avg_task_match", 0)],
        plot_dir / f"{args.tag}_match_compare.png", title="Steering quality", ylabel="Match score")
    plot_scatter(match_scores, [r["zero_loss"] for r in task_rows], plot_dir / f"{args.tag}_match_vs_loss.png",
        title="Match vs zero loss", xlabel="Match", ylabel="Loss")

    out = ckpt_dir / f"metrics_{args.tag}.json"
    out.write_text(json.dumps(metrics, indent=2))
    log_run("checkpoints/logs", f"phase4_{args.tag}", metrics, vars(args))

    print(f"\n=== Phase 4 Refine [{args.tag}] ===")
    print(f"  match: v2={metrics['avg_task_match']:.3f} v1={v1.get('avg_task_match')} gain={metrics['match_gain_vs_v1']:.3f}")
    print(f"  zero: v2={task_m['task_zero_loss']:.4f} v1={v1.get('task_zero_loss')} Δ={task_m['task_zero_delta_vs_random']:.4f}")
    for n in ft_steps:
        if n <= 0: continue
        print(f"  ft{n}: {task_m[f'task_ft{n}_loss']:.4f}")


if __name__ == "__main__":
    main()
