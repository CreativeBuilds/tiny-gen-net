#!/usr/bin/env python3
"""Phase 4: task text -> architecture + weights; compare vs Phase 3 unconditioned."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from tqdm import tqdm

from experiments.phase2_jepa_vs_diffusion import metrics_from_rows
from src.arch_gen.conditioning import normalize_multi_arch
from src.arch_gen.eval import eval_arch_row
from src.arch_gen.generator import ArchGenerator
from src.arch_gen.spec import ArchSpec, REF_DEPTH, REF_HIDDEN
from src.arch_gen.task_cond import DEFAULT_EVAL_TASKS, describe_spec, spec_matches_task
from src.arch_gen.task_generator import TaskArchGenerator, TaskArchTrainer
from src.arch_gen.task_weights import TaskCondWeightGenerator, TaskCondWeightTrainer, save_task_cond_ckpt
from src.arch_gen.weight_init import init_model_weights, init_task_weights
from src.models.variable_mlp import build_from_spec
from src.utils.device import device_str
from src.utils.logging import log_run
from src.utils.train_loop import set_seed
from src.utils.viz import plot_bar_comparison, plot_loss_curve, plot_scatter
from src.arch_gen.spec import ArchSpec as AS


def load_multi_arch(path: Path) -> tuple[list[ArchSpec], list[torch.Tensor]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    specs = [AS.from_label(lbl) for lbl in ckpt["meta"]["specs"]]
    weights = ckpt["weights"]
    if isinstance(weights, torch.Tensor): weights = [weights[i] for i in range(weights.shape[0])]
    return specs, weights


def main():
    p = argparse.ArgumentParser(description="Phase 4 task conditioning")
    p.add_argument("--multi_arch", default="checkpoints/weights/multi_arch_v2/weights_raw.pt")
    p.add_argument("--arch_train_steps", type=int, default=800)
    p.add_argument("--weight_train_steps", type=int, default=2000)
    p.add_argument("--num_tasks", type=int, default=10)
    p.add_argument("--ft_steps", default="100")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot_dir", default="checkpoints/plots/phase4_task")
    p.add_argument("--tag", default="phase4_v1")
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

    print(f"Phase 4 task cond | N={len(specs)} | tasks={len(tasks)} | device={device}")
    arch_gen = TaskArchGenerator()
    w_gen = TaskCondWeightGenerator()
    arch_losses = TaskArchTrainer(arch_gen, device=device).fit_paired(specs, texts, steps=args.arch_train_steps)
    w_losses = TaskCondWeightTrainer(w_gen, device=device).fit(specs, texts, norm_w, steps=args.weight_train_steps)
    save_task_cond_ckpt(str(ckpt_dir / f"task_cond_{args.tag}.pt"), arch_gen, w_gen, norm_meta)
    plot_loss_curve(arch_losses, plot_dir / f"{args.tag}_arch_train.png", title="Task arch generator")
    plot_loss_curve(w_losses[:: max(1, len(w_losses) // 500)], plot_dir / f"{args.tag}_weight_train.png", title="Task weight generator")

    # Phase 3 uncond baselines (frozen arch gen without task)
    uncond_arch = ArchGenerator().to(device)
    uncond_arch.load_state_dict(torch.load(ckpt_dir / "arch_generator_phase3_refine_v1.pt", map_location=device, weights_only=True))

    task_rows, uncond_rows, match_scores, examples = [], [], [], []
    from src.arch_gen.conditioning import load_cond_checkpoint
    p3, p3_norm = load_cond_checkpoint(str(ckpt_dir / "cond_weights_phase3_refine_v1.pt"), device)

    for i, task in enumerate(tqdm(tasks, desc="task eval")):
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
        examples.append({"task": task, "spec": [spec_t.hidden_dim, spec_t.depth, spec_t.skip], "match": ms,
                         "uncond_spec": [spec_u.hidden_dim, spec_u.depth, spec_u.skip]})

    rnd = []
    ref = ArchSpec(REF_HIDDEN, REF_DEPTH)
    for i in range(len(tasks)):
        m = build_from_spec(ref).to(device)
        init_model_weights(m, ref, method="random")
        set_seed(1000 + i)
        rnd.append(eval_arch_row(m, device, ft_steps, args.seed + i))

    task_m = metrics_from_rows(task_rows, rnd, ft_steps, "task")
    uncond_m = metrics_from_rows(uncond_rows, rnd, ft_steps, "uncond")

    v3 = json.loads((ckpt_dir / "metrics_phase3_refine_v1.json").read_text()) if (ckpt_dir / "metrics_phase3_refine_v1.json").exists() else {}

    metrics = {"tag": args.tag, "num_tasks": len(tasks), "avg_task_match": sum(match_scores) / len(match_scores),
               "examples": examples, "prev_phase3_zero": v3.get("cond_zero_loss"), **task_m, **uncond_m}
    metrics["task_minus_uncond_zero"] = uncond_m["uncond_zero_loss"] - task_m["task_zero_loss"]

    plot_bar_comparison(["task_cond", "uncond_p3", "phase3_ref", "rnd"],
        [task_m["task_zero_loss"], uncond_m["uncond_zero_loss"], v3.get("cond_zero_loss", 0), task_m["random_zero_loss"]],
        plot_dir / f"{args.tag}_zero_compare.png", title="Zero-shot: task vs uncond", ylabel="Loss")
    plot_scatter(match_scores, [r["zero_loss"] for r in task_rows], plot_dir / f"{args.tag}_match_vs_loss.png",
        title="Task-spec match vs zero loss", xlabel="Match score", ylabel="Zero loss")

    (plot_dir / f"{args.tag}_examples.json").write_text(json.dumps(examples, indent=2))
    out = ckpt_dir / f"metrics_{args.tag}.json"
    out.write_text(json.dumps(metrics, indent=2))
    log_run("checkpoints/logs", f"phase4_{args.tag}", metrics, vars(args))

    print(f"\n=== Phase 4 [{args.tag}] ===")
    print(f"  task zero={task_m['task_zero_loss']:.4f} uncond={uncond_m['uncond_zero_loss']:.4f} phase3_ref={v3.get('cond_zero_loss')}")
    print(f"  task Δ={task_m['task_zero_delta_vs_random']:.4f} avg arch match={metrics['avg_task_match']:.2f}")
    for ex in examples[:3]: print(f"  e.g. {ex['task'][:40]} -> h={ex['spec'][0]} d={ex['spec'][1]} skip={ex['spec'][2]} match={ex['match']:.2f}")


if __name__ == "__main__":
    main()
