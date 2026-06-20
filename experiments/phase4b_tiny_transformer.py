#!/usr/bin/env python3
"""Phase 4b: tiny transformer scale-up — collect, cond weights, task steering, eval."""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from tqdm import tqdm

from experiments.phase2_jepa_vs_diffusion import metrics_from_rows
from src.arch_gen.eval import eval_arch_row
from src.arch_gen.spec import ArchSpec, REF_DEPTH, REF_HIDDEN
from src.arch_gen.tx_conditioning import TxCondWeightGenerator, TxCondWeightTrainer, init_tx_weights, load_tx_cond_ckpt, normalize_tx_weights, save_tx_cond_ckpt
from src.arch_gen.tx_eval import eval_tx_row, finetune_tx
from src.arch_gen.tx_spec import REF_D_MODEL, REF_N_HEAD, REF_N_LAYER, TxSpec, all_valid_tx_specs, pick_diverse_tx_specs
from src.arch_gen.tx_task_cond import DEFAULT_TX_EVAL_TASKS, describe_tx_spec, tx_spec_matches_task
from src.arch_gen.tx_task_generator import TaskTxArchGenerator, TaskTxArchTrainer
from src.arch_gen.tx_task_weights import TaskTxWeightGenerator, TaskTxWeightTrainer, init_task_tx_weights, save_task_tx_ckpt
from src.arch_gen.weight_init import init_model_weights
from src.models.variable_mlp import build_from_spec
from src.models.variable_transformer import build_from_tx_spec
from src.utils.device import device_str
from src.utils.logging import log_run
from src.utils.train_loop import set_seed
from src.utils.viz import plot_bar_comparison, plot_loss_curve, plot_scatter
from src.utils.weights import flatten_state_dict


def collect_tx_weights(specs: list[TxSpec], steps: int, seed: int, device: str) -> list[torch.Tensor]:
    weights = []
    for i, spec in enumerate(tqdm(specs, desc="collect")):
        m = build_from_tx_spec(spec).to(device)
        finetune_tx(m, steps, seed=seed + i * 17, device=device)
        weights.append(flatten_state_dict(m))
    return weights


def main():
    p = argparse.ArgumentParser(description="Phase 4b tiny transformer scale-up")
    p.add_argument("--num_collect", type=int, default=30)
    p.add_argument("--train_steps", type=int, default=350)
    p.add_argument("--cond_steps", type=int, default=1500)
    p.add_argument("--arch_steps", type=int, default=800)
    p.add_argument("--weight_steps", type=int, default=1200)
    p.add_argument("--match_weight", type=float, default=0.5)
    p.add_argument("--num_tasks", type=int, default=25)
    p.add_argument("--ft_steps", default="100")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot_dir", default="checkpoints/plots/phase4b_tx")
    p.add_argument("--tag", default="phase4b_v1")
    p.add_argument("--skip_collect", action="store_true")
    args = p.parse_args()
    ft_steps = [int(x) for x in args.ft_steps.split(",") if x.strip()]
    tasks = DEFAULT_TX_EVAL_TASKS[: args.num_tasks]

    set_seed(args.seed)
    device = device_str()
    plot_dir = Path(args.plot_dir); plot_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("checkpoints/arch_gen"); ckpt_dir.mkdir(parents=True, exist_ok=True)
    wdir = Path("checkpoints/weights/tx_multi_v1"); wdir.mkdir(parents=True, exist_ok=True)
    raw_path = wdir / "weights_raw.pt"

    collect_specs = pick_diverse_tx_specs(args.num_collect, random.Random(args.seed))
    if raw_path.exists() and args.skip_collect:
        ckpt = torch.load(raw_path, map_location="cpu", weights_only=True)
        collect_specs = [TxSpec.from_label(lbl) for lbl in ckpt["meta"]["specs"]]
        raw_w = ckpt["weights"]
        if isinstance(raw_w, torch.Tensor): raw_w = [raw_w[i] for i in range(raw_w.shape[0])]
    else:
        raw_w = collect_tx_weights(collect_specs, args.train_steps, args.seed, device)
        torch.save({"weights": raw_w, "meta": {"specs": [[s.d_model, s.n_layer, s.n_head, s.ctx_len] for s in collect_specs],
            "train_steps": args.train_steps, "grid_size": len(all_valid_tx_specs())}}, raw_path)

    texts = [describe_tx_spec(s) for s in collect_specs]
    norm_w, norm_meta = normalize_tx_weights(raw_w)
    print(f"Phase 4b | collect={len(collect_specs)} grid={len(all_valid_tx_specs())} tasks={len(tasks)} device={device}")

    cond = TxCondWeightGenerator()
    cond_losses = TxCondWeightTrainer(cond, device=device).fit(collect_specs, norm_w, steps=args.cond_steps)
    save_tx_cond_ckpt(str(ckpt_dir / f"tx_cond_{args.tag}.pt"), cond, norm_meta)

    arch_gen = TaskTxArchGenerator()
    w_gen = TaskTxWeightGenerator()
    arch_losses = TaskTxArchTrainer(arch_gen, device=device, match_weight=args.match_weight).fit_paired(collect_specs, texts, steps=args.arch_steps)
    w_losses = TaskTxWeightTrainer(w_gen, device=device, match_weight=0.3).fit(collect_specs, texts, norm_w, steps=args.weight_steps)
    save_task_tx_ckpt(str(ckpt_dir / f"task_tx_{args.tag}.pt"), arch_gen, w_gen, norm_meta)

    plot_loss_curve(cond_losses[:: max(1, len(cond_losses) // 300)], plot_dir / f"{args.tag}_cond.png", title="Tx cond weights")
    plot_loss_curve(arch_losses, plot_dir / f"{args.tag}_arch.png", title="Task tx arch")

    task_rows, cond_rows, rnd_rows, match_scores, examples = [], [], [], [], []
    ref_tx = TxSpec(REF_D_MODEL, REF_N_LAYER, REF_N_HEAD)
    ref_mlp = ArchSpec(REF_HIDDEN, REF_DEPTH)

    for i, task in enumerate(tqdm(tasks, desc="eval")):
        spec_t = arch_gen.sample(1, task, device)[0]
        mt = build_from_tx_spec(spec_t).to(device)
        init_task_tx_weights(mt, spec_t, w_gen, task, norm_meta, device)
        task_rows.append(eval_tx_row(mt, device, ft_steps, args.seed + i))
        ms = tx_spec_matches_task(spec_t, task)
        match_scores.append(ms)
        examples.append({"task": task, "spec": [spec_t.d_model, spec_t.n_layer, spec_t.n_head], "match": ms})

        spec_c = collect_specs[i % len(collect_specs)]
        mc = build_from_tx_spec(spec_c).to(device)
        init_tx_weights(mc, spec_c, cond, norm_meta, device)
        cond_rows.append(eval_tx_row(mc, device, ft_steps, args.seed + 5000 + i))

        mr = build_from_tx_spec(ref_tx).to(device)
        finetune_tx(mr, 0, seed=9000 + i, device=device)
        rnd_rows.append(eval_tx_row(mr, device, ft_steps, args.seed + 9000 + i))

    task_m = metrics_from_rows(task_rows, rnd_rows, ft_steps, "task_tx")
    cond_m = metrics_from_rows(cond_rows, rnd_rows, ft_steps, "tx_cond")

    mlp_m = build_from_spec(ref_mlp).to(device)
    init_model_weights(mlp_m, ref_mlp, method="random")
    mlp_rnd = eval_arch_row(mlp_m, device, ft_steps, args.seed + 777)

    p4 = json.loads((ckpt_dir / "metrics_phase4_v2.json").read_text()) if (ckpt_dir / "metrics_phase4_v2.json").exists() else {}

    metrics = {"tag": args.tag, "num_collect": len(collect_specs), "num_tasks": len(tasks), "grid_size": len(all_valid_tx_specs()),
               "avg_task_match": sum(match_scores) / len(match_scores), "examples": examples,
               "ref_mlp_zero": mlp_rnd["zero_loss"], "phase4_mlp_task_zero": p4.get("task_zero_loss"), **task_m, **cond_m}
    plot_bar_comparison(["task_tx", "tx_cond", "rnd_tx", "mlp_rnd"],
        [task_m["task_tx_zero_loss"], cond_m["tx_cond_zero_loss"], task_m["random_zero_loss"], mlp_rnd["zero_loss"]],
        plot_dir / f"{args.tag}_zero.png", title="Zero-shot: transformer vs MLP random", ylabel="Loss")
    plot_bar_comparison(["tx_match", "mlp_match"],
        [metrics["avg_task_match"], p4.get("avg_task_match", 0)],
        plot_dir / f"{args.tag}_match.png", title="Steering: tx vs MLP", ylabel="Match")
    plot_scatter(match_scores, [r["zero_loss"] for r in task_rows], plot_dir / f"{args.tag}_match_loss.png",
        title="Tx match vs zero loss", xlabel="Match", ylabel="Loss")

    out = ckpt_dir / f"metrics_{args.tag}.json"
    out.write_text(json.dumps(metrics, indent=2))
    log_run("checkpoints/logs", f"phase4b_{args.tag}", metrics, vars(args))

    print(f"\n=== Phase 4b [{args.tag}] ===")
    print(f"  match={metrics['avg_task_match']:.3f} (mlp v2={p4.get('avg_task_match')})")
    print(f"  task_tx zero={task_m['task_tx_zero_loss']:.4f} Δ={task_m['task_tx_zero_delta_vs_random']:.4f}")
    print(f"  tx_cond zero={cond_m['tx_cond_zero_loss']:.4f} rnd_tx={task_m['random_zero_loss']:.4f} mlp_rnd={mlp_rnd['zero_loss']:.4f}")
    for n in ft_steps:
        if n <= 0: continue
        print(f"  ft{n}: task_tx={task_m[f'task_tx_ft{n}_loss']:.4f}")


if __name__ == "__main__":
    main()
