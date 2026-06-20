#!/usr/bin/env python3
"""Phase 4b refine: improved tx steering + fixed-ref JEPA/hybrid/cond init test."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from tqdm import tqdm

from experiments.phase1_lib import avg_metric
from experiments.phase2_jepa_vs_diffusion import metrics_from_rows
from src.arch_gen.tx_conditioning import TxCondWeightGenerator, init_tx_weights, load_tx_cond_ckpt, normalize_tx_weights
from src.arch_gen.tx_eval import eval_tx_row
from src.arch_gen.tx_ref_init import collect_ref_tx_weights, eval_ref_tx_inits, train_ref_tx_diffusion, train_ref_tx_jepa
from src.arch_gen.tx_spec import REF_D_MODEL, REF_N_HEAD, REF_N_LAYER, TxSpec
from src.arch_gen.tx_task_cond import DEFAULT_TX_EVAL_TASKS, augment_tx_training_pairs, describe_tx_spec, enrich_tx_task, tx_spec_matches_task
from src.arch_gen.tx_task_generator import TaskTxArchGenerator, TaskTxArchTrainer
from src.arch_gen.tx_task_weights import TaskTxWeightGenerator, TaskTxWeightTrainer, init_task_tx_weights, load_task_tx_ckpt, save_task_tx_ckpt
from src.hybrid.jepa_diffusion import JEPADiffusionHybrid
from src.models.variable_transformer import build_from_tx_spec
from src.utils.device import device_str
from src.utils.logging import log_run
from src.utils.train_loop import set_seed
from src.utils.viz import plot_bar_comparison, plot_loss_curve, plot_scatter
from src.utils.weights import denormalize_weights, flatten_state_dict


def load_collect(path: Path) -> tuple[list[TxSpec], list[torch.Tensor]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    specs = [TxSpec.from_label(lbl) for lbl in ckpt["meta"]["specs"]]
    w = ckpt["weights"]
    if isinstance(w, torch.Tensor): w = [w[i] for i in range(w.shape[0])]
    return specs, w


def main():
    p = argparse.ArgumentParser(description="Phase 4b tx refine + fixed-ref init test")
    p.add_argument("--collect_path", default="checkpoints/weights/tx_multi_v1/weights_raw.pt")
    p.add_argument("--arch_steps", type=int, default=1200)
    p.add_argument("--weight_steps", type=int, default=1500)
    p.add_argument("--match_weight", type=float, default=0.7)
    p.add_argument("--num_tasks", type=int, default=25)
    p.add_argument("--ref_samples", type=int, default=20)
    p.add_argument("--ref_collect", type=int, default=30)
    p.add_argument("--ref_train_steps", type=int, default=350)
    p.add_argument("--jepa_steps", type=int, default=1200)
    p.add_argument("--diff_steps", type=int, default=800)
    p.add_argument("--ft_steps", default="100")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot_dir", default="checkpoints/plots/phase4b_refine")
    p.add_argument("--tag", default="phase4b_v2")
    p.add_argument("--skip_ref_collect", action="store_true")
    args = p.parse_args()
    ft_steps = [int(x) for x in args.ft_steps.split(",") if x.strip()]
    tasks = DEFAULT_TX_EVAL_TASKS[: args.num_tasks]
    ref_spec = TxSpec(REF_D_MODEL, REF_N_LAYER, REF_N_HEAD)

    set_seed(args.seed)
    device = device_str()
    plot_dir = Path(args.plot_dir); plot_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("checkpoints/arch_gen"); ckpt_dir.mkdir(parents=True, exist_ok=True)
    ref_wpath = Path("checkpoints/weights/tx_ref_v1/weights_raw.pt")

    collect_specs, raw_w = load_collect(Path(args.collect_path))
    texts_base = [describe_tx_spec(s) for s in collect_specs]
    train_specs, train_texts = augment_tx_training_pairs(collect_specs, texts_base)
    norm_w, norm_meta = normalize_tx_weights(raw_w)
    print(f"Phase 4b refine | train pairs={len(train_texts)} tasks={len(tasks)} ref={ref_spec.d_model}/{ref_spec.n_layer}/{ref_spec.n_head} device={device}")

    arch_gen = TaskTxArchGenerator()
    w_gen = TaskTxWeightGenerator()
    arch_losses = TaskTxArchTrainer(arch_gen, device=device, match_weight=args.match_weight).fit_paired(train_specs, train_texts, steps=args.arch_steps)
    weight_by = {(s.d_model, s.n_layer, s.n_head): w for s, w in zip(collect_specs, norm_w)}
    aug_weights = [weight_by[(s.d_model, s.n_layer, s.n_head)] for s in train_specs]
    w_losses = TaskTxWeightTrainer(w_gen, device=device, match_weight=0.4).fit(train_specs, train_texts, aug_weights, steps=args.weight_steps)
    save_task_tx_ckpt(str(ckpt_dir / f"task_tx_{args.tag}.pt"), arch_gen, w_gen, norm_meta)

    v1 = json.loads((ckpt_dir / "metrics_phase4b_v1.json").read_text()) if (ckpt_dir / "metrics_phase4b_v1.json").exists() else {}
    task_rows, match_scores, examples = [], [], []
    for i, task in enumerate(tqdm(tasks, desc="steering eval")):
        spec_t = arch_gen.sample(1, task, device)[0]
        mt = build_from_tx_spec(spec_t).to(device)
        init_task_tx_weights(mt, spec_t, w_gen, task, norm_meta, device)
        task_rows.append(eval_tx_row(mt, device, ft_steps, args.seed + i))
        ms = tx_spec_matches_task(spec_t, task)
        match_scores.append(ms)
        examples.append({"task": task, "spec": [spec_t.d_model, spec_t.n_layer, spec_t.n_head], "match": ms})

    rnd_rows_task = []
    for i in range(len(tasks)):
        mr = build_from_tx_spec(ref_spec).to(device)
        rnd_rows_task.append(eval_tx_row(mr, device, ft_steps, args.seed + 9000 + i))
    task_m = metrics_from_rows(task_rows, rnd_rows_task, ft_steps, "steer")
    avg_match = sum(match_scores) / len(match_scores)

    if ref_wpath.exists() and args.skip_ref_collect:
        ref_ckpt = torch.load(ref_wpath, map_location="cpu", weights_only=True)
        ref_stack = torch.stack(ref_ckpt["weights"]) if isinstance(ref_ckpt["weights"], list) else ref_ckpt["weights"]
    else:
        ref_list = collect_ref_tx_weights(ref_spec, args.ref_collect, args.ref_train_steps, args.seed, device)
        ref_wpath.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"weights": ref_list, "meta": {"spec": [ref_spec.d_model, ref_spec.n_layer, ref_spec.n_head]}}, ref_wpath)
        ref_stack = torch.stack(ref_list)

    jepa, jepa_meta = train_ref_tx_jepa(ref_stack, ref_spec, device, args.jepa_steps)
    diff, diff_meta = train_ref_tx_diffusion(ref_stack, ref_spec, device, args.diff_steps)
    jepa_dir = Path("checkpoints/jepa"); jepa_dir.mkdir(parents=True, exist_ok=True)
    diff_dir = Path("checkpoints/diffusion"); diff_dir.mkdir(parents=True, exist_ok=True)
    jepa.save(str(jepa_dir / f"weight_jepa_{args.tag}.pt"))
    diff.save(str(diff_dir / f"weight_denoiser_{args.tag}.pt"))
    hybrid = JEPADiffusionHybrid(jepa, diff, refine_t=50)

    rnd_flats, jepa_flats, hyb_flats, cond_flats, task_flats = [], [], [], [], []
    ref_task = describe_tx_spec(ref_spec)
    tx_cond, _ = load_tx_cond_ckpt(str(ckpt_dir / "tx_cond_phase4b_v1.pt"), device) if (ckpt_dir / "tx_cond_phase4b_v1.pt").exists() else (None, None)
    for i in range(args.ref_samples):
        rnd_flats.append(flatten_state_dict(build_from_tx_spec(ref_spec)))
        jepa_flats.append(denormalize_weights(jepa.sample(1, device=device)[0].cpu(), jepa_meta))
        hyb_flats.append(denormalize_weights(hybrid.sample(1, device=device)[0].cpu(), jepa_meta))
        if tx_cond:
            from src.arch_gen.tx_conditioning import sample_denorm_tx
            cond_flats.append(sample_denorm_tx(tx_cond, ref_spec, norm_meta, device))
        m = build_from_tx_spec(ref_spec).to(device)
        init_task_tx_weights(m, ref_spec, w_gen, ref_task, norm_meta, device)
        task_flats.append(flatten_state_dict(m))

    methods = {"random": rnd_flats, "jepa": jepa_flats, "hybrid": hyb_flats, "task_tx": task_flats}
    if cond_flats: methods["tx_cond"] = cond_flats
    ref_eval = eval_ref_tx_inits(ref_spec, methods, device, ft_steps, args.seed + 8000)
    ref_metrics = {}
    rnd_rows = ref_eval["random"]
    for name, rows in ref_eval.items():
        m = metrics_from_rows(rows, rnd_rows, ft_steps, name)
        ref_metrics.update(m)

    mlp_jepa = json.loads((Path("checkpoints/jepa/metrics_phase2_global.json")).read_text()) if Path("checkpoints/jepa/metrics_phase2_global.json").exists() else {}
    mlp_hyb = json.loads((Path("checkpoints/hybrid/metrics_hybrid_t50.json")).read_text()) if Path("checkpoints/hybrid/metrics_hybrid_t50.json").exists() else {}

    metrics = {"tag": args.tag, "avg_task_match": avg_match, "prev_v1_match": v1.get("avg_task_match"),
               "match_gain_vs_v1": avg_match - (v1.get("avg_task_match") or 0), "train_pairs": len(train_texts),
               "examples": examples, "ref_spec": [ref_spec.d_model, ref_spec.n_layer, ref_spec.n_head],
               "ref_task": ref_task, "mlp_jepa_zero": mlp_jepa.get("jepa_zero_loss"), "mlp_hybrid_ft100": mlp_hyb.get("hybrid_ft100_loss"),
               **task_m, **ref_metrics}
    metrics["prev_v1_zero"] = v1.get("task_tx_zero_loss")
    metrics["zero_gain_vs_v1"] = (v1.get("task_tx_zero_loss") or 0) - task_m["steer_zero_loss"]

    plot_loss_curve(arch_losses, plot_dir / f"{args.tag}_arch.png", title="Task tx arch (refined)")
    plot_bar_comparison(["v2_match", "v1_match"], [avg_match, v1.get("avg_task_match", 0)], plot_dir / f"{args.tag}_match.png", title="Steering v2 vs v1", ylabel="Match")
    plot_scatter(match_scores, [r["zero_loss"] for r in task_rows], plot_dir / f"{args.tag}_match_loss.png", title="Match vs zero loss", xlabel="Match", ylabel="Loss")

    ref_labels = ["tx_jepa", "tx_hybrid", "tx_task", "tx_rnd", "mlp_jepa", "mlp_hyb"]
    ref_vals = [ref_metrics.get("jepa_zero_loss", 0), ref_metrics.get("hybrid_zero_loss", 0), ref_metrics.get("task_tx_zero_loss", 0),
                ref_metrics.get("random_zero_loss", 0), mlp_jepa.get("jepa_zero_loss", 0), mlp_hyb.get("hybrid_zero_loss", 0)]
    plot_bar_comparison(ref_labels, ref_vals, plot_dir / f"{args.tag}_ref_zero.png", title="Fixed ref zero-shot", ylabel="Loss")

    out = ckpt_dir / f"metrics_{args.tag}.json"
    out.write_text(json.dumps(metrics, indent=2))
    log_run("checkpoints/logs", f"phase4b_{args.tag}", metrics, vars(args))

    print(f"\n=== Phase 4b Refine [{args.tag}] ===")
    print(f"  match: v2={avg_match:.3f} v1={v1.get('avg_task_match')} gain={metrics['match_gain_vs_v1']:.3f}")
    print(f"  task_tx zero={task_m['steer_zero_loss']:.4f} (v1={v1.get('task_tx_zero_loss')})")
    print(f"  ref tx | jepa={ref_metrics.get('jepa_zero_loss'):.4f} hybrid={ref_metrics.get('hybrid_zero_loss'):.4f} task={ref_metrics.get('task_tx_zero_loss'):.4f} rnd={ref_metrics.get('random_zero_loss'):.4f}")
    print(f"  ref ft100 | jepa={ref_metrics.get('jepa_ft100_loss'):.4f} hybrid={ref_metrics.get('hybrid_ft100_loss'):.4f} task={ref_metrics.get('task_tx_ft100_loss'):.4f}")
    print(f"  mlp ref | jepa={mlp_jepa.get('jepa_zero_loss')} hybrid_ft100={mlp_hyb.get('hybrid_ft100_loss')}")


if __name__ == "__main__":
    main()
