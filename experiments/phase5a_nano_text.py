#!/usr/bin/env python3
"""Phase 5a: real Shakespeare text + sentence embeddings + nanoGPT-scale transformer."""

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from tqdm import tqdm

from data.shakespeare import load_shakespeare
from experiments.phase2_jepa_vs_diffusion import metrics_from_rows
from src.arch_gen.nano_conditioning import NanoCondWeightGenerator, NanoCondWeightTrainer, init_nano_weights, load_nano_cond_ckpt, normalize_nano_weights, save_nano_cond_ckpt
from src.arch_gen.nano_eval import eval_nano_row, finetune_nano
from src.arch_gen.nano_jepa import TaskCondWeightJEPA, TaskCondWeightJEPATrainer
from src.arch_gen.nano_spec import REF_N_EMBD, REF_N_HEAD, REF_N_LAYER, REF_BLOCK, NanoSpec, all_valid_nano_specs, pick_diverse_nano_specs
from src.arch_gen.nano_task_cond import DEFAULT_NANO_EVAL_TASKS, augment_nano_training_pairs, describe_nano_spec, nano_spec_matches_task, normalize_nano_task
from src.arch_gen.nano_task_generator import TaskNanoArchGenerator, TaskNanoArchTrainer
from src.arch_gen.nano_task_weights import TaskNanoWeightGenerator, TaskNanoWeightTrainer, init_task_nano_weights, load_task_nano_ckpt, save_task_nano_ckpt
from src.arch_gen.sentence_embedder import SentenceTaskEmbedder
from src.models.nano_transformer import build_from_nano_spec
from src.utils.device import device_str
from src.utils.logging import log_run
from src.utils.train_loop import set_seed
from src.utils.viz import plot_bar_comparison, plot_loss_curve, plot_scatter
from src.utils.weights import flatten_state_dict, load_flat_into_model, param_slices


def apply_profile(args):
    if args.smoke:
        args.num_collect, args.train_steps = 1, 40
        args.cond_steps, args.arch_steps, args.weight_steps, args.jepa_steps = 0, 30, 0, 0
        args.num_tasks, args.ft_steps = 1, "0"
        args.presets = "smoke"
        args.skip_cond = args.skip_weight = args.skip_jepa = args.skip_heavy_eval = True
        return
    if getattr(args, "cloud_smoke", False) or os.environ.get("TINY_GEN_CLOUD_SMOKE"):
        args.cloud = True
        args.num_collect, args.train_steps = 6, 800
        args.cond_steps, args.arch_steps, args.weight_steps, args.jepa_steps = 400, 300, 500, 300
        args.num_tasks, args.ft_steps = 6, "100"
        args.presets = "smoke,nano_1m"
        return
    if getattr(args, "cloud_first", False) or os.environ.get("TINY_GEN_CLOUD_FIRST"):
        args.cloud = True
        args.num_collect, args.train_steps = 9, 1200
        args.cond_steps, args.arch_steps, args.weight_steps, args.jepa_steps = 1000, 600, 800, 500
        args.num_tasks, args.ft_steps = 8, "100,500"
        args.presets = "nano_1m,nano_3m"
        return
    if getattr(args, "cloud_resume", False) or os.environ.get("TINY_GEN_CLOUD_RESUME"):
        args.cloud = True
        args.skip_collect = True
        args.num_collect, args.train_steps = 9, 1200
        args.cond_steps, args.arch_steps, args.weight_steps, args.jepa_steps = 1000, 600, 800, 500
        args.num_tasks, args.ft_steps = 8, "100"
        return
    if args.cloud or os.environ.get("TINY_GEN_CLOUD"):
        args.cloud = True
        args.num_collect = max(args.num_collect, 15)
        args.train_steps = max(args.train_steps, 2500)
        args.cond_steps = max(args.cond_steps, 3000)
        args.arch_steps = max(args.arch_steps, 1200)
        args.weight_steps = max(args.weight_steps, 2000)
        args.jepa_steps = max(args.jepa_steps, 1500)
        if not args.presets: args.presets = "nano_1m,nano_3m,nano_8m"


def collect_nano_weights(specs: list[NanoSpec], vocab: int, steps: int, seed: int, device: str) -> list[torch.Tensor]:
    weights = []
    for i, spec in enumerate(tqdm(specs, desc="collect")):
        m = build_from_nano_spec(spec, vocab).to(device)
        finetune_nano(m, steps, seed=seed + i * 17, device=device)
        weights.append(flatten_state_dict(m))
    return weights


def main():
    p = argparse.ArgumentParser(description="Phase 5a nano text scale-up")
    p.add_argument("--num_collect", type=int, default=12)
    p.add_argument("--train_steps", type=int, default=800)
    p.add_argument("--cond_steps", type=int, default=1200)
    p.add_argument("--arch_steps", type=int, default=600)
    p.add_argument("--weight_steps", type=int, default=1000)
    p.add_argument("--jepa_steps", type=int, default=800)
    p.add_argument("--match_weight", type=float, default=0.5)
    p.add_argument("--num_tasks", type=int, default=8)
    p.add_argument("--ft_steps", default="100,500")
    p.add_argument("--presets", default="", help="comma presets or empty=diverse grid")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot_dir", default="checkpoints/plots/phase5a_nano")
    p.add_argument("--tag", default="phase5a_v1")
    p.add_argument("--skip_collect", action="store_true")
    p.add_argument("--skip_cond", action="store_true")
    p.add_argument("--skip_weight", action="store_true")
    p.add_argument("--skip_jepa", action="store_true")
    p.add_argument("--skip_heavy_eval", action="store_true", help="skip slow hypernet init eval")
    p.add_argument("--smoke", action="store_true", help="~10s local wiring check (no weight-gen training)")
    p.add_argument("--cloud-smoke", action="store_true", help="medium cloud run (~30 min on RTX 4090)")
    p.add_argument("--cloud-first", action="store_true", help="first cloud run (~1-2 hr H100): 9 collect, nano_1m+3m")
    p.add_argument("--cloud-resume", action="store_true", help="reuse weights_raw.pt + stage ckpts; cloud-first hypernet steps, ft100 eval")
    p.add_argument("--cloud", action="store_true", help="full run on RunPod GPU (H100 recommended)")
    args = p.parse_args()
    apply_profile(args)
    if not args.cloud and not args.smoke and not os.environ.get("TINY_GEN_CLOUD"):
        print("NOTE: full Phase 5a is slow on Mac (~30+ min). Use --cloud on RunPod or --smoke locally.")
    ft_steps = [int(x) for x in args.ft_steps.split(",") if x.strip()]
    tasks = DEFAULT_NANO_EVAL_TASKS[: args.num_tasks]

    set_seed(args.seed)
    device = device_str()
    if (args.cloud or os.environ.get("TINY_GEN_CLOUD")) and device == "cpu":
        raise RuntimeError(f"Cloud run requires CUDA GPU; got device={device}. Check RunPod torch/CUDA setup.")
    _, _, tok = load_shakespeare(seed=args.seed)
    vocab = tok.vocab_size
    plot_dir = Path(args.plot_dir); plot_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("checkpoints/arch_gen"); ckpt_dir.mkdir(parents=True, exist_ok=True)
    wdir = Path("checkpoints/weights/nano_multi_v1"); wdir.mkdir(parents=True, exist_ok=True)
    raw_path = wdir / "weights_raw.pt"

    if args.presets:
        collect_specs = [NanoSpec.preset(n.strip()) for n in args.presets.split(",") if n.strip()]
        while len(collect_specs) < args.num_collect:
            collect_specs.extend(collect_specs)
        collect_specs = collect_specs[: args.num_collect]
    else:
        collect_specs = pick_diverse_nano_specs(args.num_collect, random.Random(args.seed))

    if raw_path.exists() and args.skip_collect:
        ckpt = torch.load(raw_path, map_location="cpu", weights_only=True)
        collect_specs = [NanoSpec.from_label(lbl) for lbl in ckpt["meta"]["specs"]]
        raw_w = ckpt["weights"]
        if isinstance(raw_w, torch.Tensor): raw_w = [raw_w[i] for i in range(raw_w.shape[0])]
    else:
        raw_w = collect_nano_weights(collect_specs, vocab, args.train_steps, args.seed, device)
        torch.save({"weights": raw_w, "meta": {"specs": [[s.n_embd, s.n_layer, s.n_head, s.block_size] for s in collect_specs],
            "train_steps": args.train_steps, "vocab": vocab, "grid_size": len(all_valid_nano_specs())}}, raw_path)

    texts = [describe_nano_spec(s) for s in collect_specs]
    train_specs, train_texts = augment_nano_training_pairs(collect_specs, texts, random.Random(args.seed))
    norm_w, norm_meta = normalize_nano_weights(raw_w)
    norm_meta["vocab"] = vocab
    print(f"Phase 5a | collect={len(collect_specs)} grid={len(all_valid_nano_specs())} vocab={vocab} tasks={len(tasks)} device={device} cloud={args.cloud} smoke={args.smoke} resume={getattr(args, 'cloud_resume', False)}")

    cond_path = ckpt_dir / f"nano_cond_{args.tag}.pt"
    task_path = ckpt_dir / f"task_nano_{args.tag}.pt"
    jepa_path = Path("checkpoints/jepa") / f"weight_jepa_{args.tag}.pt"

    cond = NanoCondWeightGenerator(vocab=vocab)
    cond_losses: list[float] = []
    if cond_path.exists():
        cond, ckpt_meta = load_nano_cond_ckpt(str(cond_path), device)
        norm_meta.update({k: v for k, v in ckpt_meta.items() if k in ("mean", "std", "norm_mode")})
        print(f"Resume: loaded {cond_path}")
    elif not args.skip_cond and args.cond_steps > 0:
        cond_losses = NanoCondWeightTrainer(cond, device=device).fit(collect_specs, norm_w, steps=args.cond_steps, ckpt_path=str(cond_path), norm_meta=norm_meta)
        save_nano_cond_ckpt(str(cond_path), cond, norm_meta)

    arch_gen = TaskNanoArchGenerator()
    w_gen = TaskNanoWeightGenerator(vocab=vocab)
    arch_losses: list[float] = []
    if task_path.exists():
        arch_gen, w_gen, ckpt_meta = load_task_nano_ckpt(str(task_path), device)
        norm_meta.update({k: v for k, v in ckpt_meta.items() if k in ("mean", "std", "norm_mode")})
        print(f"Resume: loaded {task_path}")
    else:
        arch_losses = TaskNanoArchTrainer(arch_gen, device=device, match_weight=args.match_weight).fit_paired(train_specs, train_texts, steps=args.arch_steps) if args.arch_steps > 0 else []
        w_losses: list[float] = []
        if not args.skip_weight and args.weight_steps > 0:
            spec_key = lambda s: (s.n_embd, s.n_layer, s.n_head, s.block_size)
            spec_to_w = {spec_key(s): w for s, w in zip(collect_specs, norm_w)}
            paired_w = [spec_to_w[spec_key(s)] for s in train_specs]
            w_losses = TaskNanoWeightTrainer(w_gen, device=device, match_weight=0.3).fit(train_specs, train_texts, paired_w, steps=args.weight_steps, ckpt_path=str(task_path), arch_gen=arch_gen, norm_meta=norm_meta)
            save_task_nano_ckpt(str(task_path), arch_gen, w_gen, norm_meta)

    ref_spec = NanoSpec(REF_N_EMBD, REF_N_LAYER, REF_N_HEAD, REF_BLOCK).validate()
    jepa_losses: list[float] = []
    jepa, ref_jepa_meta = None, None
    if jepa_path.exists():
        ref_m = build_from_nano_spec(ref_spec, vocab).to(device)
        ref_bounds = [(s["start"], s["end"]) for s in param_slices(ref_m)]
        jepa = TaskCondWeightJEPA(ref_bounds)
        jepa.load(str(jepa_path), device)
        ref_jepa_meta = {"mean": norm_meta.get("mean"), "std": norm_meta.get("std")}
        print(f"Resume: loaded {jepa_path}")
    elif not args.skip_jepa and args.jepa_steps > 0:
        ref_m = build_from_nano_spec(ref_spec, vocab).to(device)
        ref_bounds = [(s["start"], s["end"]) for s in param_slices(ref_m)]
        jepa = TaskCondWeightJEPA(ref_bounds)
        spec_key = lambda s: (s.n_embd, s.n_layer, s.n_head, s.block_size)
        ref_raw = [raw_w[i] for i, s in enumerate(collect_specs) if spec_key(s) == spec_key(ref_spec)]
        if not ref_raw:
            for j in range(4):
                rm = build_from_nano_spec(ref_spec, vocab).to(device)
                finetune_nano(rm, max(args.train_steps // 2, 100), seed=args.seed + 777 + j, device=device)
                ref_raw.append(flatten_state_dict(rm))
        ref_norm, ref_jepa_meta = normalize_nano_weights(ref_raw)
        ref_texts = [describe_nano_spec(ref_spec) for _ in ref_raw]
        jepa_losses = TaskCondWeightJEPATrainer(jepa, device=device).fit(ref_norm, ref_texts, steps=args.jepa_steps)
        jepa_path.parent.mkdir(parents=True, exist_ok=True)
        jepa.save(str(jepa_path))

    if cond_losses: plot_loss_curve(cond_losses[:: max(1, len(cond_losses) // 200)], plot_dir / f"{args.tag}_cond.png", title="Nano cond weights")
    if arch_losses: plot_loss_curve(arch_losses, plot_dir / f"{args.tag}_arch.png", title="Task nano arch")
    if jepa_losses: plot_loss_curve(jepa_losses[:: max(1, len(jepa_losses) // 200)], plot_dir / f"{args.tag}_jepa.png", title="Task nano JEPA")

    if args.smoke:
        enc = SentenceTaskEmbedder()
        enc.encode_text("small shakespeare text transformer", device)
        spec = arch_gen.sample(1, tasks[0], device)[0]
        m = build_from_nano_spec(spec, vocab).to(device)
        finetune_nano(m, 20, seed=args.seed, device=device)
        row = eval_nano_row(m, device, [0], args.seed)
        metrics = {"smoke_wiring": True, "shakespeare_vocab": vocab, "sample_spec": [spec.n_embd, spec.n_layer, spec.n_head, spec.block_size],
                   "random_eval_loss": row["zero_loss"], "grid_size": len(all_valid_nano_specs())}
        metrics_path = ckpt_dir / f"metrics_{args.tag}.json"
        metrics_path.write_text(json.dumps({"metrics": metrics}, indent=2))
        print(json.dumps(metrics, indent=2))
        print(f"Smoke OK (~2 min). Full training: RUNPOD_SSH='...' ./scripts/runpod_train.sh")
        return

    if args.skip_heavy_eval:
        print("Skipped heavy eval (--skip_heavy_eval)"); return

    task_rows, cond_rows, rnd_rows, jepa_rows, match_scores, examples = [], [], [], [], [], []
    enc = SentenceTaskEmbedder()

    for i, task in enumerate(tqdm(tasks, desc="eval")):
        spec_t = arch_gen.sample(1, task, device)[0]
        mt = build_from_nano_spec(spec_t, vocab).to(device)
        init_task_nano_weights(mt, spec_t, w_gen, task, norm_meta, device)
        task_rows.append(eval_nano_row(mt, device, ft_steps, args.seed + i))
        ms = nano_spec_matches_task(spec_t, task)
        match_scores.append(ms)
        examples.append({"task": task, "spec": [spec_t.n_embd, spec_t.n_layer, spec_t.n_head, spec_t.block_size], "match": ms})

        spec_c = collect_specs[i % len(collect_specs)]
        mc = build_from_nano_spec(spec_c, vocab).to(device)
        init_nano_weights(mc, spec_c, cond, norm_meta, device)
        cond_rows.append(eval_nano_row(mc, device, ft_steps, args.seed + 5000 + i))

        mr = build_from_nano_spec(ref_spec, vocab).to(device)
        finetune_nano(mr, 0, seed=9000 + i, device=device)
        rnd_rows.append(eval_nano_row(mr, device, ft_steps, args.seed + 9000 + i))

        mj = build_from_nano_spec(ref_spec, vocab).to(device)
        if jepa is not None and ref_jepa_meta is not None:
            tv = enc.encode_text(normalize_nano_task(task), device)
            flat = jepa.sample_cond(tv, device=device) * ref_jepa_meta["std"] + ref_jepa_meta["mean"]
            load_flat_into_model(flat, mj)
            jepa_rows.append(eval_nano_row(mj, device, ft_steps, args.seed + 7000 + i))

    task_m = metrics_from_rows(task_rows, rnd_rows, ft_steps, "task_nano")
    cond_m = metrics_from_rows(cond_rows, rnd_rows, ft_steps, "nano_cond")
    jepa_m = metrics_from_rows(jepa_rows, rnd_rows, ft_steps, "nano_jepa")
    metrics = {**task_m, **cond_m, **jepa_m, "arch_match_mean": sum(match_scores) / len(match_scores),
               "random_zero": rnd_rows[0]["zero_loss"] if rnd_rows else 0, "grid_size": len(all_valid_nano_specs()),
               "vocab": vocab, "collect_n": len(collect_specs), "cloud": args.cloud, "smoke": args.smoke}
    metrics_path = ckpt_dir / f"metrics_{args.tag}.json"
    metrics_path.write_text(json.dumps({"metrics": metrics, "examples": examples}, indent=2))
    plot_bar_comparison(["task_nano", "nano_cond", "nano_jepa"],
        [task_m.get("task_nano_zero_delta_vs_random", 0), cond_m.get("nano_cond_zero_delta_vs_random", 0), jepa_m.get("nano_jepa_zero_delta_vs_random", 0)],
        plot_dir / f"{args.tag}_zero_delta.png", title="Zero-shot Δ vs random (Shakespeare)", ylabel="Δ loss")
    plot_scatter([r["zero_loss"] for r in task_rows], [r["zero_loss"] for r in rnd_rows], plot_dir / f"{args.tag}_scatter.png", xlabel="task-nano", ylabel="random")
    log_run("checkpoints/logs", f"phase5a_{args.tag}", metrics, vars(args))
    print(json.dumps(metrics, indent=2))
    print(f"Artifacts: {metrics_path}, {plot_dir}")


if __name__ == "__main__":
    main()
