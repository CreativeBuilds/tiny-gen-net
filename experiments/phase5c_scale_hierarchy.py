#!/usr/bin/env python3
"""Phase 5c: scale-consistency + block planning + anchor warm-start for extrapolation."""

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
from src.arch_gen.nano_eval import eval_nano_row, finetune_nano
from src.arch_gen.nano_spec import NanoSpec, anchor_presets, estimate_nano_params, extrapolation_holdout_presets, training_presets
from src.arch_gen.nano_task_cond import DEFAULT_NANO_EVAL_TASKS, describe_nano_spec, normalize_nano_task
from src.arch_gen.sentence_embedder import SentenceTaskEmbedder
from src.models.nano_transformer import build_from_nano_spec
from src.nano_warmstart import pick_donor_spec, warm_start_flat
from src.progressive_generator import ProgressiveWeightGenerator, ScaleAwareProgressiveTrainer, init_progressive_weights, split_layers
from src.scale_embedding import scale_features
from src.utils.device import device_str
from src.utils.logging import log_run
from src.utils.train_loop import set_seed
from src.utils.viz import plot_bar_comparison
from src.utils.weights import flatten_state_dict


ABLATION_PROFILES = {
    "warmstart-only": {"jepa_w": 0.15, "cons_w": 0.35, "plan_scale": 0.0, "cons_strong": False, "strong_plan": False},
}
PROFILE_CONFIGS = {
    "light": {"jepa_w": 0.1, "cons_w": 0.15, "plan_scale": 0.1, "cons_strong": False, "strong_plan": True},
    "light-noplan": {"jepa_w": 0.1, "cons_w": 0.15, "plan_scale": 0.1, "cons_strong": False, "strong_plan": False},
    "hybrid": {"jepa_w": 0.15, "cons_w": 0.35, "plan_scale": 0.1, "cons_strong": False, "strong_plan": True},
    "mid": {"jepa_w": 0.12, "cons_w": 0.25, "plan_scale": 0.1, "cons_strong": False, "strong_plan": True},
    "fine": {"jepa_w": 0.11, "cons_w": 0.20, "plan_scale": 0.1, "cons_strong": False, "strong_plan": True},
}
PROFILE_TAGS = {"light": "phase5c_light", "light-noplan": "phase5c_light_noplan", "hybrid": "phase5c_hybrid", "mid": "phase5c_mid", "fine": "phase5c_fine"}
DEFAULT_TAGS = ("phase5c_v1", "phase5c_v2", "phase5c_ws_ablation", "phase5c_light", "phase5c_light_noplan", "phase5c_hybrid", "phase5c_mid", "phase5c_fine", "")
DEFAULT_RUN = {"jepa_w": 0.2, "cons_w": 0.5, "plan_scale": 0.35, "cons_strong": True, "strong_plan": False}


def resolve_run_config(args):
    if args.ablation and args.profile: raise ValueError("Use only one of --ablation or --profile")
    cfg = dict(DEFAULT_RUN)
    if args.ablation:
        args.retrain_anchors, args.retrain_prog = True, True
        if args.tag in ("phase5c_v1", "phase5c_v2", ""): args.tag = "phase5c_ws_ablation"
        cfg.update(ABLATION_PROFILES[args.ablation])
    if args.profile:
        args.retrain_anchors, args.retrain_prog = True, True
        if args.tag in DEFAULT_TAGS: args.tag = PROFILE_TAGS.get(args.profile, f"phase5c_{args.profile.replace('-', '_')}")
        cfg.update(PROFILE_CONFIGS[args.profile])
    return cfg


def apply_profile(args):
    if args.smoke:
        args.num_collect, args.train_steps, args.prog_steps = 2, 40, 60
        args.num_tasks, args.ft_steps, args.anchor_steps = 2, "0", 30
        args.presets, args.collect_anchors = "smoke,nano_1m", False
        args.skip_heavy_eval, args.retrain_prog = True, True
        return
    if getattr(args, "cloud_resume", False) or os.environ.get("TINY_GEN_CLOUD_RESUME"):
        args.cloud = True
        args.skip_collect = True
        args.num_collect, args.train_steps, args.prog_steps = 15, 1200, 1000
        args.num_tasks, args.ft_steps, args.anchor_steps = 6, "100", 800
        args.presets = "nano_1m,nano_3m,nano_8m"
        args.collect_anchors = True
        return
    if args.cloud:
        args.num_collect, args.train_steps, args.prog_steps = 12, 1200, 1000
        args.num_tasks, args.ft_steps, args.anchor_steps = 8, "100,500", 600
        args.presets = "nano_1m,nano_3m,nano_8m"
        args.collect_anchors = True


def collect_weights(specs: list[NanoSpec], vocab: int, steps: int, seed: int, device: str, init_flat=None) -> list[torch.Tensor]:
    out = []
    for i, spec in enumerate(tqdm(specs, desc="collect")):
        m = build_from_nano_spec(spec, vocab).to(device)
        if init_flat is not None and init_flat.numel() == sum(p.numel() for p in m.parameters()):
            from src.utils.weights import load_flat_into_model
            load_flat_into_model(init_flat, m)
        finetune_nano(m, steps, seed=seed + i * 17, device=device)
        out.append(flatten_state_dict(m))
    return out


def collect_anchor_weights(names: list[str], base_specs: list[NanoSpec], base_weights: list[torch.Tensor], vocab: int, steps: int, seed: int, device: str) -> tuple[list[NanoSpec], list[torch.Tensor]]:
    specs, weights = [], []
    for i, name in enumerate(names):
        target = NanoSpec.preset(name)
        donor_spec = pick_donor_spec(target, base_specs, vocab)
        if donor_spec is None:
            print(f"anchor {name}: no donor, cold start"); flat = collect_weights([target], vocab, steps, seed + 900 + i, device)[0]
        else:
            di = base_specs.index(donor_spec)
            flat, _, matched, transferred = warm_start_flat(target, donor_spec, base_weights[di], vocab)
            print(f"anchor {name}: warm-start from [{donor_spec.n_embd},{donor_spec.n_layer},{donor_spec.n_head}] ({matched} tensors, {transferred:,} params)")
            m = build_from_nano_spec(target, vocab).to(device)
            from src.utils.weights import load_flat_into_model
            load_flat_into_model(flat, m)
            finetune_nano(m, steps, seed=seed + 800 + i, device=device)
            flat = flatten_state_dict(m)
        specs.append(target); weights.append(flat)
    return specs, weights


def main():
    p = argparse.ArgumentParser(description="Phase 5c scale hierarchy + anchors")
    p.add_argument("--num_collect", type=int, default=9)
    p.add_argument("--train_steps", type=int, default=800)
    p.add_argument("--prog_steps", type=int, default=800)
    p.add_argument("--anchor_steps", type=int, default=400)
    p.add_argument("--num_tasks", type=int, default=6)
    p.add_argument("--ft_steps", default="100")
    p.add_argument("--presets", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="phase5c_v1")
    p.add_argument("--plot_dir", default="checkpoints/plots/phase5c_scale")
    p.add_argument("--skip_collect", action="store_true")
    p.add_argument("--collect-anchors", action="store_true")
    p.add_argument("--retrain-anchors", action="store_true")
    p.add_argument("--skip_heavy_eval", action="store_true")
    p.add_argument("--retrain-prog", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--cloud-resume", action="store_true")
    p.add_argument("--ablation", default="", choices=["", "warmstart-only"])
    p.add_argument("--profile", default="", choices=["", "light", "light-noplan", "hybrid", "mid", "fine"])
    p.add_argument("--cloud", action="store_true")
    args = p.parse_args()
    apply_profile(args)
    run_cfg = resolve_run_config(args)
    ft_steps = [int(x) for x in args.ft_steps.split(",") if x.strip()]
    eval_tasks = DEFAULT_NANO_EVAL_TASKS[: args.num_tasks]

    set_seed(args.seed)
    device = device_str()
    if args.cloud and device == "cpu": raise RuntimeError("Cloud run requires CUDA")
    _, _, tok = load_shakespeare(seed=args.seed)
    vocab = tok.vocab_size
    plot_dir = Path(args.plot_dir); plot_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("checkpoints/arch_gen"); ckpt_dir.mkdir(parents=True, exist_ok=True)
    wdir = Path("checkpoints/weights/nano_multi_v1"); wdir.mkdir(parents=True, exist_ok=True)
    raw_path = wdir / "weights_raw.pt"
    anchor_path = wdir / "weights_with_anchors.pt"

    train_names = [n.strip() for n in (args.presets or ",".join(training_presets())).split(",") if n.strip()]
    collect_specs = [NanoSpec.preset(n) for n in train_names]
    while len(collect_specs) < args.num_collect: collect_specs.extend(collect_specs)
    collect_specs = collect_specs[: args.num_collect]

    if raw_path.exists() and args.skip_collect:
        ckpt = torch.load(raw_path, map_location="cpu", weights_only=True)
        collect_specs = [NanoSpec.from_label(lbl) for lbl in ckpt["meta"]["specs"]]
        raw_w = ckpt["weights"]
        if isinstance(raw_w, torch.Tensor): raw_w = [raw_w[i] for i in range(raw_w.shape[0])]
        collect_specs, raw_w = collect_specs[: args.num_collect], raw_w[: args.num_collect]
        print(f"Loaded {len(collect_specs)} weights from {raw_path}")
    else:
        raw_w = collect_weights(collect_specs, vocab, args.train_steps, args.seed, device)
        torch.save({"weights": raw_w, "meta": {"specs": [[s.n_embd, s.n_layer, s.n_head, s.block_size] for s in collect_specs]}}, raw_path)

    if args.collect_anchors:
        if anchor_path.exists() and not args.retrain_anchors and args.skip_collect:
            ckpt = torch.load(anchor_path, map_location="cpu", weights_only=True)
            collect_specs = [NanoSpec.from_label(lbl) for lbl in ckpt["meta"]["specs"]]
            raw_w = ckpt["weights"]
            if isinstance(raw_w, torch.Tensor): raw_w = [raw_w[i] for i in range(raw_w.shape[0])]
            print(f"Loaded {len(collect_specs)} weights (with anchors) from {anchor_path}")
        else:
            a_specs, a_w = collect_anchor_weights(anchor_presets(), collect_specs, raw_w, vocab, args.anchor_steps, args.seed, device)
            collect_specs.extend(a_specs); raw_w.extend(a_w)
            torch.save({"weights": raw_w, "meta": {"specs": [[s.n_embd, s.n_layer, s.n_head, s.block_size] for s in collect_specs], "anchors": anchor_presets()}}, anchor_path)

    enc = SentenceTaskEmbedder()
    texts = [describe_nano_spec(s) for s in collect_specs]
    tasks_t = [enc.encode_text(normalize_nano_task(t), device) for t in texts]
    cons_tasks = [enc.encode_text(normalize_nano_task(t), device) for t in eval_tasks]
    scales_t = [scale_features(s, vocab).to(device) for s in collect_specs]
    layer_batches = [split_layers(w, s, vocab) for w, s in zip(raw_w, collect_specs)]

    prog = ProgressiveWeightGenerator(vocab=vocab, plan_scale=run_cfg["plan_scale"], strong_plan=run_cfg["strong_plan"])
    prog.high.cons_strong = run_cfg["cons_strong"]
    prog_path = ckpt_dir / f"prog_{args.tag}.pt"
    if prog_path.exists() and not args.retrain_prog:
        prog.load_state_dict(torch.load(prog_path, map_location=device, weights_only=True))
        prog.to(device); print(f"Resume: loaded {prog_path}")
    else:
        hist = ScaleAwareProgressiveTrainer(prog, device=device, jepa_w=run_cfg["jepa_w"], cons_w=run_cfg["cons_w"]).fit(tasks_t, scales_t, layer_batches, cons_tasks, steps=args.prog_steps)
        torch.save(prog.state_dict(), prog_path)
        if hist:
            plot_bar_comparison(["start", "end"], [hist[0]["total"], hist[-1]["total"]], plot_dir / f"{args.tag}_train_loss.png", title="5c train loss", ylabel="loss")
            plot_bar_comparison(["prog", "cons"], [hist[-1]["prog"], hist[-1]["cons"]], plot_dir / f"{args.tag}_components.png", title="5c final components", ylabel="loss")

    if args.skip_heavy_eval: print("Smoke/heavy-skip OK"); return

    holdout = [NanoSpec.preset(n) for n in extrapolation_holdout_presets()]
    eval_specs = collect_specs[: min(3, len(collect_specs))] + holdout
    prog_rows, rnd_rows = [], []
    for i, spec in enumerate(tqdm(eval_specs, desc="eval")):
        task = eval_tasks[i % len(eval_tasks)]
        mp = build_from_nano_spec(spec, vocab).to(device)
        init_progressive_weights(mp, spec, prog, task, enc, device)
        prog_rows.append(eval_nano_row(mp, device, ft_steps, args.seed + i))
        mr = build_from_nano_spec(spec, vocab).to(device)
        finetune_nano(mr, 0, seed=9000 + i, device=device)
        rnd_rows.append(eval_nano_row(mr, device, ft_steps, args.seed + 9000 + i))

    in_rows, in_rnd = prog_rows[: min(3, len(prog_rows))], rnd_rows[: min(3, len(rnd_rows))]
    out_rows, out_rnd = prog_rows[min(3, len(prog_rows)):], rnd_rows[min(3, len(rnd_rows)):]
    in_m = metrics_from_rows(in_rows, in_rnd, ft_steps, "prog_in")
    out_m = metrics_from_rows(out_rows, out_rnd, ft_steps, "prog_extrap") if out_rows else {}
    metrics = {**in_m, **out_m, "random_zero": rnd_rows[0]["zero_loss"] if rnd_rows else 0,
               "collect_n": len(collect_specs), "anchors": args.collect_anchors, "retrain_anchors": args.retrain_anchors,
               "ablation": args.ablation or None, "profile": args.profile or None, "run_cfg": run_cfg, "holdout": extrapolation_holdout_presets(),
               "cloud": args.cloud, "smoke": args.smoke}
    metrics_path = ckpt_dir / f"metrics_{args.tag}.json"
    examples = [{"spec": [s.n_embd, s.n_layer, s.n_head, s.block_size], "preset_params": estimate_nano_params(s.n_embd, s.n_layer, s.n_head, vocab, s.block_size),
                 "zero_loss": prog_rows[i]["zero_loss"], "random_zero": rnd_rows[i]["zero_loss"],
                 "extrapolation": i >= min(3, len(collect_specs))} for i, s in enumerate(eval_specs)]
    metrics_path.write_text(json.dumps({"metrics": metrics, "examples": examples}, indent=2))
    plot_bar_comparison(["in_grid", "extrap"],
        [in_m.get("prog_in_zero_delta_vs_random", 0), out_m.get("prog_extrap_zero_delta_vs_random", 0)],
        plot_dir / f"{args.tag}_extrap.png", title="5c zero-shot Δ vs random", ylabel="Δ loss")
    log_run("checkpoints/logs", f"phase5c_{args.tag}", metrics, vars(args))
    print(json.dumps(metrics, indent=2))
    print(f"Artifacts: {metrics_path}, {plot_dir}")


if __name__ == "__main__":
    main()
