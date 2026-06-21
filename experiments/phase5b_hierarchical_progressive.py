#!/usr/bin/env python3
"""Phase 5b: scale-aware hierarchical progressive weight generation + extrapolation eval."""

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
from src.arch_gen.nano_spec import NanoSpec, extrapolation_holdout_presets, training_presets
from src.arch_gen.nano_task_cond import DEFAULT_NANO_EVAL_TASKS, describe_nano_spec, normalize_nano_task
from src.arch_gen.sentence_embedder import SentenceTaskEmbedder
from src.models.nano_transformer import build_from_nano_spec
from src.progressive_generator import ProgressiveWeightGenerator, ProgressiveWeightTrainer, init_progressive_weights, split_layers
from src.scale_embedding import scale_features
from src.utils.device import device_str
from src.utils.logging import log_run
from src.utils.train_loop import set_seed
from src.utils.viz import plot_bar_comparison
from src.utils.weights import flatten_state_dict


def apply_profile(args):
    if args.smoke:
        args.num_collect, args.train_steps, args.prog_steps = 2, 40, 40
        args.num_tasks, args.ft_steps = 2, "0"
        args.presets = "smoke,nano_1m"
        args.skip_heavy_eval = True
        return
    if getattr(args, "cloud_resume", False) or os.environ.get("TINY_GEN_CLOUD_RESUME"):
        args.cloud = True
        args.skip_collect = True
        args.num_collect, args.train_steps, args.prog_steps = 15, 1200, 800
        args.num_tasks, args.ft_steps = 6, "100"
        args.presets = "nano_1m,nano_3m,nano_8m"
        return
    if args.cloud:
        args.num_collect, args.train_steps, args.prog_steps = 12, 1200, 1000
        args.num_tasks, args.ft_steps = 8, "100,500"
        args.presets = "nano_1m,nano_3m,nano_8m"


def collect_weights(specs: list[NanoSpec], vocab: int, steps: int, seed: int, device: str, init_flat: torch.Tensor | None = None) -> list[torch.Tensor]:
    out = []
    for i, spec in enumerate(tqdm(specs, desc="collect")):
        m = build_from_nano_spec(spec, vocab).to(device)
        if init_flat is not None and init_flat.numel() == sum(p.numel() for p in m.parameters()):
            from src.utils.weights import load_flat_into_model
            load_flat_into_model(init_flat, m)
        finetune_nano(m, steps, seed=seed + i * 17, device=device)
        out.append(flatten_state_dict(m))
    return out


def main():
    p = argparse.ArgumentParser(description="Phase 5b hierarchical progressive extrapolation")
    p.add_argument("--num_collect", type=int, default=9)
    p.add_argument("--train_steps", type=int, default=800)
    p.add_argument("--prog_steps", type=int, default=600)
    p.add_argument("--num_tasks", type=int, default=6)
    p.add_argument("--ft_steps", default="100")
    p.add_argument("--presets", default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", default="phase5b_v1")
    p.add_argument("--plot_dir", default="checkpoints/plots/phase5b_prog")
    p.add_argument("--skip_collect", action="store_true")
    p.add_argument("--skip_heavy_eval", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--cloud-resume", action="store_true")
    p.add_argument("--retrain-prog", action="store_true", help="Ignore saved prog checkpoint and retrain")
    p.add_argument("--cloud", action="store_true")
    args = p.parse_args()
    apply_profile(args)
    ft_steps = [int(x) for x in args.ft_steps.split(",") if x.strip()]
    tasks = DEFAULT_NANO_EVAL_TASKS[: args.num_tasks]

    set_seed(args.seed)
    device = device_str()
    if args.cloud and device == "cpu":
        raise RuntimeError("Cloud run requires CUDA")
    _, _, tok = load_shakespeare(seed=args.seed)
    vocab = tok.vocab_size
    plot_dir = Path(args.plot_dir); plot_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("checkpoints/arch_gen"); ckpt_dir.mkdir(parents=True, exist_ok=True)
    wdir = Path("checkpoints/weights/nano_multi_v1"); wdir.mkdir(parents=True, exist_ok=True)
    raw_path = wdir / "weights_raw.pt"

    train_names = [n.strip() for n in (args.presets or ",".join(training_presets())).split(",") if n.strip()]
    collect_specs = [NanoSpec.preset(n) for n in train_names]
    while len(collect_specs) < args.num_collect:
        collect_specs.extend(collect_specs)
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

    enc = SentenceTaskEmbedder()
    texts = [describe_nano_spec(s) for s in collect_specs]
    tasks_t = [enc.encode_text(normalize_nano_task(t), device) for t in texts]
    scales_t = [scale_features(s, vocab).to(device) for s in collect_specs]
    layer_batches = [split_layers(w, s, vocab) for w, s in zip(raw_w, collect_specs)]

    prog = ProgressiveWeightGenerator(vocab=vocab)
    prog_path = ckpt_dir / f"prog_{args.tag}.pt"
    if prog_path.exists() and not args.retrain_prog:
        prog.load_state_dict(torch.load(prog_path, map_location=device, weights_only=True))
        prog.to(device)
        print(f"Resume: loaded {prog_path}")
    else:
        losses = ProgressiveWeightTrainer(prog, device=device).fit(tasks_t, scales_t, layer_batches, steps=args.prog_steps)
        torch.save(prog.state_dict(), prog_path)
        if losses: plot_bar_comparison(["start", "end"], [losses[0], losses[-1]], plot_dir / f"{args.tag}_prog_loss.png", title="Progressive train loss", ylabel="MSE")

    if args.skip_heavy_eval:
        print("Smoke/heavy-skip OK"); return

    holdout = [NanoSpec.preset(n) for n in extrapolation_holdout_presets()]
    eval_specs = collect_specs[: min(3, len(collect_specs))] + holdout
    prog_rows, rnd_rows = [], []
    for i, spec in enumerate(tqdm(eval_specs, desc="eval")):
        task = tasks[i % len(tasks)]
        mp = build_from_nano_spec(spec, vocab).to(device)
        init_progressive_weights(mp, spec, prog, task, enc, device)
        prog_rows.append(eval_nano_row(mp, device, ft_steps, args.seed + i))
        mr = build_from_nano_spec(spec, vocab).to(device)
        finetune_nano(mr, 0, seed=9000 + i, device=device)
        rnd_rows.append(eval_nano_row(mr, device, ft_steps, args.seed + 9000 + i))

    in_rows = prog_rows[: min(3, len(prog_rows))]
    in_rnd = rnd_rows[: min(3, len(rnd_rows))]
    out_rows = prog_rows[min(3, len(prog_rows)):]
    out_rnd = rnd_rows[min(3, len(rnd_rows)):]
    in_m = metrics_from_rows(in_rows, in_rnd, ft_steps, "prog_in")
    out_m = metrics_from_rows(out_rows, out_rnd, ft_steps, "prog_extrap") if out_rows else {}
    metrics = {**in_m, **out_m, "random_zero": rnd_rows[0]["zero_loss"] if rnd_rows else 0,
               "collect_n": len(collect_specs), "train_presets": train_names, "holdout": extrapolation_holdout_presets(),
               "cloud": args.cloud, "smoke": args.smoke}
    metrics_path = ckpt_dir / f"metrics_{args.tag}.json"
    examples = [{"spec": [s.n_embd, s.n_layer, s.n_head, s.block_size], "preset_params": s.param_count(vocab),
                 "zero_loss": prog_rows[i]["zero_loss"], "random_zero": rnd_rows[i]["zero_loss"],
                 "extrapolation": i >= min(3, len(collect_specs))} for i, s in enumerate(eval_specs)]
    metrics_path.write_text(json.dumps({"metrics": metrics, "examples": examples}, indent=2))
    plot_bar_comparison(["in_grid", "extrap"],
        [in_m.get("prog_in_zero_delta_vs_random", 0), out_m.get("prog_extrap_zero_delta_vs_random", 0)],
        plot_dir / f"{args.tag}_extrap.png", title="Progressive zero-shot Δ vs random", ylabel="Δ loss")
    log_run("checkpoints/logs", f"phase5b_{args.tag}", metrics, vars(args))
    print(json.dumps(metrics, indent=2))
    print(f"Artifacts: {metrics_path}, {plot_dir}")


if __name__ == "__main__":
    main()
