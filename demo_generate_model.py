#!/usr/bin/env python3
"""Demo: generate tiny MLPs or transformers from task description + conditioned weights."""

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch

from src.arch_gen.conditioning import load_cond_checkpoint, ref_similarity
from src.arch_gen.eval import eval_arch_row
from src.arch_gen.generator import ArchGenerator
from src.arch_gen.spec import ArchSpec, REF_DEPTH, REF_HIDDEN, random_spec
from src.arch_gen.task_cond import DEMO_EXAMPLES, arch_summary, describe_spec, spec_matches_task
from src.arch_gen.task_weights import load_task_cond_ckpt
from src.arch_gen.tx_conditioning import load_tx_cond_ckpt, tx_ref_similarity
from src.arch_gen.tx_eval import eval_tx_row
from src.arch_gen.tx_spec import REF_D_MODEL, REF_N_HEAD, REF_N_LAYER, TxSpec, random_tx_spec
from src.arch_gen.tx_task_cond import TX_DEMO_EXAMPLES, describe_tx_spec, tx_arch_summary, tx_spec_matches_task
from src.arch_gen.tx_task_weights import init_task_tx_weights, load_task_tx_ckpt
from src.arch_gen.weight_init import init_model_weights, init_task_weights
from src.models.variable_mlp import build_from_spec
from src.models.variable_transformer import build_from_tx_spec
from src.utils.device import device_str
from src.utils.train_loop import set_seed

DEFAULT_TASK = "medium balanced char predictor"
DEFAULT_TASK_CKPT = "checkpoints/arch_gen/task_cond_phase4_v2.pt"
DEFAULT_TX_TASK_CKPT = "checkpoints/arch_gen/task_tx_phase4b_v2.pt"


def pick_mlp_specs(n, diversity, device, arch_ckpt, task_text, task_ckpt, seed):
    if task_ckpt.exists() and task_text:
        gen, _, _ = load_task_cond_ckpt(str(task_ckpt), device)
        return [gen.sample(1, task_text, device)[0] for _ in range(n)], "task_cond"
    if diversity == "low": return [ArchSpec(REF_HIDDEN, REF_DEPTH) for _ in range(n)], "phase3"
    if diversity == "medium": return [random_spec(random.Random(seed)) for _ in range(n)], "phase3"
    gen = ArchGenerator()
    if arch_ckpt.exists():
        gen.load_state_dict(torch.load(arch_ckpt, map_location=device, weights_only=True))
        return gen.to(device).sample(n=n, device=device), "phase3"
    return [random_spec(random.Random(seed + i)) for i in range(n)], "phase3"


def pick_tx_specs(n, task_text, task_ckpt, device, seed):
    if task_ckpt.exists() and task_text:
        gen, _, _ = load_task_tx_ckpt(str(task_ckpt), device)
        return [gen.sample(1, task_text, device)[0] for _ in range(n)], "task_tx"
    return [random_tx_spec(random.Random(seed + i)) for i in range(n)], "random_tx"


def main():
    ex = TX_DEMO_EXAMPLES if "--arch" in sys.argv and "transformer" in sys.argv else DEMO_EXAMPLES
    p = argparse.ArgumentParser(description="Generate tiny models from task description", formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example tasks:\n  " + "\n  ".join(ex))
    p.add_argument("--arch", default="mlp", choices=["mlp", "transformer"])
    p.add_argument("--task", default=DEFAULT_TASK)
    p.add_argument("--list_examples", action="store_true")
    p.add_argument("--diversity", default="high", choices=["low", "medium", "high"])
    p.add_argument("--num", type=int, default=5)
    p.add_argument("--ft_steps", default="")
    p.add_argument("--noise_scale", type=float, default=0.5)
    p.add_argument("--task_ckpt", default="")
    p.add_argument("--cond_ckpt", default="checkpoints/arch_gen/cond_weights_phase3_refine_v1.pt")
    p.add_argument("--arch_ckpt", default="checkpoints/arch_gen/arch_generator_phase3_refine_v1.pt")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.list_examples:
        for t in (TX_DEMO_EXAMPLES if args.arch == "transformer" else DEMO_EXAMPLES): print(t)
        return
    if not args.task_ckpt:
        args.task_ckpt = DEFAULT_TX_TASK_CKPT if args.arch == "transformer" else DEFAULT_TASK_CKPT
    ft_steps = [int(x) for x in args.ft_steps.split(",") if x.strip()]
    set_seed(args.seed)
    device = device_str()
    task_ckpt = ROOT / args.task_ckpt
    print(f"tiny-gen-net demo | arch={args.arch} | task=\"{args.task}\" | n={args.num} | device={device}\n")

    if args.arch == "transformer":
        specs, mode = pick_tx_specs(args.num, args.task, task_ckpt, device, args.seed)
        if mode != "task_tx":
            print(f"Missing {task_ckpt}. Run phase4b_tiny_transformer.py"); sys.exit(1)
        _, wgen, norm_meta = load_task_tx_ckpt(str(task_ckpt), device)
        rows, matches = [], []
        for i, spec in enumerate(specs):
            model = build_from_tx_spec(spec).to(device)
            init_task_tx_weights(model, spec, wgen, args.task, norm_meta, device, args.noise_scale)
            row = eval_tx_row(model, device, ft_steps, args.seed + i)
            rows.append(row)
            ms = tx_spec_matches_task(spec, args.task)
            matches.append(ms)
            print(f"--- model {i} | {tx_arch_summary(spec)} | match={ms:.2f} ref_sim={tx_ref_similarity(spec):.2f} params={model.num_parameters():,} ---")
            print(spec.diagram())
            print(f"  zero_loss={row['zero_loss']:.4f} acc={row['zero_acc']:.3f}\n")
        print(f"Summary: avg zero_loss={sum(r['zero_loss'] for r in rows)/len(rows):.4f} avg_match={sum(matches)/len(matches):.2f}")
        return

    specs, mode = pick_mlp_specs(args.num, args.diversity, device, ROOT / args.arch_ckpt, args.task, task_ckpt, args.seed)
    if mode == "task_cond" and describe_spec(specs[0]): print(f"  canonical: \"{describe_spec(specs[0])}\"\n")
    rows, matches = [], []
    if mode == "task_cond":
        _, wgen, norm_meta = load_task_cond_ckpt(str(task_ckpt), device)
        for i, spec in enumerate(specs):
            model = build_from_spec(spec).to(device)
            init_task_weights(model, spec, wgen, args.task, norm_meta, device, args.noise_scale)
            row = eval_arch_row(model, device, ft_steps, args.seed + i)
            rows.append(row)
            matches.append(spec_matches_task(spec, args.task))
            print(f"--- model {i} | {arch_summary(spec)} | match={matches[-1]:.2f} ref_sim={ref_similarity(spec):.2f} params={model.num_parameters():,} ---")
            print(spec.diagram())
            print(f"  zero_loss={row['zero_loss']:.4f} acc={row['zero_acc']:.3f}\n")
    else:
        cond, cond_norm = load_cond_checkpoint(str(ROOT / args.cond_ckpt), device)
        for i, spec in enumerate(specs):
            model = build_from_spec(spec).to(device)
            init_model_weights(model, spec, cond_gen=cond, cond_norm=cond_norm, device=device, method="conditioned", noise_scale=args.noise_scale)
            row = eval_arch_row(model, device, ft_steps, args.seed + i)
            rows.append(row)
            matches.append(spec_matches_task(spec, args.task))
            print(f"--- model {i} | {arch_summary(spec)} | match={matches[-1]:.2f} params={model.num_parameters():,} ---")
            print(spec.diagram())
            print(f"  zero_loss={row['zero_loss']:.4f}\n")
    print(f"Summary: avg zero_loss={sum(r['zero_loss'] for r in rows)/len(rows):.4f} avg_match={sum(matches)/len(matches):.2f}")


if __name__ == "__main__":
    main()
