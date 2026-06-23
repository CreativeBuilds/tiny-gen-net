#!/usr/bin/env python3
"""Iterative (multi-step) width growth — Task 007.

Tests whether function-preserving width growth can be applied ITERATIVELY:
  d512 -> d768 -> d1024 (two growth steps with intermediate training)

This answers:
  1. Does FP hold across multiple growth steps? (error accumulation)
  2. Does iterative growth (with intermediate training) beat single-step?
  3. Does the correction layer compound productively or degrade?

KEY INSIGHT for ActiveLayerNorm:
  After growing d512->d768, the growth operator sets active_dim=512 on all
  LayerNorms. After training the d768 model, those LayerNorms STILL have
  active_dim=512 (the model learned with partial normalization). When growing
  d768->d1024, the growth operator would set active_dim=768, but the source
  was trained with active_dim=512 -> FP would break. FIX: after growing to
  d1024, manually set active_dim=512 on all LayerNorms to match the source's
  normalization statistics. This preserves FP because:
    - Dims 0-511: same values, same normalization stats (from dims 0-511) -> same output
    - Dims 512-767: same values, same normalization stats -> same output
    - Dims 768-1023: zero-padded, gamma=0 -> output 0 (doesn't affect residual)

Arms (all matched at d1024, same total d1024 training FLOPs):
  A "random"       : fresh random-init d1024, trained 2000 steps
  B "single_step"  : grow d512->d1024 (FP), train 2000 steps
  C "iterative"    : grow d512->d768 (FP), train 1000; grow d768->d1024 (FP), train 1000
  D "iterative_corr": same as C but with rank-r correction after final growth

Usage
-----
CPU smoke (<60s):
  python experiments/phase_iterative_growth.py --smoke

Real scale (on H100):
  python experiments/phase_iterative_growth.py \
      --source-d 512 --mid-d 768 --target-d 1024 --n-layer 8 --src-head 8 \
      --ctx 256 --batch 64 --pretrain-steps 2000 --mid-steps 1000 --steps 1000 \
      --rank 32 --pretrain-lr 3e-4 --lr 3e-4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn

from src.arch_gen.tx_spec import TxSpec
from src.models.variable_transformer import VariableTinyTransformer, ActiveLayerNorm
from src.growth.tx_width_growth import (
    grow_tx_width,
    LowRankCorrection,
    function_preservation_max_diff,
)
from data.synthetic_text import (
    SyntheticConfig, generate_corpus, encode_char, VOCAB_SIZE,
)

STATUS_JSON = os.path.expanduser(
    "~/.hermes-chats/tiny-gen-net/status/forgecritic.json"
)


# --------------------------------------------------------------------------- #
# Data (identical to phase_grow_width_realtext.py)
# --------------------------------------------------------------------------- #
def build_corpus(args):
    used = "synthetic_text"
    if args.corpus == "tinystories":
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "data", "tinystories.txt"),
            os.path.expanduser("~/data/tinystories.txt"),
        ]
        for p in candidates:
            if os.path.isfile(p):
                with open(p, "r", errors="ignore") as f:
                    txt = f.read()
                step = args.max_len
                seqs = [txt[i:i + step] for i in range(0, len(txt), step)]
                return [s for s in seqs if len(s) > args.ctx + 1], "tinystories"
        used = "synthetic_text(tinystories_fallback)"
    min_len = max(args.ctx * 2, 64)
    max_len = max(args.max_len, min_len * 2)
    corpus = generate_corpus(SyntheticConfig(
        num_sequences=args.num_sequences,
        min_len=min_len,
        max_len=max_len,
    ))
    return corpus, used


def make_windows(corpus, ctx, n_windows, seed):
    rng = torch.Generator().manual_seed(seed)
    seqs = [[encode_char(c) for c in t] for t in corpus if len(t) > ctx]
    if not seqs:
        raise RuntimeError("corpus produced no sequences longer than ctx")
    xs, ys = [], []
    while len(xs) < n_windows:
        si = int(torch.randint(len(seqs), (1,), generator=rng))
        s = seqs[si]
        if len(s) <= ctx:
            continue
        t = int(torch.randint(len(s) - ctx, (1,), generator=rng))
        xs.append(s[t:t + ctx])
        ys.append(s[t + ctx])
    return (torch.tensor(xs, dtype=torch.long),
            torch.tensor(ys, dtype=torch.long))


# --------------------------------------------------------------------------- #
# Correction wrapper (identical injection point)
# --------------------------------------------------------------------------- #
class Corrected(nn.Module):
    def __init__(self, base, corr):
        super().__init__()
        self.base, self.corr = base, corr

    def forward(self, x):
        B, T = x.shape
        T = min(T, self.base.ctx_len)
        x = x[:, -T:]
        pos = torch.arange(T, device=x.device)
        h = self.base.tok(x) + self.base.pos(pos).unsqueeze(0)
        h = self.corr(h)  # x + low-rank; zero at step0
        mask = self.base._causal_mask(T, x.device)
        for blk in self.base.blocks:
            h = blk(h, mask)
        return self.base.head(h[:, -1, :]), None


# --------------------------------------------------------------------------- #
# Train loop (identical)
# --------------------------------------------------------------------------- #
def train_arm(model, xtr, ytr, steps, batch, lr, log_every, seed):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    losses, nan_seen = {}, False
    n = xtr.shape[0]
    for step in range(steps):
        off = (step * batch) % max(1, (n - batch))
        xb, yb = xtr[off:off + batch], ytr[off:off + batch]
        logits, _ = model(xb)
        loss = nn.functional.cross_entropy(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        if torch.isnan(loss):
            nan_seen = True
            losses[str(step)] = float("nan")
            break
        if step % log_every == 0 or step == steps - 1:
            losses[str(step)] = round(loss.item(), 5)
    return losses, nan_seen


@torch.no_grad()
def eval_ce(model, xb, yb):
    model.eval()
    logits, _ = model(xb)
    return nn.functional.cross_entropy(logits, yb).item()


# --------------------------------------------------------------------------- #
# Helper: set active_dim on all LayerNorms in a model
# --------------------------------------------------------------------------- #
def set_all_active_dim(model: VariableTinyTransformer, dim: int):
    """Set active_dim on every ActiveLayerNorm in the model."""
    for m in model.modules():
        if isinstance(m, ActiveLayerNorm):
            m.set_active_dim(dim)


def get_active_dims(model: VariableTinyTransformer) -> list:
    """Read active_dim from all ActiveLayerNorms (for reporting)."""
    return [int(m.active_dim) for m in model.modules()
            if isinstance(m, ActiveLayerNorm)]


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny CPU sizes, <60s")
    ap.add_argument("--source-d", type=int, default=32)
    ap.add_argument("--mid-d", type=int, default=48)
    ap.add_argument("--target-d", type=int, default=64)
    ap.add_argument("--n-layer", type=int, default=3)
    ap.add_argument("--src-head", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--pretrain-steps", type=int, default=20)
    ap.add_argument("--mid-steps", type=int, default=25)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pretrain-lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--corpus", choices=["synthetic_text", "tinystories"],
                    default="synthetic_text")
    ap.add_argument("--num-sequences", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if args.smoke:
        args.source_d, args.mid_d, args.target_d = 32, 48, 64
        args.n_layer, args.src_head = 3, 4
        args.ctx, args.batch = 8, 16
        args.pretrain_steps, args.mid_steps, args.steps = 20, 25, 50
        args.rank = 8

    t0 = time.time()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    head_dim = args.source_d // args.src_head

    report = {
        "task": "007",
        "kind": "iterative_width_growth",
        "mode": "smoke" if args.smoke else "scale",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "source_d": args.source_d, "mid_d": args.mid_d,
            "target_d": args.target_d,
            "n_layer": args.n_layer, "src_head": args.src_head,
            "head_dim": head_dim,
            "ctx": args.ctx, "batch": args.batch,
            "pretrain_steps": args.pretrain_steps,
            "mid_steps": args.mid_steps,
            "steps": args.steps,
            "rank": args.rank,
            "pretrain_lr": args.pretrain_lr, "lr": args.lr,
        },
    }
    print(f"Device: {device}  mode: {report['mode']}")
    print(f"Growth path: d{args.source_d} -> d{args.mid_d} -> d{args.target_d}")

    # --- data ---------------------------------------------------------------
    corpus, corpus_used = build_corpus(args)
    report["corpus"] = corpus_used
    total_steps = max(args.pretrain_steps, args.mid_steps + args.steps)
    n_windows = args.batch * (total_steps + 10)
    xtr, ytr = make_windows(corpus, args.ctx, n_windows, args.seed)
    xval, yval = make_windows(corpus, args.ctx, args.batch * 4, args.seed + 1)
    xtr, ytr = xtr.to(device), ytr.to(device)
    xval, yval = xval.to(device), yval.to(device)
    probe = xtr[:args.batch]

    FP_TOL = 1e-3

    # ===================================================================== #
    # SOURCE: pretrain d512
    # ===================================================================== #
    src_spec = TxSpec(d_model=args.source_d, n_layer=args.n_layer,
                      n_head=args.src_head, ctx_len=args.ctx, scale=True)
    source = VariableTinyTransformer(src_spec, vocab_size=VOCAB_SIZE).to(device)
    src_params = source.num_parameters()
    print(f"\nSource d={args.source_d} h={args.src_head} L={args.n_layer}: "
          f"{src_params:,} params")
    pl, pnan = train_arm(source, xtr, ytr, args.pretrain_steps, args.batch,
                         args.pretrain_lr, max(1, args.pretrain_steps // 4),
                         args.seed)
    src_ce = eval_ce(source, xval, yval)
    print(f"Source pretrain done; val CE {src_ce:.4f}")
    report["source_params"] = src_params
    report["source_val_ce"] = round(src_ce, 5)
    report["source_pretrain_nan"] = pnan

    # ===================================================================== #
    # ITERATIVE PATH: d512 -> d768 -> d1024
    # ===================================================================== #
    print("\n--- ITERATIVE PATH ---")

    # Step 1: grow d_source -> d_mid
    mid_model = grow_tx_width(source, args.mid_d, correct_layernorm=True).to(device)
    mid_params = mid_model.num_parameters()
    mid_head = args.mid_d // head_dim
    print(f"Step 1: grew d{args.source_d} -> d{args.mid_d} (h={mid_head}, "
          f"{mid_params:,} params)")
    fp1 = function_preservation_max_diff(source, mid_model, probe)
    mid_ce_init = eval_ce(mid_model, xval, yval)
    print(f"  FP step 1: {fp1:.3e} (tol {FP_TOL:.0e}) | CE @ init: {mid_ce_init:.4f}")
    report["iterative"] = {
        "step1": {
            "from_d": args.source_d, "to_d": args.mid_d,
            "params": mid_params,
            "fp_max_abs_diff": fp1,
            "function_preserving": bool(fp1 < FP_TOL),
            "ce_at_init": round(mid_ce_init, 5),
            "active_dims": get_active_dims(mid_model),
        }
    }
    assert abs(mid_ce_init - src_ce) < 1e-2, (
        f"iterative step 1 init CE {mid_ce_init} != source CE {src_ce}")

    # Train d_mid for mid_steps
    mid_losses, mid_nan = train_arm(mid_model, xtr, ytr, args.mid_steps,
                                    args.batch, args.lr,
                                    max(1, args.mid_steps // 4), args.seed)
    mid_ce_trained = eval_ce(mid_model, xval, yval)
    print(f"  Trained d{args.mid_d} for {args.mid_steps} steps; "
          f"CE {mid_ce_trained:.4f}")
    report["iterative"]["step1"]["ce_after_train"] = round(mid_ce_trained, 5)
    report["iterative"]["step1"]["train_losses"] = mid_losses
    report["iterative"]["step1"]["train_nan"] = mid_nan

    # Step 2: grow d_mid -> d_target
    # KEY: preserve the source's active_dim to maintain FP
    iter_model = grow_tx_width(mid_model, args.target_d, correct_layernorm=True).to(device)
    # The growth operator set active_dim = mid_d on all norms.
    # But mid_model was trained with active_dim = source_d (from step 1 growth).
    # Fix: set active_dim = source_d to match what mid_model was trained with.
    source_active = int(mid_model.blocks[0].ln1.active_dim)
    if source_active != args.mid_d:
        print(f"  NOTE: mid_model active_dim={source_active} (not {args.mid_d}). "
              f"Setting target active_dim={source_active} to preserve FP.")
        set_all_active_dim(iter_model, source_active)

    tgt_params = iter_model.num_parameters()
    tgt_head = args.target_d // head_dim
    print(f"Step 2: grew d{args.mid_d} -> d{args.target_d} (h={tgt_head}, "
          f"{tgt_params:,} params)")
    # FP check: compare mid_model vs iter_model
    fp2 = function_preservation_max_diff(mid_model, iter_model, probe)
    iter_ce_init = eval_ce(iter_model, xval, yval)
    print(f"  FP step 2: {fp2:.3e} (tol {FP_TOL:.0e}) | CE @ init: {iter_ce_init:.4f}")
    report["iterative"]["step2"] = {
        "from_d": args.mid_d, "to_d": args.target_d,
        "params": tgt_params,
        "fp_max_abs_diff": fp2,
        "function_preserving": bool(fp2 < FP_TOL),
        "ce_at_init": round(iter_ce_init, 5),
        "active_dims": get_active_dims(iter_model),
        "source_active_dim_preserved": source_active,
    }
    assert abs(iter_ce_init - mid_ce_trained) < 1e-2, (
        f"iterative step 2 init CE {iter_ce_init} != mid trained CE {mid_ce_trained}")

    # ===================================================================== #
    # ITERATIVE + CORRECTION: same growth path, add correction after step 2
    # ===================================================================== #
    iter_model_c = grow_tx_width(mid_model, args.target_d, correct_layernorm=True).to(device)
    if source_active != args.mid_d:
        set_all_active_dim(iter_model_c, source_active)
    correction = LowRankCorrection(args.target_d, rank=args.rank).to(device)
    assert correction.B.weight.norm().item() == 0.0, "B must start at zero"
    model_iter_corr = Corrected(iter_model_c, correction).to(device)
    corr_ce_init = eval_ce(model_iter_corr, xval, yval)
    print(f"  Iterative+corr CE @ init: {corr_ce_init:.4f} (identity at step0)")
    assert abs(corr_ce_init - iter_ce_init) < 1e-3, "correction not identity at init"
    report["iterative_corr_init_val_ce"] = round(corr_ce_init, 5)

    # ===================================================================== #
    # SINGLE-STEP PATH: d_source -> d_target (for comparison)
    # ===================================================================== #
    print("\n--- SINGLE-STEP PATH ---")
    single_model = grow_tx_width(source, args.target_d, correct_layernorm=True).to(device)
    fp_single = function_preservation_max_diff(source, single_model, probe)
    single_ce_init = eval_ce(single_model, xval, yval)
    print(f"Single-step d{args.source_d} -> d{args.target_d}: "
          f"FP {fp_single:.3e} | CE @ init: {single_ce_init:.4f}")
    report["single_step"] = {
        "from_d": args.source_d, "to_d": args.target_d,
        "params": single_model.num_parameters(),
        "fp_max_abs_diff": fp_single,
        "function_preserving": bool(fp_single < FP_TOL),
        "ce_at_init": round(single_ce_init, 5),
        "active_dims": get_active_dims(single_model),
    }

    # ===================================================================== #
    # RANDOM INIT: fresh d_target
    # ===================================================================== #
    print("\n--- RANDOM INIT ---")
    tgt_spec = TxSpec(d_model=args.target_d, n_layer=args.n_layer,
                      n_head=tgt_head, ctx_len=args.ctx, scale=True)
    random_t = VariableTinyTransformer(tgt_spec, vocab_size=VOCAB_SIZE).to(device)
    print(f"Random d{args.target_d} h={tgt_head}: "
          f"{random_t.num_parameters():,} params")

    # ===================================================================== #
    # TRAIN ALL ARMS (matched total d_target training FLOPs)
    # ===================================================================== #
    # iterative: mid_steps at d_mid + steps at d_target (steps == single-step steps)
    # single-step: steps at d_target
    # random: steps at d_target
    # All have the SAME d_target training steps (args.steps), so d_target FLOPs match.
    # The iterative arm has bonus d_mid training (args.mid_steps) at lower FLOPs.
    arms = {}
    total_target_steps = args.steps  # all arms train this many steps at d_target

    for name, mdl, train_steps in [
        ("random", random_t, total_target_steps),
        ("single_step", single_model, total_target_steps),
        ("iterative", iter_model, total_target_steps),
        ("iterative_corr", model_iter_corr, total_target_steps),
    ]:
        losses, nan_seen = train_arm(mdl, xtr, ytr, train_steps, args.batch,
                                     args.lr, max(1, train_steps // 5),
                                     args.seed)
        val_ce = eval_ce(mdl, xval, yval)
        arms[name] = {
            "losses": losses,
            "final_val_ce": round(val_ce, 5),
            "nan_seen": nan_seen,
            "target_train_steps": train_steps,
        }
        print(f"arm {name:16s} | final val CE {val_ce:.4f} | nan={nan_seen}")

    report["arms"] = arms
    report["correction_B_norm_after"] = correction.B.weight.norm().item()

    all_finite = all(not a["nan_seen"] for a in arms.values()) and not pnan
    report["all_arms_finite"] = all_finite
    report["all_fp_hold"] = bool(
        report["iterative"]["step1"]["function_preserving"] and
        report["iterative"]["step2"]["function_preserving"] and
        report["single_step"]["function_preserving"]
    )

    # Summary
    r_ce = arms["random"]["final_val_ce"]
    s_ce = arms["single_step"]["final_val_ce"]
    i_ce = arms["iterative"]["final_val_ce"]
    ic_ce = arms["iterative_corr"]["final_val_ce"]
    report["summary"] = {
        "fp_step1_diff": fp1,
        "fp_step2_diff": fp2,
        "fp_single_step_diff": fp_single,
        "random_ce": r_ce,
        "single_step_ce": s_ce,
        "iterative_ce": i_ce,
        "iterative_corr_ce": ic_ce,
        "single_vs_random_relative": round((s_ce - r_ce) / r_ce, 4),
        "iterative_vs_random_relative": round((i_ce - r_ce) / r_ce, 4),
        "iterative_corr_vs_random_relative": round((ic_ce - r_ce) / r_ce, 4),
        "iterative_vs_single_relative": round((i_ce - s_ce) / s_ce, 4),
        "iterative_corr_vs_single_relative": round((ic_ce - s_ce) / s_ce, 4),
        "correction_learned": bool(correction.B.weight.norm().item() > 0.01),
        "key_finding": (
            f"Iterative growth d{args.source_d}->d{args.mid_d}->d{args.target_d}: "
            f"FP step1={fp1:.2e}, step2={fp2:.2e}, single={fp_single:.2e}. "
            f"CE: random={r_ce:.4f}, single={s_ce:.4f}, iterative={i_ce:.4f}, "
            f"iter+corr={ic_ce:.4f}. "
            f"{'FP HOLDS at both steps.' if report['all_fp_hold'] else 'FP BREAKS at one or more steps.'}"
        ),
    }

    report["elapsed_sec"] = round(time.time() - t0, 2)
    report["status"] = "complete" if (
        all_finite and report["all_fp_hold"]
    ) else "complete_with_caveats"
    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    os.makedirs(os.path.dirname(STATUS_JSON), exist_ok=True)
    with open(STATUS_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote results -> {STATUS_JSON}")
    print(json.dumps(report["summary"], indent=2))
    print("Iterative growth runner complete.")


if __name__ == "__main__":
    main()
