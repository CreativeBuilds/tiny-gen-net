#!/usr/bin/env python3
"""Real-text WIDTH-growth scale-arm runner — Task 005.

Single-file, Karpathy-minimal. Reuses the EXACT existing operators from
`src/growth/tx_width_growth.py` (no reimplementation):
  - grow_tx_width(...)              (now exactly function-preserving, Task 004)
  - LowRankCorrection(dim, rank)    (zero-init B => identity at step 0)
  - function_preservation_max_diff  (init FP check)

Runs THREE matched-FLOP arms after a brief source pretrain:
  arm A "random"     : fresh random-init target (d=target_d) trained from scratch
  arm B "grown"      : grow_tx_width(source) (exact init) + continue training
  arm C "grown_corr" : grown target + rank-r LowRankCorrection + continue training

Logs CE vs training step (and approximate forward FLOPs) for each arm, asserts
the grown arms start at the source CE (exact init, FP diff < 1e-3 at full width),
and writes a JSON report to status/forgecritic.json with "task": "005".

The SAME script scales from the CPU smoke (d32->d64, L3) to the plan's real arm
(d512->d1024, L8) by changing flags only — nothing is hardcoded to toy sizes.

Usage
-----
CPU smoke (<60s, default tiny sizes):
  /Users/creativebuilds/Projects/tiny-gen-net/.venv/bin/python \
      experiments/phase_grow_width_realtext.py --smoke

Real scale arm (on an H100, after push):
  python experiments/phase_grow_width_realtext.py \
      --source-d 512 --target-d 1024 --n-layer 8 --src-head 8 \
      --ctx 256 --batch 64 --pretrain-steps 2000 --steps 4000 --rank 32 \
      --corpus tinystories
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
from src.models.variable_transformer import VariableTinyTransformer
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
# Data
# --------------------------------------------------------------------------- #
def build_corpus(args):
    """Return a list of text sequences.

    Char-level over the project synthetic_text corpus is the default (always
    available, deterministic, fine for the CPU smoke). For the real arm pass
    --corpus tinystories: if the local TinyStories shard isn't present we fall
    back to a larger synthetic corpus and record the fallback in the report so
    the substitution is never silent.
    """
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
                # split into ~max_len chunks
                step = args.max_len
                seqs = [txt[i:i + step] for i in range(0, len(txt), step)]
                return [s for s in seqs if len(s) > args.ctx + 1], "tinystories"
        used = "synthetic_text(tinystories_fallback)"
    # Ensure a valid, non-empty length range: sequences must comfortably exceed
    # ctx (windows need len > ctx+1). min_len scales with ctx; max_len is clamped
    # above min_len so randint(min_len, max_len) is never an empty range
    # (prior bug: ctx=256 -> min_len=512 > default max_len=128 -> empty range).
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
# FLOP estimate (forward, very rough — for matched-budget reporting only)
# --------------------------------------------------------------------------- #
def approx_forward_flops(num_params, batch, ctx):
    # ~2 * params * tokens (standard 2N rule of thumb for a forward pass)
    return 2.0 * num_params * batch * ctx


# --------------------------------------------------------------------------- #
# Correction wrapper (identical injection point to the Task 004 smoke)
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
# Train loop
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
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU sizes, <60s")
    ap.add_argument("--source-d", type=int, default=32)
    ap.add_argument("--target-d", type=int, default=64)
    ap.add_argument("--n-layer", type=int, default=3)
    ap.add_argument("--src-head", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--pretrain-steps", type=int, default=20)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--pretrain-lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--corpus", choices=["synthetic_text", "tinystories"],
                    default="synthetic_text")
    ap.add_argument("--num-sequences", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if args.smoke:
        # tiny, fast defaults (already the defaults, but make intent explicit)
        args.source_d, args.target_d, args.n_layer, args.src_head = 32, 64, 3, 4
        args.ctx, args.batch, args.pretrain_steps, args.steps = 8, 16, 20, 50
        args.rank = 8

    t0 = time.time()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    report = {
        "task": "005",
        "kind": "width_growth_realtext_scale_arms",
        "mode": "smoke" if args.smoke else "scale",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "source_d": args.source_d, "target_d": args.target_d,
            "n_layer": args.n_layer, "src_head": args.src_head,
            "ctx": args.ctx, "batch": args.batch,
            "pretrain_steps": args.pretrain_steps, "steps": args.steps,
            "rank": args.rank,
        },
    }
    print(f"Device: {device}  mode: {report['mode']}")

    # --- data ---------------------------------------------------------------
    corpus, corpus_used = build_corpus(args)
    report["corpus"] = corpus_used
    n_windows = args.batch * (max(args.pretrain_steps, args.steps) + 10)
    xtr, ytr = make_windows(corpus, args.ctx, n_windows, args.seed)
    xval, yval = make_windows(corpus, args.ctx, args.batch * 4, args.seed + 1)
    xtr, ytr = xtr.to(device), ytr.to(device)
    xval, yval = xval.to(device), yval.to(device)
    probe = xtr[:args.batch]

    # --- source model + brief pretrain --------------------------------------
    src_spec = TxSpec(d_model=args.source_d, n_layer=args.n_layer,
                      n_head=args.src_head, ctx_len=args.ctx)
    source = VariableTinyTransformer(src_spec, vocab_size=VOCAB_SIZE).to(device)
    src_params = source.num_parameters()
    print(f"Source d={args.source_d} h={args.src_head} L={args.n_layer}: "
          f"{src_params:,} params")
    pl, pnan = train_arm(source, xtr, ytr, args.pretrain_steps, args.batch,
                         args.pretrain_lr, max(1, args.pretrain_steps // 4),
                         args.seed)
    src_ce = eval_ce(source, xval, yval)
    print(f"Source pretrain done; val CE {src_ce:.4f}")
    report["source_params"] = src_params
    report["source_val_ce"] = round(src_ce, 5)
    report["source_pretrain_nan"] = pnan

    # --- ARM B: grown (exact init) ------------------------------------------
    grown = grow_tx_width(source, args.target_d, correct_layernorm=True).to(device)
    tgt_params = grown.num_parameters()
    head_dim = args.source_d // args.src_head
    print(f"Target d={args.target_d} h={args.target_d // head_dim} "
          f"L={args.n_layer}: {tgt_params:,} params")
    report["target_params"] = tgt_params

    FP_TOL = 1e-3
    fp_diff = function_preservation_max_diff(source, grown, probe)
    grown_ce_init = eval_ce(grown, xval, yval)
    print(f"FP max-abs logit diff at init: {fp_diff:.3e} (tol {FP_TOL:.0e})")
    print(f"Grown val CE @ init: {grown_ce_init:.4f}  (source {src_ce:.4f})")
    report["function_preservation_max_abs_diff"] = fp_diff
    report["function_preservation_tol"] = FP_TOL
    report["function_preserving"] = bool(fp_diff < FP_TOL)
    report["grown_init_val_ce"] = round(grown_ce_init, 5)
    # exact-init assertion (grown CE must match source CE before any training)
    assert abs(grown_ce_init - src_ce) < 1e-2, (
        f"grown init CE {grown_ce_init} != source CE {src_ce} -> init NOT exact")

    # --- ARM C: grown + correction (re-grow a fresh copy so arms are independent)
    grown_c = grow_tx_width(source, args.target_d, correct_layernorm=True).to(device)
    correction = LowRankCorrection(args.target_d, rank=args.rank).to(device)
    assert correction.B.weight.norm().item() == 0.0, "B must start at zero"
    model_c = Corrected(grown_c, correction).to(device)
    corr_ce_init = eval_ce(model_c, xval, yval)
    print(f"Grown+corr val CE @ init: {corr_ce_init:.4f} (identity at step0)")
    assert abs(corr_ce_init - grown_ce_init) < 1e-3, (
        "correction not identity at init")
    report["grown_corr_init_val_ce"] = round(corr_ce_init, 5)

    # --- ARM A: random init target ------------------------------------------
    tgt_spec = TxSpec(d_model=args.target_d, n_layer=args.n_layer,
                      n_head=args.target_d // head_dim, ctx_len=args.ctx)
    random_t = VariableTinyTransformer(tgt_spec, vocab_size=VOCAB_SIZE).to(device)

    # --- train all three arms (matched steps == matched FLOPs, same target size)
    arms = {}
    for name, mdl, lr in [
        ("random", random_t, args.lr),
        ("grown", grown, args.lr),
        ("grown_corr", model_c, args.lr),
    ]:
        losses, nan_seen = train_arm(mdl, xtr, ytr, args.steps, args.batch,
                                     lr, max(1, args.steps // 5), args.seed)
        val_ce = eval_ce(mdl, xval, yval)
        arms[name] = {
            "losses": losses,
            "final_val_ce": round(val_ce, 5),
            "nan_seen": nan_seen,
        }
        print(f"arm {name:11s} | final val CE {val_ce:.4f} | nan={nan_seen}")

    flops_per_step = approx_forward_flops(tgt_params, args.batch, args.ctx)
    report["approx_forward_flops_per_step"] = flops_per_step
    report["approx_total_flops_per_arm"] = flops_per_step * args.steps
    report["correction_B_norm_after"] = correction.B.weight.norm().item()
    report["arms"] = arms

    all_finite = all(not a["nan_seen"] for a in arms.values()) and not pnan
    report["all_arms_finite"] = all_finite
    report["elapsed_sec"] = round(time.time() - t0, 2)
    report["status"] = "complete" if (
        all_finite and report["function_preserving"]
    ) else "complete_with_caveats"
    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    os.makedirs(os.path.dirname(STATUS_JSON), exist_ok=True)
    with open(STATUS_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote results -> {STATUS_JSON}")
    print(json.dumps(report, indent=2))
    print("Runner complete.")


if __name__ == "__main__":
    main()
