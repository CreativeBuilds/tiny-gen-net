#!/usr/bin/env python3
"""Local CPU smoke: function-preserving WIDTH growth + low-rank correction.

Task 003 deliverable. Single-file, Karpathy-minimal. NO pods — CPU only.

Grows a VariableTinyTransformer d_model 48 -> 96 (n_head 4 -> 8, head_dim=12
constant, n_layer=3, ctx=8 — all valid under TxSpec/MAX_TX_PARAMS), verifies the
grown model reproduces the source function on a fixed batch BEFORE training,
attaches a zero-init low-rank correction, and runs a 50-step CPU train loop.

Run (from worktree root, using the project venv that has torch):
  /Users/creativebuilds/Projects/tiny-gen-net/.venv/bin/python \
      experiments/phase_grow_width_realtext_smoke.py
"""

import json
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

CTX = 8
BATCH = 16
SOURCE_D = 32
TARGET_D = 64      # valid width-growth: both in D_MODEL_CHOICES, target <100k params
N_LAYER = 3
SRC_HEAD = 4       # head_dim = 32/4 = 8 ; target n_head = 64/8 = 8
STEPS = 50
RANK = 8
SEED = 1234

STATUS_JSON = os.path.expanduser(
    "~/.hermes-chats/tiny-gen-net/status/forgecritic.json"
)

device = torch.device("cpu")


def make_windows(corpus, ctx, n_windows, seed=SEED):
    """Build (input_window, next_char) pairs of length `ctx`."""
    rng = torch.Generator().manual_seed(seed)
    seqs = [[encode_char(c) for c in t] for t in corpus if len(t) > ctx]
    xs, ys = [], []
    while len(xs) < n_windows and seqs:
        si = int(torch.randint(len(seqs), (1,), generator=rng))
        s = seqs[si]
        if len(s) <= ctx:
            continue
        t = int(torch.randint(len(s) - ctx, (1,), generator=rng))
        xs.append(s[t:t + ctx])
        ys.append(s[t + ctx])
    x = torch.tensor(xs, dtype=torch.long)
    y = torch.tensor(ys, dtype=torch.long)
    return x, y


def main():
    t0 = time.time()
    torch.manual_seed(SEED)
    report = {"task": "004", "kind": "width_growth_cpu_smoke",
              "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    print(f"Device: {device}")

    # --- source model -------------------------------------------------------
    src_spec = TxSpec(d_model=SOURCE_D, n_layer=N_LAYER, n_head=SRC_HEAD, ctx_len=CTX)
    source = VariableTinyTransformer(src_spec, vocab_size=VOCAB_SIZE).to(device)
    src_params = source.num_parameters()
    print(f"Source d={SOURCE_D} h={SRC_HEAD} L={N_LAYER}: {src_params:,} params")

    # Lightly pre-train source so growth is non-trivial (not random) ----------
    corpus = generate_corpus(SyntheticConfig(num_sequences=128, min_len=64, max_len=128))
    xtr, ytr = make_windows(corpus, CTX, BATCH * (STEPS + 10))
    opt_s = torch.optim.AdamW(source.parameters(), lr=3e-3)
    source.train()
    for i in range(20):
        xb = xtr[i * BATCH:(i + 1) * BATCH]
        yb = ytr[i * BATCH:(i + 1) * BATCH]
        logits, _ = source(xb)
        loss = nn.functional.cross_entropy(logits, yb)
        opt_s.zero_grad(); loss.backward(); opt_s.step()
    print(f"Source warm pre-train loss: {loss.item():.4f}")

    # --- grow width (function-preserving) -----------------------------------
    target = grow_tx_width(source, TARGET_D, correct_layernorm=True).to(device)
    tgt_params = target.num_parameters()
    print(f"Target d={TARGET_D} h={TARGET_D // (SOURCE_D // SRC_HEAD)} L={N_LAYER}: {tgt_params:,} params")

    # function-preservation check BEFORE any training, fixed batch.
    # Task 004: ActiveLayerNorm normalizes over the original Ds dims only, so
    # zero-pad width growth is now EXACTLY function-preserving. We report both the
    # "before" diff (full-width LayerNorm stats, active_dim=Dt — the Task 003 bug)
    # and the "after" diff (active_dim=Ds — the fix), to make the improvement
    # verifiable rather than asserted.
    FP_TOL = 1e-3
    probe = xtr[:BATCH]

    # BEFORE: simulate the old full-width-LayerNorm behavior by setting every
    # grown norm's active_dim back to the full target width Dt.
    for blk in target.blocks:
        blk.ln1.set_active_dim(target.d_model)
        blk.ln2.set_active_dim(target.d_model)
    fp_diff_before = function_preservation_max_diff(source, target, probe)
    # AFTER: restore the fix (stats over original Ds dims).
    for blk in target.blocks:
        blk.ln1.set_active_dim(SOURCE_D)
        blk.ln2.set_active_dim(SOURCE_D)
    fp_diff = function_preservation_max_diff(source, target, probe)
    print(f"FP max-abs logit diff BEFORE (full-width LN): {fp_diff_before:.3e}")
    print(f"FP max-abs logit diff AFTER  (active_dim=Ds):  {fp_diff:.3e} (tol {FP_TOL:.0e})")
    report["source_params"] = src_params
    report["target_params"] = tgt_params
    report["function_preservation_max_abs_diff_before"] = fp_diff_before
    report["function_preservation_max_abs_diff"] = fp_diff
    report["function_preservation_tol"] = FP_TOL
    report["function_preserving"] = bool(fp_diff < FP_TOL)
    report["fp_fix"] = "ActiveLayerNorm: normalize over original Ds dims only -> exact preservation"

    # --- attach low-rank correction (zero-init B => identity at step0) -------
    correction = LowRankCorrection(TARGET_D, rank=RANK).to(device)
    b_norm_before = correction.B.weight.norm().item()
    assert b_norm_before == 0.0, f"B should start at zero, got {b_norm_before}"

    # wire correction into the residual stream after embeddings via a hook-free
    # wrapper: apply correction to token embeddings input (simplest smoke point)
    class Corrected(nn.Module):
        def __init__(self, base, corr):
            super().__init__()
            self.base, self.corr = base, corr
        def forward(self, x):
            # inject correction on the embedding sum by monkeypatching tok output
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

    model = Corrected(target, correction).to(device)

    # confirm correction is initially identity (logits unchanged)
    with torch.no_grad():
        l_plain, _ = target(probe)
        l_corr, _ = model(probe)
        corr_diff0 = (l_plain - l_corr).abs().max().item()
    print(f"Correction identity check @ step0 (max diff): {corr_diff0:.3e}")
    report["correction_identity_diff_step0"] = corr_diff0
    report["correction_B_norm_before"] = b_norm_before

    # --- 50-step CPU train loop ---------------------------------------------
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses = {}
    model.train()
    nan_seen = False
    for step in range(STEPS):
        xb = xtr[step * BATCH:(step + 1) * BATCH]
        yb = ytr[step * BATCH:(step + 1) * BATCH]
        logits, _ = model(xb)
        loss = nn.functional.cross_entropy(logits, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        if torch.isnan(loss):
            nan_seen = True
            break
        if step in (0, 10, 20, 30, 40, STEPS - 1):
            losses[str(step)] = round(loss.item(), 5)
            print(f"step {step:3d} | loss {loss.item():.4f}")

    b_norm_after = correction.B.weight.norm().item()
    b_grad_norm = (correction.B.weight.grad.norm().item()
                   if correction.B.weight.grad is not None else 0.0)
    print(f"Correction B norm before={b_norm_before:.3e} after={b_norm_after:.3e} "
          f"(grad norm last step {b_grad_norm:.3e})")

    report["losses"] = losses
    report["nan_seen"] = nan_seen
    report["correction_B_norm_after"] = b_norm_after
    report["correction_B_grad_norm_last"] = b_grad_norm
    report["correction_learned"] = bool(b_norm_after > 0 and not nan_seen)
    report["elapsed_sec"] = round(time.time() - t0, 2)
    report["status"] = "complete" if (
        not nan_seen and report["function_preserving"] and b_norm_after > 0
    ) else "complete_with_caveats"
    report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    os.makedirs(os.path.dirname(STATUS_JSON), exist_ok=True)
    with open(STATUS_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote results -> {STATUS_JSON}")
    print(json.dumps(report, indent=2))
    print("Smoke run complete.")


if __name__ == "__main__":
    main()
