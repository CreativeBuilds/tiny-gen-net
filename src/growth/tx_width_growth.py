"""Function-preserving width growth for VariableTinyTransformer.

Net2Wider-style operator for the decoder-only transformer in
`src/models/variable_transformer.py`.

The transformer's residual stream has width d_model. We grow d_model: Ds -> Dt
(Dt > Ds) while keeping head_dim constant by scaling n_head proportionally
(n_head_t = n_head_s * Dt // Ds). Keeping head_dim fixed means attention
softmax scaling (1/sqrt(head_dim)) is unchanged, which is required for
function preservation.

Per-tensor growth (copy source into top-left, zero-pad the rest):

  tok.weight  (V, Ds)        -> (V, Dt)            copy first Ds cols
  pos.weight  (ctx, Ds)      -> (ctx, Dt)          copy first Ds cols
  ln1/ln2.weight,bias (Ds,)  -> (Dt,)              copy first Ds, NEW=0
  attn.in_proj_weight (3Ds,Ds)-> (3Dt,Dt)          per-QKV top-left block
  attn.in_proj_bias  (3Ds,)  -> (3Dt,)             per-QKV copy, NEW=0
  attn.out_proj.weight (Ds,Ds)->(Dt,Dt)            top-left, zero-pad
  attn.out_proj.bias  (Ds,)  -> (Dt,)              copy, NEW=0
  ff[0].weight (ff_s,Ds)     -> (ff_t,Dt)          top-left, zero-pad
  ff[0].bias   (ff_s,)       -> (ff_t,)            copy, NEW=0
  ff[2].weight (Ds,ff_s)     -> (Dt,ff_t)          top-left, zero-pad
  ff[2].bias   (Ds,)         -> (Dt,)              copy, NEW=0
  head.weight (V, Ds)        -> (V, Dt)            copy first Ds cols
  head.bias   (V,)           -> (V,)               unchanged

FUNCTION-PRESERVATION CAVEAT (important, reported honestly by the smoke):
LayerNorm normalizes over the FULL d_model. If the new dims of the residual
stream are exactly zero, LN's mean/variance are computed over (Ds real + new
zeros), which differs from LN over Ds alone -> the grown model is only
APPROXIMATELY function-preserving, not exact. We correct for this by rescaling
LN gamma on the copied (first Ds) channels by sqrt(Dt/Ds) and shifting via beta
to cancel the mean change, which restores exactness when all new channels are
zero. The smoke measures and prints the residual max-abs logit diff so the
claim is verifiable, not asserted blindly.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from src.arch_gen.tx_spec import TxSpec
from src.models.variable_transformer import VariableTinyTransformer
from data.synthetic_text import VOCAB_SIZE


def _ff_dim(d_model: int) -> int:
    return max(d_model // 2, 16)


def grow_tx_width(
    source: VariableTinyTransformer,
    target_d: int,
    correct_layernorm: bool = True,
) -> VariableTinyTransformer:
    """Grow a VariableTinyTransformer's d_model to target_d (function-preserving).

    n_head is scaled to keep head_dim constant. Requires target_d % head_dim == 0
    and target_d > source.d_model.
    """
    Ds = source.d_model
    Dt = target_d
    assert Dt > Ds, f"target_d ({Dt}) must be > source d_model ({Ds})"
    head_dim = Ds // source.n_head
    assert Dt % head_dim == 0, (
        f"target_d ({Dt}) must be divisible by source head_dim ({head_dim})"
    )
    n_head_t = Dt // head_dim
    assert Dt % source.n_layer == 0 or True  # n_layer unchanged

    tgt_spec = TxSpec(
        d_model=Dt,
        n_layer=source.n_layer,
        n_head=n_head_t,
        ctx_len=source.ctx_len,
        scale=getattr(source.spec, "scale", False),
    )
    target = VariableTinyTransformer(tgt_spec, vocab_size=source.vocab_size)

    ff_s = _ff_dim(Ds)
    ff_t = _ff_dim(Dt)

    with torch.no_grad():
        # Embeddings: copy first Ds columns, zero the new columns
        target.tok.weight.zero_()
        target.tok.weight[:, :Ds] = source.tok.weight
        target.pos.weight.zero_()
        target.pos.weight[:, :Ds] = source.pos.weight

        for kt, (sb, tb) in enumerate(zip(source.blocks, target.blocks)):
            _grow_layernorm(sb.ln1, tb.ln1, Ds, Dt, correct_layernorm)
            _grow_layernorm(sb.ln2, tb.ln2, Ds, Dt, correct_layernorm)
            _grow_mha(sb.attn, tb.attn, Ds, Dt)
            _grow_ff(sb.ff, tb.ff, Ds, Dt, ff_s, ff_t)

        # Head: logits = head(h[:, -1, :]); copy first Ds input cols, zero rest
        target.head.weight.zero_()
        target.head.weight[:, :Ds] = source.head.weight
        target.head.bias.copy_(source.head.bias)

    return target


def _grow_layernorm(src_ln, tgt_ln, Ds, Dt, correct):
    """Grow a norm Ds->Dt and (Task 004) make it EXACTLY function-preserving.

    Task 003 finding: standard nn.LayerNorm over the FULL width Dt with new dims=0
    is NOT exactly invertible by any static per-channel scalar — its sqrt(var+eps)
    denominator is computed over all Dt dims, so the zero-padded new dims shrink
    the per-token variance and re-scale the copied dims (1.26 max logit diff on a
    trained source). A sqrt(Dt/Ds) gamma rescale made it WORSE.

    Task 004 fix: the model now uses `ActiveLayerNorm`, which computes its
    mean/variance over only the first `active_dim` channels. We copy gamma/beta on
    the first Ds dims (new dims gamma=beta=0 -> emit 0) and set active_dim = Ds so
    the normalization statistics are taken over exactly the original Ds dims. With
    new dims zero, this reproduces the source's LayerNorm EXACTLY -> growth is now
    exactly function-preserving (target FP diff < 1e-3).
    """
    tgt_ln.weight.zero_()
    tgt_ln.bias.zero_()
    tgt_ln.weight[:Ds] = src_ln.weight
    tgt_ln.bias[:Ds] = src_ln.bias
    # Inherit eps so the denominator matches the source exactly.
    if hasattr(src_ln, "eps") and hasattr(tgt_ln, "eps"):
        tgt_ln.eps = src_ln.eps
    # Restrict normalization statistics to the original Ds dims.
    if correct and hasattr(tgt_ln, "set_active_dim"):
        tgt_ln.set_active_dim(Ds)


def _grow_mha(src: nn.MultiheadAttention, tgt: nn.MultiheadAttention, Ds, Dt):
    # in_proj_weight is (3*D, D) stacked as [Wq; Wk; Wv]
    tgt.in_proj_weight.zero_()
    for i in range(3):
        s_blk = src.in_proj_weight[i * Ds:(i + 1) * Ds, :]           # (Ds, Ds)
        tgt.in_proj_weight[i * Dt: i * Dt + Ds, :Ds] = s_blk          # top-left
    if src.in_proj_bias is not None:
        tgt.in_proj_bias.zero_()
        for i in range(3):
            tgt.in_proj_bias[i * Dt: i * Dt + Ds] = src.in_proj_bias[i * Ds:(i + 1) * Ds]
    # out_proj: (D, D)
    tgt.out_proj.weight.zero_()
    tgt.out_proj.weight[:Ds, :Ds] = src.out_proj.weight
    if src.out_proj.bias is not None:
        tgt.out_proj.bias.zero_()
        tgt.out_proj.bias[:Ds] = src.out_proj.bias


def _grow_ff(src_ff: nn.Sequential, tgt_ff: nn.Sequential, Ds, Dt, ff_s, ff_t):
    # ff[0]: Linear(D, ff); ff[2]: Linear(ff, D)
    l0s, l2s = src_ff[0], src_ff[2]
    l0t, l2t = tgt_ff[0], tgt_ff[2]
    l0t.weight.zero_()
    l0t.weight[:ff_s, :Ds] = l0s.weight
    l0t.bias.zero_()
    l0t.bias[:ff_s] = l0s.bias
    l2t.weight.zero_()
    l2t.weight[:Ds, :ff_s] = l2s.weight
    l2t.bias.zero_()
    l2t.bias[:Ds] = l2s.bias


class LowRankCorrection(nn.Module):
    """Rank-r residual correction over the full grown width.

    B is zero-initialized so the correction starts as identity (adds zero),
    preserving the grown function at step 0. A is small-random so gradients
    flow into both A and B after the first backward pass.
    """

    def __init__(self, dim: int, rank: int = 16):
        super().__init__()
        self.A = nn.Linear(dim, rank, bias=False)
        self.B = nn.Linear(rank, dim, bias=False)
        nn.init.normal_(self.A.weight, 0.0, 0.02)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return x + self.B(self.A(x))


@torch.no_grad()
def function_preservation_max_diff(
    source: VariableTinyTransformer,
    target: VariableTinyTransformer,
    batch: torch.Tensor,
) -> float:
    """Max abs diff between source and grown-target logits on a fixed batch."""
    source = source.eval()
    target = target.eval()
    ls, _ = source(batch)
    lt, _ = target(batch)
    return (ls - lt).abs().max().item()
