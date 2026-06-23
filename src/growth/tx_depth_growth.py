"""Depth-growth for VariableTinyTransformer: function-preserving block insertion.

VariableTinyTransformer forward:
  h = tok(x) + pos(pos)
  for blk in blocks: h = blk(h, mask)     # each block: h = h + attn(ln1(h)) + ff(ln2(h))
  logits = head(h[:, -1, :])

Each block adds a RESIDUAL contribution: x + attn_output + ff_output.
If we insert a block with zero attn output and zero ff output, the residual
stream passes through unchanged: x + 0 + 0 = x. This is function-preserving.

To zero-init a block:
  - attn: MultiheadAttention has out_proj (Linear(d, d)). Zero its weight+bias.
  - ff: nn.Sequential(Linear(d,ff), GELU, Linear(ff,d)). Zero the SECOND Linear's
    weight+bias. (Zeroing the first would also work but zeroing the output is
    cleaner — the intermediate GELU output doesn't matter.)

This is the Net2Net "net2deeper" for residual transformers, same as Gstack/bert2BERT.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.variable_transformer import VariableTinyTransformer
from src.arch_gen.tx_spec import TxSpec


class GrowTxSpec:
    """Bypasses TxSpec validation to allow n_layer > N_LAYER_CHOICES max."""
    def __init__(self, d_model: int, n_layer: int, n_head: int, ctx_len: int = 8):
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head
        self.ctx_len = ctx_len

    def validate(self):
        return self


def build_tx_model(d_model: int, n_layer: int, n_head: int, ctx_len: int = 8,
                   vocab: int = 16) -> VariableTinyTransformer:
    """Build a VariableTinyTransformer with any n_layer (bypasses TxSpec validation)."""
    spec = GrowTxSpec(d_model, n_layer, n_head, ctx_len)
    return VariableTinyTransformer(spec, vocab_size=vocab)


def zero_init_block(block: nn.Module):
    """Zero-init a TxBlock so it outputs zero (identity via residual).

    Zeros: attn.out_proj.weight, attn.out_proj.bias, ff[-1].weight, ff[-1].bias.
    """
    # attn out_proj
    block.attn.out_proj.weight.data.zero_()
    if block.attn.out_proj.bias is not None:
        block.attn.out_proj.bias.data.zero_()
    # ff second linear (last in Sequential)
    block.ff[-1].weight.data.zero_()
    block.ff[-1].bias.data.zero_()
    # ln1, ln2 can stay at default (they don't affect zero output: ln(0)=0 with default affine)


def grow_tx_depth(source_model: VariableTinyTransformer, target_n_layer: int) -> VariableTinyTransformer:
    """Grow source transformer to target_n_layer by inserting zero-init blocks.

    Source blocks 0..n_layer-1 are copied directly.
    New blocks n_layer..target_n_layer-1 are zero-init (identity via residual).

    Returns the target model with grown weights loaded.
    """
    D = source_model.d_model
    H = source_model.n_head
    n_src = source_model.n_layer
    ctx = source_model.ctx_len
    vocab = source_model.vocab_size
    K = target_n_layer - n_src
    assert K > 0, f"target_n_layer ({target_n_layer}) must be > source ({n_src})"

    target = build_tx_model(D, target_n_layer, H, ctx_len=ctx, vocab=vocab)

    # Copy tok, pos, head
    target.tok.weight.data.copy_(source_model.tok.weight.data)
    target.pos.weight.data.copy_(source_model.pos.weight.data)
    target.head.weight.data.copy_(source_model.head.weight.data)
    target.head.bias.data.copy_(source_model.head.bias.data)

    # Copy source blocks
    for k in range(n_src):
        _copy_block(target.blocks[k], source_model.blocks[k])

    # Zero-init new blocks
    for k in range(n_src, target_n_layer):
        zero_init_block(target.blocks[k])

    return target


def _copy_block(dst: nn.Module, src: nn.Module):
    """Copy all parameters from src block to dst block."""
    dst.load_state_dict(src.state_dict())


def add_tx_correction(model: VariableTinyTransformer, source_n_layer: int,
                     rank: int = 8, init_scale: float = 0.05):
    """Add low-rank perturbation to new blocks (indices source_n_layer..n_layer-1).

    Perturbs attn.out_proj.weight and ff[-1].weight (the two output projections
    that determine the block's contribution to the residual stream).
    """
    for k in range(source_n_layer, model.n_layer):
        block = model.blocks[k]

        # Perturb attn.out_proj: (d, d) += A @ B, rank=rank
        W = block.attn.out_proj.weight.data  # (d, d)
        A = torch.randn(W.shape[0], rank, device=W.device) * (1.0 / rank**0.5)
        B = torch.randn(rank, W.shape[1], device=W.device) * init_scale
        W.add_(A @ B)
        if block.attn.out_proj.bias is not None:
            block.attn.out_proj.bias.data.add_(torch.randn_like(block.attn.out_proj.bias.data) * init_scale * 0.1)

        # Perturb ff[-1] (output linear): (d, ff) += A @ B
        W2 = block.ff[-1].weight.data  # (d, ff)
        A2 = torch.randn(W2.shape[0], rank, device=W2.device) * (1.0 / rank**0.5)
        B2 = torch.randn(rank, W2.shape[1], device=W2.device) * init_scale
        W2.add_(A2 @ B2)
        block.ff[-1].bias.data.add_(torch.randn_like(block.ff[-1].bias.data) * init_scale * 0.1)


def verify_tx_growth_preserves_function(
    source_model: VariableTinyTransformer,
    target_model: VariableTinyTransformer,
    n_probe: int = 32,
    atol: float = 1e-4,
    seed: int = 12345,
) -> float:
    """Verify that grown transformer computes the same function as source.

    Runs random context windows through both models and checks max abs logit diff.
    """
    source_model = source_model.cpu().eval()
    target_model = target_model.cpu().eval()

    g = torch.Generator().manual_seed(seed)
    V = source_model.vocab_size
    ctx = source_model.ctx_len

    # Generate random context windows
    x = torch.randint(0, V, (n_probe, ctx), generator=g)

    max_diff = 0.0
    with torch.no_grad():
        logits_s, _ = source_model(x)
        logits_t, _ = target_model(x)
        max_diff = (logits_s - logits_t).abs().max().item()

    assert max_diff < atol, (
        f"TX GROWTH NOT FUNCTION-PRESERVING: max abs logit diff {max_diff:.3e} >= {atol:.1e}."
    )
    return max_diff
