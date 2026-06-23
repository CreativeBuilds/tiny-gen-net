"""Width-growth for VariableTinyMLP: function-preserving width expansion.

Growth operation: given a trained source model (H=H_s), produce a target
model (H=H_t > H_s) where:
  - Source weights are copied into the top-left H_s x H_s block of each
    target matrix (function-preserving: the new hidden units output zero,
    so the source function is preserved)
  - New hidden units (columns/rows H_s..H_t-1) are initialized to produce
    zero output (weight rows = 0 for new output units, so they don't affect
    downstream layers)

This is the Net2Net "net2wider" operation: widen a layer by copying the
existing units and adding new units that output zero. The function is
preserved because the new units contribute nothing.

Specifically for VariableTinyMLP:
  embed.weight: (V, H_s) -> (V, H_t): copy first H_s columns, zero-pad rest
  blocks[0].weight: (H_s, 2*H_s) -> (H_t, 2*H_t):
    - A block (H_s, H_s): copy into top-left, zero-pad rows and cols
    - B block (H_s, H_s): copy into top-left, zero-pad rows and cols
  blocks[0].bias: (H_s,) -> (H_t,): copy first H_s, zero-pad rest
  blocks[k].weight (k>0): (H_s, H_s) -> (H_t, H_t): copy top-left, zero-pad
  blocks[k].bias: (H_s,) -> (H_t,): copy first H_s, zero-pad rest
  head.weight: (V, H_s) -> (V, H_t): copy first H_s columns, zero-pad rest
  head.bias: unchanged (V,)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.variable_mlp import VariableTinyMLP
from data.synthetic_text import VOCAB_SIZE


class GrowSpec:
    """Bypasses ArchSpec validation to allow depth > 3."""
    def __init__(self, hidden_dim: int, depth: int, skip: bool = False):
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.skip = skip

    def validate(self):
        return self


def build_model(hidden: int, depth: int, skip: bool = False, vocab: int = VOCAB_SIZE) -> VariableTinyMLP:
    """Build a VariableTinyMLP with any depth/width."""
    spec = GrowSpec(hidden, depth, skip)
    return VariableTinyMLP(spec, vocab_size=vocab)


def grow_width(source_model: VariableTinyMLP, target_hidden: int) -> VariableTinyMLP:
    """Grow source model to target_hidden width (function-preserving net2wider).

    New hidden units are initialized with zero weights so they output zero,
    preserving the source function exactly.
    """
    Hs = source_model.hidden_dim
    Ht = target_hidden
    D = source_model.depth
    assert Ht > Hs, f"target_hidden ({Ht}) must be > source ({Hs})"

    target = build_model(Ht, D, skip=source_model.skip, vocab=source_model.vocab_size)

    # embed.weight: (V, H_s) -> (V, H_t): copy first H_s columns
    target.embed.weight.data.zero_()
    target.embed.weight.data[:, :Hs] = source_model.embed.weight.data

    # blocks[0].weight: (H_s, 2*H_s) -> (H_t, 2*H_t) = [A | B]
    # A (H_s, H_s) -> top-left of (H_t, H_t), rest zero
    # B (H_s, H_s) -> top-left of (H_t, H_t), rest zero
    target.blocks[0].weight.data.zero_()
    sW = source_model.blocks[0].weight.data.view(Hs, 2 * Hs)
    tW = target.blocks[0].weight.data.view(Ht, 2 * Ht)
    tW[:Hs, :Hs] = sW[:, :Hs]          # A block
    tW[:Hs, Ht:Ht + Hs] = sW[:, Hs:2*Hs]  # B block (cols Ht..Ht+Hs-1)
    # The recurrent cols for new units (Hs..Ht-1 in B's input space) are zero

    # blocks[0].bias: copy first H_s, zero-pad
    target.blocks[0].bias.data.zero_()
    target.blocks[0].bias.data[:Hs] = source_model.blocks[0].bias.data

    # blocks[k] (k>0): (H_s, H_s) -> (H_t, H_t): copy top-left, zero-pad
    for k in range(1, D):
        target.blocks[k].weight.data.zero_()
        target.blocks[k].weight.data[:Hs, :Hs] = source_model.blocks[k].weight.data
        target.blocks[k].bias.data.zero_()
        target.blocks[k].bias.data[:Hs] = source_model.blocks[k].bias.data

    # head.weight: (V, H_s) -> (V, H_t): copy first H_s columns, zero-pad
    target.head.weight.data.zero_()
    target.head.weight.data[:, :Hs] = source_model.head.weight.data
    # head.bias unchanged
    target.head.bias.data.copy_(source_model.head.bias.data)

    return target


def add_width_correction(
    model: VariableTinyMLP,
    source_hidden: int,
    rank: int = 8,
    init_scale: float = 0.01,
):
    """Add low-rank correction to the NEW hidden units (indices source_hidden..H-1).

    Perturbs the weights of the new units to break symmetry and allow
    specialization during fine-tuning. Directly modifies weights in-place.
    """
    Hs = source_hidden
    Ht = model.hidden_dim

    for k in range(model.depth):
        W = model.blocks[k].weight.data
        # Perturb only the rows/cols corresponding to new units
        if k == 0:
            Wmat = W.view(Ht, 2 * Ht)
            # A block: perturb new rows (Hs..Ht-1, all cols)
            A_new = Wmat[Hs:, :Ht]
            dA = torch.randn(A_new.shape[0], rank, device=A_new.device) * (1.0 / rank**0.5)
            dB = torch.randn(rank, A_new.shape[1], device=A_new.device) * init_scale
            A_new.add_(dA @ dB)
            # B block: perturb new rows/cols
            B_new = Wmat[Hs:, Ht:2*Ht]
            dA2 = torch.randn(B_new.shape[0], rank, device=B_new.device) * (1.0 / rank**0.5)
            dB2 = torch.randn(rank, B_new.shape[1], device=B_new.device) * init_scale
            B_new.add_(dA2 @ dB2)
        else:
            Wmat = W.view(Ht, Ht)
            # Perturb new rows and new cols
            A_new = Wmat[Hs:, :]  # new output units
            dA = torch.randn(A_new.shape[0], rank, device=A_new.device) * (1.0 / rank**0.5)
            dB = torch.randn(rank, A_new.shape[1], device=A_new.device) * init_scale
            A_new.add_(dA @ dB)
            # Also perturb new input cols (for existing output units)
            B_new = Wmat[:Hs, Hs:]
            dA2 = torch.randn(B_new.shape[0], rank, device=B_new.device) * (1.0 / rank**0.5)
            dB2 = torch.randn(rank, B_new.shape[1], device=B_new.device) * init_scale
            B_new.add_(dA2 @ dB2)

        # Perturb new biases
        model.blocks[k].bias.data[Hs:].add_(
            torch.randn_like(model.blocks[k].bias.data[Hs:]) * init_scale * 0.1
        )

    # Perturb head's new columns
    head_new = model.head.weight.data[:, Hs:]
    dA = torch.randn(head_new.shape[0], rank, device=head_new.device) * (1.0 / rank**0.5)
    dB = torch.randn(rank, head_new.shape[1], device=head_new.device) * init_scale
    head_new.add_(dA @ dB)

    return model


def verify_growth_preserves_function(
    source_model: VariableTinyMLP,
    target_model: VariableTinyMLP,
    n_probe: int = 64,
    rollout: int = 4,
    atol: float = 1e-4,
    seed: int = 12345,
) -> float:
    """Verify that grown model computes the same function as source."""
    source_model = source_model.cpu().eval()
    target_model = target_model.cpu().eval()

    g = torch.Generator().manual_seed(seed)
    V = source_model.vocab_size
    x_seq = torch.randint(0, V, (rollout, n_probe), generator=g)

    max_diff = 0.0
    with torch.no_grad():
        h_s = h_t = None
        for step in range(rollout):
            x = x_seq[step]
            logits_s, h_s = source_model(x, h_s)
            logits_t, h_t = target_model(x, h_t)
            d = (logits_s - logits_t).abs().max().item()
            max_diff = max(max_diff, d)

    assert max_diff < atol, (
        f"WIDTH GROWTH NOT FUNCTION-PRESERVING: max abs logit diff {max_diff:.3e} >= {atol:.1e}."
    )
    return max_diff
