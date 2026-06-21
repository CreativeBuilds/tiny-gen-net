"""Git Re-Basin weight matching for TinyMLP / VariableTinyMLP (any depth).

TinyMLP / VariableTinyMLP forward:
    emb = Embed[x]                            # (B, H)
    h   = relu(blocks[0]( [emb | h_prev] ))   # blocks[0].weight: (H, 2H) = [A | B]
    [skip: h = h + h_prev]
    for k in 1..depth-1: h = relu(blocks[k](h))
    logits = head(h)                          # head.weight: (V, H)
    returns (logits, h)   # h fed back as h_prev

The hidden axis after each block k admits a discrete permutation symmetry P_k
in S_H. For depth=d there are d independent axes. The function is EXACTLY
preserved under:

    blocks[0].weight[:, 0:H]  (= A, embed cols)      -> P_0 . A           (rows by P_0)
    blocks[0].weight[:, H:2H] (= B, recurrent cols)  -> P_0 . B . P_{d-1}^T  (CROSS-CONJUGATION)
    blocks[0].bias                                  -> P_0 . b
    blocks[k].weight  (k > 0, shape HxH)             -> P_k . W . P_{k-1}^T
    blocks[k].bias    (k > 0)                        -> P_k . b
    head.weight       (cols)                         -> head.weight . P_{d-1}^T
    embed.weight, head.bias                          -> unchanged

The recurrent block B is conjugated by TWO DIFFERENT permutations (P_0 on rows,
P_{d-1} on cols) because the hidden state h fed back is the output of the LAST
block (axis d-1). For depth=1 this collapses to self-conjugation P_0 B P_0^T.

verify_function_preserved rolls the recurrence forward to catch conjugation bugs
that only manifest through the feedback path.

We canonicalize over the hidden-axis permutations only. The embedding GL(H)
gauge and embedding-feature permutation are deliberately NOT removed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from scipy.optimize import linear_sum_assignment

__all__ = [
    "ModelLayout",
    "build_layout",
    "apply_perms_to_flat",
    "canonical_sort_perms",
    "weight_match_perms",
    "verify_function_preserved",
    "align_collection",
    # backward-compat wrappers (depth=1)
    "layer_index",
    "split_fc1",
    "apply_perm_to_flat",
    "canonical_sort_perm",
    "weight_match_perm",
]


# ---------------------------------------------------------------------------
# Layout detection
# ---------------------------------------------------------------------------
@dataclass
class ModelLayout:
    """Structural layout of a TinyMLP / VariableTinyMLP flat weight vector."""

    H: int
    depth: int
    vocab: int
    embed_slice: tuple[int, int]
    block_weight_slices: list[tuple[int, int]]  # blocks[k].weight
    block_bias_slices: list[tuple[int, int]]  # blocks[k].bias
    head_weight_slice: tuple[int, int]
    head_bias_slice: tuple[int, int]

    @property
    def D(self) -> int:
        return self.head_bias_slice[1]


def build_layout(meta: dict, H: int, depth: int) -> ModelLayout:
    """Build layout from meta['layer_slices'] + H + depth.

    layer_slices is a list of [start, end] pairs in named_parameters order:
      embed.weight, blocks.0.weight, blocks.0.bias,
      [blocks.1.weight, blocks.1.bias, ...], head.weight, head.bias

    For TinyMLP (depth=1) the names differ (fc1/fc2) but the ORDER is the same:
      embed.weight, fc1.weight, fc1.bias, fc2.weight, fc2.bias
    """
    slices_raw = meta.get("layer_slices")
    if not slices_raw:
        raise ValueError("meta missing 'layer_slices'; re-run collect_weights.py")
    slices = [(int(s[0]), int(s[1])) for s in slices_raw]

    vocab = (slices[0][1] - slices[0][0]) // H
    expected = 2 * depth + 3  # embed + d*(weight+bias) + head_weight + head_bias
    if len(slices) != expected:
        raise ValueError(
            f"Expected {expected} layer slices for depth={depth}, got {len(slices)}. "
            f"rebasin.py only supports TinyMLP / VariableTinyMLP layouts."
        )

    block_weight_slices = [slices[1 + 2 * k] for k in range(depth)]
    block_bias_slices = [slices[1 + 2 * k + 1] for k in range(depth)]
    head_weight_slice = slices[1 + 2 * depth]
    head_bias_slice = slices[1 + 2 * depth + 1]

    return ModelLayout(
        H=H,
        depth=depth,
        vocab=vocab,
        embed_slice=slices[0],
        block_weight_slices=block_weight_slices,
        block_bias_slices=block_bias_slices,
        head_weight_slice=head_weight_slice,
        head_bias_slice=head_bias_slice,
    )


def _get(flat: torch.Tensor, s: tuple[int, int]) -> torch.Tensor:
    return flat[s[0] : s[1]]


# ---------------------------------------------------------------------------
# Apply permutations (generalized, any depth)
# ---------------------------------------------------------------------------
def apply_perms_to_flat(
    flat: torch.Tensor, perms: list[torch.Tensor], layout: ModelLayout
) -> torch.Tensor:
    """Apply d hidden-unit permutations to one flat vector. Function-preserving.

    perms[k] is a LongTensor of length H: new unit i takes the role of old unit
    perms[k][i] on axis k.
    """
    H, d = layout.H, layout.depth
    perms = [p.long() for p in perms]
    out = flat.clone()

    for k in range(d):
        Pk = perms[k]

        # blocks[k].weight
        ws = layout.block_weight_slices[k]
        W = _get(flat, ws)

        if k == 0:
            # Recurrent block: (H, 2H) = [A | B]
            Wmat = W.view(H, 2 * H)
            A, B = Wmat[:, :H], Wmat[:, H:]
            A_new = A[Pk, :]  # rows by P_0
            B_new = B[Pk, :][:, perms[d - 1]]  # rows by P_0, cols by P_{d-1}
            out[ws[0] : ws[1]] = torch.cat([A_new, B_new], dim=1).reshape(-1)
        else:
            # Standard hidden layer: (H, H), rows by P_k, cols by P_{k-1}
            Wmat = W.view(H, H)
            W_new = Wmat[Pk, :][:, perms[k - 1]]
            out[ws[0] : ws[1]] = W_new.reshape(-1)

        # blocks[k].bias -> P_k
        bs = layout.block_bias_slices[k]
        out[bs[0] : bs[1]] = _get(flat, bs)[Pk]

    # head.weight: cols -> P_{d-1}
    hs = layout.head_weight_slice
    head = _get(flat, hs).view(layout.vocab, H)
    out[hs[0] : hs[1]] = head[:, perms[d - 1]].reshape(-1)

    return out


# ---------------------------------------------------------------------------
# Canonical-sort frame (parameter-free, per-axis)
# ---------------------------------------------------------------------------
def canonical_sort_perms(flat: torch.Tensor, layout: ModelLayout) -> list[torch.Tensor]:
    """Returns d permutations, each sorting hidden units by block row-norm desc,
    tie-broken by bias desc. Parameter-free canonical frame."""
    H, d = layout.H, layout.depth
    perms = []
    for k in range(d):
        ws = layout.block_weight_slices[k]
        W = _get(flat, ws).view(H, -1)
        row_norm = W.norm(dim=1)
        bs = layout.block_bias_slices[k]
        bias = _get(flat, bs)
        key = row_norm + 1e-6 * bias
        perms.append(torch.argsort(key, descending=True).long())
    return perms


# ---------------------------------------------------------------------------
# Git Re-Basin weight matching (generalized, any depth)
# ---------------------------------------------------------------------------
def weight_match_perms(
    flat_m: torch.Tensor,
    flat_ref: torch.Tensor,
    layout: ModelLayout,
    iters: int = 3,
) -> list[torch.Tensor]:
    """Permutations aligning model `flat_m`'s hidden units to `flat_ref`.

    Coordinate descent over d axes. For each axis k, builds a cost matrix from
    all weight blocks touching that axis, solves the LAP, updates P_k.
    The recurrent B block couples axis 0 (rows) and axis d-1 (cols).
    """
    H, d, V = layout.H, layout.depth, layout.vocab

    # Pre-extract all weight blocks for model and ref
    def extract(flat):
        blocks_w = []
        blocks_b = []
        for k in range(d):
            ws = layout.block_weight_slices[k]
            W = _get(flat, ws)
            if k == 0:
                Wmat = W.view(H, 2 * H)
                blocks_w.append((Wmat[:, :H].contiguous(), Wmat[:, H:].contiguous()))  # (A, B)
            else:
                blocks_w.append(W.view(H, H).contiguous())  # plain (H,H)
            bs = layout.block_bias_slices[k]
            blocks_b.append(_get(flat, bs))
        head_w = _get(flat, layout.head_weight_slice).view(V, H)
        return blocks_w, blocks_b, head_w

    bw_m, bb_m, head_m = extract(flat_m)
    bw_r, bb_r, head_r = extract(flat_ref)

    perms = [torch.arange(H, dtype=torch.long) for _ in range(d)]

    for _ in range(max(1, iters)):
        changed = False
        for k in range(d):
            # --- Build cost matrix C[i,j]: ref unit i vs model unit j on axis k ---
            C = torch.zeros(H, H)

            # 1. blocks[k] output rows
            if k == 0:
                Ar, Am = bw_r[0][0], bw_m[0][0]  # A blocks
                Br, Bm = bw_r[0][1], bw_m[0][1]  # B blocks
                C += Ar @ Am.t()  # A rows: static
                # B rows: depends on current P_{d-1}
                P_last = perms[d - 1]
                C += Br[:, P_last] @ Bm[:, P_last].t()
            else:
                Wr, Wm = bw_r[k], bw_m[k]  # (H,H)
                C += Wr @ Wm.t()  # rows: static

            # 2. blocks[k].bias
            C += torch.outer(bb_r[k], bb_m[k])

            # 3. Next layer input cols
            if k < d - 1:
                # blocks[k+1] weight: cols indexed by P_k
                Wr_next, Wm_next = bw_r[k + 1], bw_m[k + 1]
                if k + 1 == 0:
                    # This won't happen (k >= 0, k+1 >= 1)
                    pass
                else:
                    C += Wr_next.t() @ Wm_next  # [i,j] = <Wr_next[:,i], Wm_next[:,j]>
            else:
                # head.weight: cols indexed by P_{d-1}
                C += head_r.t() @ head_m

            # 4. B block cols contribution (only for axis d-1 when d > 1)
            if k == d - 1 and d > 1:
                Br, Bm = bw_r[0][1], bw_m[0][1]  # B blocks from blocks[0]
                P_first = perms[0]
                # [i,j] = <Br[P_first, i], Bm[P_first, j]>
                #       = (Br[P_first, :].T @ Bm[P_first, :])[i, j]
                C += Br[P_first, :].t() @ Bm[P_first, :]

            # Solve LAP (maximize C = minimize -C)
            cost = (-C).cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost)
            new_perm = torch.zeros(H, dtype=torch.long)
            new_perm[torch.as_tensor(row_ind)] = torch.as_tensor(col_ind, dtype=torch.long)

            if not torch.equal(new_perm, perms[k]):
                changed = True
            perms[k] = new_perm

        if not changed:
            break

    return perms


# ---------------------------------------------------------------------------
# Function-preservation HARD GATE
# ---------------------------------------------------------------------------
def verify_function_preserved(
    flat: torch.Tensor,
    perms: list[torch.Tensor],
    layout: ModelLayout,
    model_factory,
    n_probe: int = 64,
    rollout: int = 4,
    atol: float = 1e-4,
    seed: int = 12345,
) -> float:
    """Load `flat` and `apply_perms_to_flat(flat, perms, layout)` into two models
    and assert identical outputs across a RECURRENT rollout (h fed forward >=3
    steps). The conjugation bug in B only manifests through the feedback path.

    Returns max abs logit difference. RAISES AssertionError if >= atol.
    """
    from src.utils.weights import load_flat_into_model

    if rollout < 3:
        raise ValueError("rollout must be >= 3 to exercise the recurrent feedback path")

    V = layout.vocab

    m_orig = model_factory()
    m_perm = model_factory()
    load_flat_into_model(flat, m_orig)
    load_flat_into_model(apply_perms_to_flat(flat, perms, layout), m_perm)
    m_orig.eval()
    m_perm.eval()

    g = torch.Generator().manual_seed(seed)
    x_seq = torch.randint(0, V, (rollout, n_probe), generator=g)

    max_diff = 0.0
    with torch.no_grad():
        h_o = h_p = None
        for t in range(rollout):
            lo, h_o = m_orig(x_seq[t], h_o)
            lp, h_p = m_perm(x_seq[t], h_p)
            d = (lo - lp).abs().max().item()
            max_diff = max(max_diff, d)

    assert max_diff < atol, (
        f"FUNCTION PRESERVATION FAILED: max abs logit diff {max_diff:.3e} >= {atol:.1e} "
        f"over {rollout}-step rollout. The permutation is NOT function-preserving "
        f"(likely the recurrent B-block conjugation is wrong)."
    )
    return max_diff


# ---------------------------------------------------------------------------
# Align a whole collection to a canonical reference frame
# ---------------------------------------------------------------------------
def align_collection(
    raw: torch.Tensor,
    meta: dict,
    H: int,
    depth: int,
    model_factory,
    method: str = "weight_match",
    iters: int = 3,
    verify: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Align every row of `raw` (N, D) to a canonical reference frame.

    Reference = canonical_sort of raw[0]. Each model is then matched to it.
    Returns (aligned (N, D), info dict with max function-preservation diff).
    """
    if method not in ("weight_match", "canonical_sort"):
        raise ValueError(f"Unknown align method: {method}")

    layout = build_layout(meta, H, depth)
    N = raw.shape[0]

    # Canonicalize the reference itself
    ref_perms = canonical_sort_perms(raw[0], layout)
    ref = apply_perms_to_flat(raw[0], ref_perms, layout)

    aligned = torch.empty_like(raw)
    max_fn_diff = 0.0
    for i in range(N):
        flat = raw[i]
        if method == "canonical_sort":
            perms = canonical_sort_perms(flat, layout)
        else:
            perms = weight_match_perms(flat, ref, layout, iters=iters)
        if verify:
            d = verify_function_preserved(flat, perms, layout, model_factory)
            max_fn_diff = max(max_fn_diff, d)
        aligned[i] = apply_perms_to_flat(flat, perms, layout)

    info = {
        "method": method,
        "iters": iters,
        "num_models": N,
        "depth": depth,
        "function_preservation_max_diff": max_fn_diff,
    }
    return aligned, info


# ---------------------------------------------------------------------------
# Backward-compat wrappers for depth=1 (TinyMLP)
# ---------------------------------------------------------------------------
def layer_index(meta: dict) -> dict[str, tuple[int, int]]:
    """Map parameter name -> (start, end) for the depth=1 TinyMLP layout."""
    layout = build_layout(meta, meta.get("hidden_dim", 64), 1)
    names = ["embed.weight", "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"]
    return {
        "embed.weight": layout.embed_slice,
        "fc1.weight": layout.block_weight_slices[0],
        "fc1.bias": layout.block_bias_slices[0],
        "fc2.weight": layout.head_weight_slice,
        "fc2.bias": layout.head_bias_slice,
    }


def split_fc1(fc1_w_flat: torch.Tensor, H: int) -> tuple[torch.Tensor, torch.Tensor]:
    """fc1 flat (H*2H,) -> (A:(H,H) embed cols, B:(H,H) recurrent cols)."""
    W = fc1_w_flat.view(H, 2 * H)
    return W[:, 0:H].contiguous(), W[:, H : 2 * H].contiguous()


def apply_perm_to_flat(flat: torch.Tensor, perm: torch.Tensor, meta: dict, H: int) -> torch.Tensor:
    """Single-perm wrapper for depth=1."""
    layout = build_layout(meta, H, 1)
    return apply_perms_to_flat(flat, [perm], layout)


def canonical_sort_perm(flat: torch.Tensor, meta: dict, H: int) -> torch.Tensor:
    """Single-perm canonical sort for depth=1."""
    layout = build_layout(meta, H, 1)
    return canonical_sort_perms(flat, layout)[0]


def weight_match_perm(
    flat_m: torch.Tensor,
    flat_ref: torch.Tensor,
    meta: dict,
    H: int,
    iters: int = 3,
) -> torch.Tensor:
    """Single-perm weight match for depth=1."""
    layout = build_layout(meta, H, 1)
    return weight_match_perms(flat_m, flat_ref, layout, iters=iters)[0]
