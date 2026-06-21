"""Git Re-Basin weight matching for TinyMLP hidden-unit permutation symmetry.

TinyMLP forward:
    emb = Embed[x]                       # (B, H)
    h   = relu(fc1( [emb | h_prev] ))    # fc1.weight: (H, 2H) = [A | B]
    logits = fc2(h)                      # fc2.weight: (V, H)

The hidden axis (width H, the output of fc1 / input of fc2) admits a discrete
permutation symmetry P in S_H under which the function is EXACTLY preserved:

    fc1.weight[:, 0:H]  (= A, embed cols)      -> P . A          (rows by P)
    fc1.weight[:, H:2H] (= B, recurrent cols)  -> P . B . P^T    (CONJUGATION)
    fc1.bias                                   -> P . b
    fc2.weight          (input cols)           -> fc2.weight . P^T
    embed.weight, fc2.bias                     -> unchanged

The recurrent block B is conjugated (P on BOTH sides) because the hidden state
h is fed back into fc1's input. A one-sided permutation of B silently changes
the function — and only through the feedback path. verify_function_preserved
rolls the recurrence forward to catch exactly that bug.

We canonicalize over P only. The embedding GL(H) gauge and embedding-feature
permutation are deliberately NOT removed (out of scope for the first decisive
result); this makes the aligned arm a lower bound on achievable variance
collapse, which only strengthens a positive result.

Default algorithm: weight matching (Git Re-Basin "PERM" objective), solved via
coordinate descent over a linear assignment problem (scipy.linear_sum_assignment).
No forward passes, deterministic given a reference.
"""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment

from src.models.tiny_mlp import TinyMLP

__all__ = [
    "layer_index",
    "split_fc1",
    "apply_perm_to_flat",
    "canonical_sort_perm",
    "weight_match_perm",
    "align_collection",
    "verify_function_preserved",
]


# ---------------------------------------------------------------------------
# Layout helpers — key off parameter NAMES in meta["layer_slices"], not order.
# ---------------------------------------------------------------------------
def layer_index(meta: dict) -> dict[str, tuple[int, int]]:
    """Map parameter name -> (start, end) flat slice from meta['layer_slices'].

    meta['layer_slices'] is a list of [start, end] pairs in named_parameters
    order: embed.weight, fc1.weight, fc1.bias, fc2.weight, fc2.bias.
    """
    slices = meta.get("layer_slices")
    if not slices:
        raise ValueError("meta missing 'layer_slices'; re-run collect_weights.py")
    names = ["embed.weight", "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"]
    if len(slices) != len(names):
        raise ValueError(
            f"Expected {len(names)} layer slices for TinyMLP, got {len(slices)}. "
            "rebasin.py only supports the fixed TinyMLP layout."
        )
    return {n: (int(s[0]), int(s[1])) for n, s in zip(names, slices)}


def _get(flat: torch.Tensor, idx: dict, name: str) -> torch.Tensor:
    s, e = idx[name]
    return flat[s:e]


def _vocab_from_idx(idx: dict, H: int) -> int:
    s, e = idx["fc2.weight"]
    return (e - s) // H


def split_fc1(fc1_w_flat: torch.Tensor, H: int) -> tuple[torch.Tensor, torch.Tensor]:
    """fc1 flat (H*2H,) -> (A:(H,H) embed cols, B:(H,H) recurrent cols)."""
    W = fc1_w_flat.view(H, 2 * H)
    return W[:, 0:H].contiguous(), W[:, H : 2 * H].contiguous()


# ---------------------------------------------------------------------------
# Apply a hidden-unit permutation, function-preserving.
# ---------------------------------------------------------------------------
def apply_perm_to_flat(flat: torch.Tensor, perm: torch.Tensor, meta: dict, H: int) -> torch.Tensor:
    """Apply hidden-unit permutation P (LongTensor len H) to one flat vector.

    perm[i] = j means: new hidden unit i takes the role of old hidden unit j.
    Equivalent to left-multiplying the relevant matrices by the permutation
    matrix that reorders rows according to `perm`.
    """
    if perm.dtype != torch.long:
        perm = perm.long()
    idx = layer_index(meta)
    V = _vocab_from_idx(idx, H)
    out = flat.clone()

    # fc1.weight: rows -> perm ; recurrent cols (B) -> perm  (A cols untouched)
    A, B = split_fc1(_get(flat, idx, "fc1.weight"), H)
    A_new = A[perm, :]                 # rows by P
    B_new = B[perm, :][:, perm]        # rows AND cols by P  (conjugation P B P^T)
    fc1_new = torch.cat([A_new, B_new], dim=1).reshape(-1)
    s, e = idx["fc1.weight"]
    out[s:e] = fc1_new

    # fc1.bias: rows -> perm
    b = _get(flat, idx, "fc1.bias")
    s, e = idx["fc1.bias"]
    out[s:e] = b[perm]

    # fc2.weight: (V, H) input cols -> perm
    fc2 = _get(flat, idx, "fc2.weight").view(V, H)
    s, e = idx["fc2.weight"]
    out[s:e] = fc2[:, perm].reshape(-1)

    # embed.weight, fc2.bias: unchanged
    return out


# ---------------------------------------------------------------------------
# Canonical-sort frame (parameter-free) — used to canonicalize the reference
# and as a cheap baseline alignment method.
# ---------------------------------------------------------------------------
def canonical_sort_perm(flat: torch.Tensor, meta: dict, H: int) -> torch.Tensor:
    """Permutation sorting hidden units by fc1 row-norm of [A|B] desc,
    tie-broken by fc1.bias desc. Returns LongTensor perm (len H)."""
    idx = layer_index(meta)
    W = _get(flat, idx, "fc1.weight").view(H, 2 * H)
    row_norm = W.norm(dim=1)
    bias = _get(flat, idx, "fc1.bias")
    # sort key: primary row_norm, secondary bias; descending
    key = row_norm + 1e-6 * bias
    perm = torch.argsort(key, descending=True)
    return perm.long()


# ---------------------------------------------------------------------------
# Git Re-Basin weight matching to a reference.
# ---------------------------------------------------------------------------
def weight_match_perm(
    flat_m: torch.Tensor,
    flat_ref: torch.Tensor,
    meta: dict,
    H: int,
    iters: int = 3,
) -> torch.Tensor:
    """Permutation P aligning model `flat_m`'s hidden units to `flat_ref`.

    Maximizes weight-space agreement across the function-coupled blocks:
      A   (fc1 embed cols, rows indexed by hidden unit)
      bias(fc1)
      fc2 (input cols, i.e. fc2^T rows indexed by hidden unit)
      B   (fc1 recurrent block, conjugated -> handled by coordinate descent)

    The B term couples P on both sides, so the joint objective is bilinear in P.
    We solve by coordinate descent: hold the current P for B's column side fixed,
    build the linear assignment cost, solve, repeat `iters` times.
    """
    idx = layer_index(meta)
    V = _vocab_from_idx(idx, H)

    Am, Bm = split_fc1(_get(flat_m, idx, "fc1.weight"), H)
    Ar, Br = split_fc1(_get(flat_ref, idx, "fc1.weight"), H)
    bm = _get(flat_m, idx, "fc1.bias")
    br = _get(flat_ref, idx, "fc1.bias")
    fc2m = _get(flat_m, idx, "fc2.weight").view(V, H)  # cols indexed by hidden
    fc2r = _get(flat_ref, idx, "fc2.weight").view(V, H)

    # Static (P-independent) part of the similarity between ref unit i and
    # model unit j: <A_ref[i], A_m[j]> + bias_ref[i]*bias_m[j] + <fc2_ref[:,i], fc2_m[:,j]>
    # All are (H, H) matrices: rows = ref unit i, cols = model unit j.
    S_static = Ar @ Am.t()                        # (H,H)
    S_static += torch.outer(br, bm)               # bias term
    S_static += fc2r.t() @ fc2m                   # fc2 cols: (H,V)@(V,H) -> (H,H)

    perm = torch.arange(H, dtype=torch.long)  # current best P (ref<-model)
    for _ in range(max(1, iters)):
        # Recurrent term contribution given current perm applied to B_m.
        # B contributes <Br[i,:] , (P B_m P^T)[i,:]> aggregated; coordinate
        # descent: fix the column-side permutation (current perm), then the
        # row-side cost for assigning ref row i <- model row j is
        # <Br[i, perm], Bm[j, perm]> = (Br[:, perm] @ Bm[:, perm]^T)[i, j].
        Bm_cols = Bm[:, perm]      # apply current P to model B columns
        Br_cols = Br[:, perm]
        S_rec = Br_cols @ Bm_cols.t()             # (H,H)
        S = S_static + S_rec
        # Maximize agreement -> minimize negative cost.
        cost = (-S).cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost)
        # row_ind is sorted 0..H-1; perm[i] = model unit assigned to ref unit i
        new_perm = torch.zeros(H, dtype=torch.long)
        new_perm[torch.as_tensor(row_ind)] = torch.as_tensor(col_ind, dtype=torch.long)
        if torch.equal(new_perm, perm):
            perm = new_perm
            break
        perm = new_perm
    return perm


# ---------------------------------------------------------------------------
# Function-preservation HARD GATE — rolls the recurrence forward.
# ---------------------------------------------------------------------------
def verify_function_preserved(
    flat: torch.Tensor,
    perm: torch.Tensor,
    meta: dict,
    H: int,
    n_probe: int = 64,
    rollout: int = 4,
    atol: float = 1e-5,
    seed: int = 12345,
) -> float:
    """Load `flat` and `apply_perm_to_flat(flat, perm)` into two TinyMLPs and
    assert identical outputs across a RECURRENT rollout (h fed forward >=3
    steps). The conjugation bug in B only manifests through the feedback path,
    so a single-step check is insufficient.

    Returns max abs logit difference. RAISES AssertionError if >= atol.
    """
    from src.utils.weights import load_flat_into_model

    if rollout < 3:
        raise ValueError("rollout must be >= 3 to exercise the recurrent feedback path")

    idx = layer_index(meta)
    V = _vocab_from_idx(idx, H)

    m_orig = TinyMLP(hidden_dim=H, vocab_size=V)
    m_perm = TinyMLP(hidden_dim=H, vocab_size=V)
    load_flat_into_model(flat, m_orig)
    load_flat_into_model(apply_perm_to_flat(flat, perm, meta, H), m_perm)
    m_orig.eval()
    m_perm.eval()

    g = torch.Generator().manual_seed(seed)
    x_seq = torch.randint(0, V, (rollout, n_probe), generator=g)

    max_diff = 0.0
    with torch.no_grad():
        h_o = h_p = None
        for t in range(rollout):
            xt = x_seq[t]
            lo, h_o = m_orig(xt, h_o)
            lp, h_p = m_perm(xt, h_p)
            d = (lo - lp).abs().max().item()
            max_diff = max(max_diff, d)
            # also track hidden-state divergence implicitly via next-step logits
    assert max_diff < atol, (
        f"FUNCTION PRESERVATION FAILED: max abs logit diff {max_diff:.3e} >= {atol:.1e} "
        f"over {rollout}-step rollout. The permutation is NOT function-preserving "
        f"(likely the recurrent B-block conjugation is wrong)."
    )
    return max_diff


# ---------------------------------------------------------------------------
# Align a whole collection to a canonical reference frame.
# ---------------------------------------------------------------------------
def align_collection(
    raw: torch.Tensor,
    meta: dict,
    H: int,
    method: str = "weight_match",
    iters: int = 3,
    verify: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Align every row of `raw` (N, D) to a canonical reference frame.

    Reference = canonical_sort of raw[0] (breaks the reference's own arbitrary
    ordering so the frame is deterministic). Each model is then matched to it.

    Returns (aligned (N, D), info dict with max function-preservation diff).
    """
    if method not in ("weight_match", "canonical_sort"):
        raise ValueError(f"Unknown align method: {method}")

    N = raw.shape[0]
    # Canonicalize the reference itself.
    ref_perm = canonical_sort_perm(raw[0], meta, H)
    ref = apply_perm_to_flat(raw[0], ref_perm, meta, H)

    aligned = torch.empty_like(raw)
    max_fn_diff = 0.0
    for i in range(N):
        flat = raw[i]
        if method == "canonical_sort":
            perm = canonical_sort_perm(flat, meta, H)
        else:
            perm = weight_match_perm(flat, ref, meta, H, iters=iters)
        if verify:
            d = verify_function_preserved(flat, perm, meta, H)
            max_fn_diff = max(max_fn_diff, d)
        aligned[i] = apply_perm_to_flat(flat, perm, meta, H)

    info = {
        "method": method,
        "iters": iters,
        "num_models": N,
        "function_preservation_max_diff": max_fn_diff,
    }
    return aligned, info
