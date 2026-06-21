#!/usr/bin/env python3
"""Smoke test for alignment core: GATE 0 passes on correct perm, FIRES on wrong perm.

Tests both depth=1 (TinyMLP) and depth=3 (VariableTinyMLP):
  1. apply_perms_to_flat is function-preserving (verify passes) — depth=1
  2. canonical_sort_perms is function-preserving — depth=1
  3. weight_match_perms of a permuted model recovers a function-preserving frame — depth=1
  4. A deliberately WRONG perm on the recurrent block makes GATE 0 raise — depth=1
  5-8. Same checks for depth=3 (VariableTinyMLP) — exercises cross-axis B conjugation
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.align import rebasin
from src.models.tiny_mlp import TinyMLP
# Import arch_gen first to break the circular import (arch_gen.__init__ loads
# conditioning -> variable_mlp -> arch_gen.spec; importing the package first
# lets the full initialization chain complete).
from src.arch_gen.spec import ArchSpec  # noqa: F401 — forces package init
from src.models.variable_mlp import VariableTinyMLP
from data.synthetic_text import VOCAB_SIZE
from src.utils.weights import flatten_state_dict, param_slices


def make_meta(model):
    return {"layer_slices": [[s["start"], s["end"]] for s in param_slices(model)]}


def randomize(model, scale=0.3):
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn_like(p) * scale)


# ===========================================================================
# DEPTH=1 (TinyMLP) — backward-compat path
# ===========================================================================
print("=== DEPTH=1 (TinyMLP) ===")
H = 16
m = TinyMLP(hidden_dim=H)
randomize(m)
flat = flatten_state_dict(m)
meta = make_meta(m)
layout = rebasin.build_layout(meta, H, 1)
factory = lambda: TinyMLP(hidden_dim=H)

# 1. random perm is function-preserving
perm = [torch.randperm(H)]
d = rebasin.verify_function_preserved(flat, perm, layout, factory)
print(f"[1] random perm function-preserved, max diff={d:.2e}  OK")

# 2. canonical sort
cperms = rebasin.canonical_sort_perms(flat, layout)
d = rebasin.verify_function_preserved(flat, cperms, layout, factory)
print(f"[2] canonical_sort function-preserved, max diff={d:.2e}  OK")

# 3. weight match: permute a copy, then match back
permuted = rebasin.apply_perms_to_flat(flat, [torch.randperm(H)], layout)
mperms = rebasin.weight_match_perms(permuted, flat, layout, iters=5)
d = rebasin.verify_function_preserved(permuted, mperms, layout, factory)
print(f"[3] weight_match function-preserved, max diff={d:.2e}  OK")

# 4. WRONG perm: one-sided B permutation (conjugation bug)
broken = flat.clone()
idx = rebasin.layer_index(meta)
V = VOCAB_SIZE
A, B = rebasin.split_fc1(flat[idx["fc1.weight"][0]:idx["fc1.weight"][1]], H)
wrong = torch.randperm(H)
A_new = A[wrong, :]
B_new = B[wrong, :]  # BUG: rows only, NOT cols -> breaks feedback
fc1_new = torch.cat([A_new, B_new], dim=1).reshape(-1)
broken[idx["fc1.weight"][0]:idx["fc1.weight"][1]] = fc1_new
b = flat[idx["fc1.bias"][0]:idx["fc1.bias"][1]]
broken[idx["fc1.bias"][0]:idx["fc1.bias"][1]] = b[wrong]
fc2 = flat[idx["fc2.weight"][0]:idx["fc2.weight"][1]].view(V, H)
broken[idx["fc2.weight"][0]:idx["fc2.weight"][1]] = fc2[:, wrong].reshape(-1)

from src.utils.weights import load_flat_into_model
mo = TinyMLP(hidden_dim=H); load_flat_into_model(flat, mo); mo.eval()
mb = TinyMLP(hidden_dim=H); load_flat_into_model(broken, mb); mb.eval()
g = torch.Generator().manual_seed(7)
xseq = torch.randint(0, V, (4, 32), generator=g)
maxd = 0.0
with torch.no_grad():
    ho = hb = None
    for t in range(4):
        lo, ho = mo(xseq[t], ho)
        lb, hb = mb(xseq[t], hb)
        maxd = max(maxd, (lo - lb).abs().max().item())
assert maxd > 1e-3, f"[4] FAIL: broken perm should diverge but max diff={maxd:.2e}"
print(f"[4] one-sided B perm (conjugation bug) DETECTED, max diff={maxd:.2e}  OK")


# ===========================================================================
# DEPTH=3 (VariableTinyMLP) — multi-axis path, cross-axis B conjugation
# ===========================================================================
print("\n=== DEPTH=3 (VariableTinyMLP) ===")
H = 24
spec = ArchSpec(H, 3, skip=False)
m = VariableTinyMLP(spec, vocab_size=VOCAB_SIZE)
randomize(m)
flat = flatten_state_dict(m)
meta = make_meta(m)
meta["depth"] = 3
layout = rebasin.build_layout(meta, H, 3)
factory = lambda: VariableTinyMLP(spec, vocab_size=VOCAB_SIZE)

# 5. random perms (one per axis) are function-preserving
perms = [torch.randperm(H) for _ in range(3)]
d = rebasin.verify_function_preserved(flat, perms, layout, factory)
print(f"[5] random perms function-preserved, max diff={d:.2e}  OK")

# 6. canonical sort (3 axes)
cperms = rebasin.canonical_sort_perms(flat, layout)
d = rebasin.verify_function_preserved(flat, cperms, layout, factory)
print(f"[6] canonical_sort (3 axes) function-preserved, max diff={d:.2e}  OK")

# 7. weight match: permute a copy, then match back
rand_perms = [torch.randperm(H) for _ in range(3)]
permuted = rebasin.apply_perms_to_flat(flat, rand_perms, layout)
mperms = rebasin.weight_match_perms(permuted, flat, layout, iters=5)
d = rebasin.verify_function_preserved(permuted, mperms, layout, factory)
print(f"[7] weight_match (3 axes) function-preserved, max diff={d:.2e}  OK")

# 8. WRONG perm: corrupt B block with a one-sided permutation on cols only
#    (should break the P_0 x P_2 cross-conjugation)
broken = flat.clone()
ws0 = layout.block_weight_slices[0]
W0 = flat[ws0[0]:ws0[1]].view(H, 2 * H)
A0, B0 = W0[:, :H], W0[:, H:]
wrong_col = torch.randperm(H)
B0_broken = B0[:, wrong_col]  # BUG: cols only, NOT rows -> breaks P_0 x P_2 conjugation
W0_new = torch.cat([A0, B0_broken], dim=1).reshape(-1)
broken[ws0[0]:ws0[1]] = W0_new

mo = VariableTinyMLP(spec, vocab_size=VOCAB_SIZE); load_flat_into_model(flat, mo); mo.eval()
mb = VariableTinyMLP(spec, vocab_size=VOCAB_SIZE); load_flat_into_model(broken, mb); mb.eval()
g = torch.Generator().manual_seed(7)
xseq = torch.randint(0, VOCAB_SIZE, (4, 32), generator=g)
maxd = 0.0
with torch.no_grad():
    ho = hb = None
    for t in range(4):
        lo, ho = mo(xseq[t], ho)
        lb, hb = mb(xseq[t], hb)
        maxd = max(maxd, (lo - lb).abs().max().item())
assert maxd > 1e-3, f"[8] FAIL: broken B-cols perm should diverge but max diff={maxd:.2e}"
print(f"[8] one-sided B-cols perm (cross-axis conjugation bug) DETECTED, max diff={maxd:.2e}  OK")

print("\nALL SMOKE CHECKS PASSED")
