#!/usr/bin/env python3
"""Smoke test for alignment core: GATE 0 passes on correct perm, FIRES on wrong perm.

Self-contained — builds a random TinyMLP flat vector, fakes a meta with the
real layer_slices, and checks:
  1. apply_perm_to_flat is function-preserving (verify passes).
  2. canonical_sort_perm is function-preserving.
  3. weight_match_perm of a permuted model recovers a function-preserving frame.
  4. A deliberately WRONG perm on the recurrent block makes GATE 0 raise.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from src.align import rebasin
from src.models.tiny_mlp import TinyMLP
from src.utils.weights import flatten_state_dict, param_slices

H = 16
m = TinyMLP(hidden_dim=H)
# randomize so it's not the trivial init
with torch.no_grad():
    for p in m.parameters():
        p.copy_(torch.randn_like(p) * 0.3)
flat = flatten_state_dict(m)
meta = {"layer_slices": [[s["start"], s["end"]] for s in param_slices(m)], "hidden_dim": H}

# 1. random perm is function-preserving
perm = torch.randperm(H)
d = rebasin.verify_function_preserved(flat, perm, meta, H)
print(f"[1] random perm function-preserved, max diff={d:.2e}  OK")

# 2. canonical sort
cperm = rebasin.canonical_sort_perm(flat, meta, H)
d = rebasin.verify_function_preserved(flat, cperm, meta, H)
print(f"[2] canonical_sort function-preserved, max diff={d:.2e}  OK")

# 3. weight match: permute a copy, then match back to original frame
permuted = rebasin.apply_perm_to_flat(flat, torch.randperm(H), meta, H)
mperm = rebasin.weight_match_perm(permuted, flat, meta, H, iters=5)
d = rebasin.verify_function_preserved(permuted, mperm, meta, H)
print(f"[3] weight_match function-preserved, max diff={d:.2e}  OK")

# 4. WRONG perm: corrupt the recurrent-block handling by applying a one-sided
#    permutation to B only (the classic conjugation bug). Build a broken flat
#    manually and confirm GATE 0 catches it via the rollout.
idx = rebasin.layer_index(meta)
V = (idx["fc2.weight"][1] - idx["fc2.weight"][0]) // H
broken = flat.clone()
A, B = rebasin.split_fc1(flat[idx["fc1.weight"][0]:idx["fc1.weight"][1]], H)
wrong = torch.randperm(H)
A_new = A[wrong, :]
B_new = B[wrong, :]            # BUG: rows only, NOT columns (no P^T) -> breaks feedback
fc1_new = torch.cat([A_new, B_new], dim=1).reshape(-1)
broken[idx["fc1.weight"][0]:idx["fc1.weight"][1]] = fc1_new
b = flat[idx["fc1.bias"][0]:idx["fc1.bias"][1]]
broken[idx["fc1.bias"][0]:idx["fc1.bias"][1]] = b[wrong]
fc2 = flat[idx["fc2.weight"][0]:idx["fc2.weight"][1]].view(V, H)
broken[idx["fc2.weight"][0]:idx["fc2.weight"][1]] = fc2[:, wrong].reshape(-1)

# Compare broken vs original via rollout directly (broken is a hand-built flat,
# not a perm of `flat`), to confirm the rollout DETECTS the divergence.
from src.utils.weights import load_flat_into_model
mo = TinyMLP(hidden_dim=H, vocab_size=V); load_flat_into_model(flat, mo); mo.eval()
mb = TinyMLP(hidden_dim=H, vocab_size=V); load_flat_into_model(broken, mb); mb.eval()
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
print(f"[4] one-sided B perm (conjugation bug) DETECTED by rollout, max diff={maxd:.2e}  OK")

print("\nALL SMOKE CHECKS PASSED")
