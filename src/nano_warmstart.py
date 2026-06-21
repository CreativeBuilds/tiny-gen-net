"""Warm-start larger nano specs from smaller trained checkpoints."""

import re

import torch

from src.arch_gen.nano_spec import NanoSpec, estimate_nano_params
from src.models.nano_transformer import build_from_nano_spec
from src.utils.weights import flatten_state_dict, load_flat_into_model

_BLOCK_RE = re.compile(r"^blocks\.(\d+)\.")


def _transfer_into(dst: torch.Tensor, src: torch.Tensor) -> int:
    if dst.shape == src.shape: dst.copy_(src); return dst.numel()
    if dst.dim() == 1:
        n = min(dst.shape[0], src.shape[0]); dst[:n].copy_(src[:n]); return n
    if dst.dim() == 2:
        r, c = min(dst.shape[0], src.shape[0]), min(dst.shape[1], src.shape[1])
        dst[:r, :c].copy_(src[:r, :c]); return r * c
    return 0


def pick_donor_spec(target: NanoSpec, donors: list[NanoSpec], vocab: int = 65) -> NanoSpec | None:
    tp = estimate_nano_params(target.n_embd, target.n_layer, target.n_head, vocab, target.block_size)
    cands = []
    for d in donors:
        dp = estimate_nano_params(d.n_embd, d.n_layer, d.n_head, vocab, d.block_size)
        if dp > tp: continue
        cands.append((d, abs(d.n_embd - target.n_embd), abs(d.n_layer - target.n_layer), -dp))
    if not cands: return None
    cands.sort(key=lambda x: (x[1], x[2], x[3]))
    return cands[0][0]


def warm_start_flat(target: NanoSpec, donor: NanoSpec, donor_flat, vocab: int = 65):
    tm, dm = build_from_nano_spec(target, vocab), build_from_nano_spec(donor, vocab)
    load_flat_into_model(donor_flat, dm)
    t_sd, d_sd = tm.state_dict(), dm.state_dict()
    matched, transferred = 0, 0
    for k in t_sd:
        m = _BLOCK_RE.match(k)
        if m and int(m.group(1)) >= donor.n_layer: continue
        if k not in d_sd: continue
        ts = t_sd[k].clone()
        n = _transfer_into(ts, d_sd[k])
        if n <= 0: continue
        t_sd[k] = ts; matched += 1; transferred += n
    tm.load_state_dict(t_sd)
    return flatten_state_dict(tm), tm, matched, transferred
