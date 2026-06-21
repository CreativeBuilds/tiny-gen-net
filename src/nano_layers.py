"""Logical layer groups for progressive weight generation (embed → blocks → head)."""

import re

from src.arch_gen.nano_spec import NanoSpec
from src.models.nano_transformer import build_from_nano_spec
from src.utils.weights import param_slices

_BLOCK_RE = re.compile(r"blocks\.(\d+)\.")


def logical_layer_bounds(spec: NanoSpec, vocab: int) -> list[tuple[int, int]]:
    """Return (start, end) flat slices: [tok+pos, block0..L-1, ln_f+head]."""
    m = build_from_nano_spec(spec, vocab)
    slices = param_slices(m)
    groups: dict[int, list[tuple[int, int]]] = {}
    embed, tail = [], []
    for sl in slices:
        name, bounds = sl["name"], (sl["start"], sl["end"])
        if name.startswith(("tok.", "pos.")): embed.append(bounds); continue
        if name.startswith(("ln_f.", "lm_head.")): tail.append(bounds); continue
        mch = _BLOCK_RE.search(name)
        if not mch: continue
        groups.setdefault(int(mch.group(1)), []).append(bounds)
    out: list[tuple[int, int]] = []
    if embed: out.append((embed[0][0], embed[-1][1]))
    for i in range(spec.n_layer):
        if i not in groups: continue
        out.append((groups[i][0][0], groups[i][-1][1]))
    if tail: out.append((tail[0][0], tail[-1][1]))
    return out


def layer_dims(spec: NanoSpec, vocab: int) -> list[int]:
    return [e - s for s, e in logical_layer_bounds(spec, vocab)]
