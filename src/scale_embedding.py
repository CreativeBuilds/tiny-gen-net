"""Continuous scale features for extrapolation beyond discrete arch tokens."""

import math

import torch
import torch.nn as nn

from src.arch_gen.nano_spec import NanoSpec, estimate_nano_params

SCALE_DIM = 8
LOG_PARAM_REF = math.log10(256_000_000)


def scale_features(spec: NanoSpec, vocab: int = 65) -> torch.Tensor:
    """Relative scale vector: log-params + normalized d/L/H/ctx + layer density."""
    p = float(estimate_nano_params(spec.n_embd, spec.n_layer, spec.n_head, vocab, spec.block_size))
    from src.arch_gen import nano_spec as ns
    return torch.tensor([
        math.log10(max(p, 1.0)) / LOG_PARAM_REF,
        spec.n_layer / max(ns.N_LAYER_CHOICES),
        spec.n_embd / max(ns.N_EMBD_CHOICES),
        spec.n_head / max(ns.N_HEAD_CHOICES),
        spec.block_size / max(ns.BLOCK_SIZE_CHOICES),
        p / max(ns.MAX_NANO_PARAMS, 1),
        spec.n_embd / max(spec.n_head, 1) / 64.0,
        spec.n_layer * spec.n_embd / 8192.0,
    ], dtype=torch.float32)


class ScaleEmbedding(nn.Module):
    """Maps continuous scale features + optional task vector to cond dim."""

    def __init__(self, scale_dim: int = SCALE_DIM, task_dim: int = 0, out_dim: int = 64):
        super().__init__()
        d_in = scale_dim + task_dim
        self.net = nn.Sequential(nn.Linear(d_in, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim))

    def forward(self, scale: torch.Tensor, task: torch.Tensor | None = None) -> torch.Tensor:
        x = torch.cat([scale, task], dim=-1) if task is not None else scale
        return self.net(x)
