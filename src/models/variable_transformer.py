"""Minimal decoder-only transformer for next-char prediction (Phase 4b).

Width-growth note (Task 004)
----------------------------
`TxBlock` supports a width-growth-safe normalization. The default `nn.LayerNorm`
normalizes over the FULL residual width; when a model is grown Ds->Dt by
zero-padding new dims, LN's per-token variance is computed over (Ds real + new
zeros), which shrinks the denominator and re-scales the copied dims -> the grown
model is only APPROXIMATELY function-preserving (Task 003 measured 1.26 max-abs
logit diff on a trained source).

`ActiveLayerNorm` fixes this: it computes the LayerNorm mean/variance over only
the first `active_dim` channels (the original Ds dims), while still applying the
per-channel affine (gamma/beta) across the full width Dt. When the new dims are
exactly zero (gamma=beta=0 there), this reproduces the source's normalization
EXACTLY, so zero-pad width growth becomes exactly function-preserving.

Backward compatible: a freshly built model has `active_dim == d_model`, which is
numerically identical to standard LayerNorm, so prior phases do not regress.
`grow_tx_width` sets each grown norm's `active_dim = Ds` to exclude the padded
dims from the statistics.
"""

import torch
import torch.nn as nn

from data.synthetic_text import VOCAB_SIZE
from src.arch_gen.tx_spec import CTX_LEN, TxSpec


class ActiveLayerNorm(nn.Module):
    """LayerNorm whose mean/variance use only the first `active_dim` channels.

    Affine params (weight/bias) span the full width `d`. With active_dim == d this
    is exactly nn.LayerNorm. After width growth, active_dim is set to the original
    Ds so zero-padded new dims do not perturb the normalization of the copied dims.
    """

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.d = d
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))
        # default: normalize over the full width (== nn.LayerNorm)
        self.register_buffer("active_dim", torch.tensor(d, dtype=torch.long), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = int(self.active_dim)
        if a == self.d:
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, unbiased=False, keepdim=True)
            xhat = (x - mean) / torch.sqrt(var + self.eps)
        else:
            # stats over first `a` channels only; normalize ALL channels by them.
            active = x[..., :a]
            mean = active.mean(dim=-1, keepdim=True)
            var = active.var(dim=-1, unbiased=False, keepdim=True)
            xhat = (x - mean) / torch.sqrt(var + self.eps)
        return xhat * self.weight + self.bias

    def set_active_dim(self, a: int) -> None:
        self.active_dim = torch.tensor(int(a), dtype=torch.long, device=self.active_dim.device)


class TxBlock(nn.Module):
    def __init__(self, d: int, n_head: int, ff: int):
        super().__init__()
        self.ln1, self.ln2 = ActiveLayerNorm(d), ActiveLayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_head, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + h
        return x + self.ff(self.ln2(x))


class VariableTinyTransformer(nn.Module):
    """Causal transformer: (B,T) char ctx -> logits for next char at last position."""

    def __init__(self, spec: TxSpec, vocab_size: int = VOCAB_SIZE):
        super().__init__()
        spec.validate()
        self.spec, self.d_model, self.n_layer, self.n_head = spec, spec.d_model, spec.n_layer, spec.n_head
        self.ctx_len, self.vocab_size = spec.ctx_len, vocab_size
        ff = max(spec.d_model // 2, 16)
        self.tok = nn.Embedding(vocab_size, spec.d_model)
        self.pos = nn.Embedding(spec.ctx_len, spec.d_model)
        self.blocks = nn.ModuleList([TxBlock(spec.d_model, spec.n_head, ff) for _ in range(spec.n_layer)])
        self.head = nn.Linear(spec.d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.normal_(m.weight, 0.0, 0.02); nn.init.zeros_(m.bias) if m.bias is not None else None
            elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, 0.0, 0.02)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        B, T = x.shape
        T = min(T, self.ctx_len)
        x = x[:, -T:]
        pos = torch.arange(T, device=x.device)
        h = self.tok(x) + self.pos(pos).unsqueeze(0)
        mask = self._causal_mask(T, x.device)
        for blk in self.blocks: h = blk(h, mask)
        return self.head(h[:, -1, :]), None

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def config_dict(self) -> dict:
        return {"d_model": self.d_model, "n_layer": self.n_layer, "n_head": self.n_head, "ctx_len": self.ctx_len,
                "vocab_size": self.vocab_size, "arch": "VariableTinyTransformer"}


def build_from_tx_spec(spec: TxSpec) -> VariableTinyTransformer:
    return VariableTinyTransformer(spec.validate())
