"""Minimal decoder-only transformer for next-char prediction (Phase 4b)."""

import torch
import torch.nn as nn

from data.synthetic_text import VOCAB_SIZE
from src.arch_gen.tx_spec import CTX_LEN, TxSpec


class TxBlock(nn.Module):
    def __init__(self, d: int, n_head: int, ff: int):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
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
