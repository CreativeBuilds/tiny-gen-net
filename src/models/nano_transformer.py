"""nanoGPT-style causal transformer for real char-level text (Phase 5a)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.arch_gen.nano_spec import NanoSpec


class NanoBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, batch_first=True)
        ff = 4 * n_embd
        self.mlp = nn.Sequential(nn.Linear(n_embd, ff), nn.GELU(), nn.Linear(ff, n_embd))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + h
        return x + self.mlp(self.ln2(x))


class NanoTransformer(nn.Module):
    """Full-sequence causal LM: (B,T) -> logits (B,T,V) + optional CE loss."""

    def __init__(self, spec: NanoSpec, vocab_size: int):
        super().__init__()
        spec.validate()
        self.spec, self.n_embd, self.n_layer, self.n_head = spec, spec.n_embd, spec.n_layer, spec.n_head
        self.block_size, self.vocab_size = spec.block_size, vocab_size
        self.tok = nn.Embedding(vocab_size, spec.n_embd)
        self.pos = nn.Embedding(spec.block_size, spec.n_embd)
        self.blocks = nn.ModuleList([NanoBlock(spec.n_embd, spec.n_head) for _ in range(spec.n_layer)])
        self.ln_f = nn.LayerNorm(spec.n_embd)
        self.lm_head = nn.Linear(spec.n_embd, vocab_size, bias=False)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.normal_(m.weight, 0.0, 0.02); 
            if isinstance(m, nn.Linear) and m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, 0.0, 0.02)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((T, T), float("-inf"), device=device), diagonal=1)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        T = min(T, self.block_size)
        idx = idx[:, -T:]
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos).unsqueeze(0)
        mask = self._causal_mask(T, idx.device)
        for blk in self.blocks: x = blk(x, mask)
        logits = self.lm_head(self.ln_f(x))
        if targets is None: return logits, None
        targets = targets[:, -T:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def config_dict(self) -> dict:
        return {"n_embd": self.n_embd, "n_layer": self.n_layer, "n_head": self.n_head, "block_size": self.block_size,
                "vocab_size": self.vocab_size, "arch": "NanoTransformer"}


def build_from_nano_spec(spec: NanoSpec, vocab_size: int) -> NanoTransformer:
    return NanoTransformer(spec.validate(), vocab_size)
