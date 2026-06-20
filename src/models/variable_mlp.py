"""Variable-width/depth char MLP with optional skip — same forward API as TinyMLP."""

import torch
import torch.nn as nn

from data.synthetic_text import VOCAB_SIZE
from src.arch_gen.spec import ArchSpec


class VariableTinyMLP(nn.Module):
    """embed -> (depth x Linear+ReLU [+ optional skip]) -> logits."""

    def __init__(self, spec: ArchSpec, vocab_size: int = VOCAB_SIZE):
        super().__init__()
        spec.validate()
        self.spec = spec
        self.hidden_dim, self.depth, self.skip, self.vocab_size = spec.hidden_dim, spec.depth, spec.skip, vocab_size
        h = spec.hidden_dim
        self.embed = nn.Embedding(vocab_size, h)
        blocks: list[nn.Module] = [nn.Linear(h * 2, h)]
        for _ in range(spec.depth - 1): blocks.append(nn.Linear(h, h))
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Linear(h, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear): nn.init.normal_(m.weight, 0.0, 0.02); nn.init.zeros_(m.bias) if m.bias is not None else None
            elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        B = x.shape[0]
        emb = self.embed(x)
        if h_prev is None: h_prev = torch.zeros(B, self.hidden_dim, device=x.device, dtype=emb.dtype)
        h = torch.relu(self.blocks[0](torch.cat([emb, h_prev], dim=-1)))
        if self.skip: h = h + h_prev
        for layer in self.blocks[1:]: h = torch.relu(layer(h))
        return self.head(h), h

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def config_dict(self) -> dict:
        return {"hidden_dim": self.hidden_dim, "depth": self.depth, "skip": self.skip, "vocab_size": self.vocab_size, "arch": "VariableTinyMLP"}


def build_from_spec(spec: ArchSpec) -> VariableTinyMLP:
    return VariableTinyMLP(spec.validate())
