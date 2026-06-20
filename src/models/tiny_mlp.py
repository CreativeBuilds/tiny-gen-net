"""Tiny MLP for next-character prediction (Phase 0 fixed architecture).

Architecture:
  embed(char) -> concat with prev hidden or zeros -> Linear -> ReLU -> Linear -> logits

We keep it deliberately small (~2k params) so weight vectors are easy to diffuse.
"""

import torch
import torch.nn as nn
from data.synthetic_text import VOCAB_SIZE


class TinyMLP(nn.Module):
    def __init__(self, hidden_dim: int = 64, vocab_size: int = VOCAB_SIZE):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        # Input: current embedding + previous hidden state
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B,) char indices
        h_prev: (B, hidden_dim) or None (uses zeros)
        Returns: logits (B, vocab), new hidden (B, hidden_dim)
        """
        B = x.shape[0]
        emb = self.embed(x)  # (B, H)
        if h_prev is None: h_prev = torch.zeros(B, self.hidden_dim, device=x.device, dtype=emb.dtype)
        inp = torch.cat([emb, h_prev], dim=-1)
        h = torch.relu(self.fc1(inp))
        logits = self.fc2(h)
        return logits, h

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def config_dict(self) -> dict:
        return {"hidden_dim": self.hidden_dim, "vocab_size": self.vocab_size, "arch": "TinyMLP"}


def build_tiny_mlp(hidden_dim: int = 64) -> TinyMLP:
    m = TinyMLP(hidden_dim=hidden_dim)
    print(f"TinyMLP params: {m.num_parameters():,}")
    return m
