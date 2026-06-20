"""Minimal autoregressive architecture token generator (GRU)."""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.arch_gen.spec import ArchSpec, TOK_BOS, TOK_EOS, VOCAB_SIZE, all_valid_specs, random_spec


class ArchGenerator(nn.Module):
    """Predict next arch token given prefix; trained on valid BOS-H-D-EOS sequences."""

    def __init__(self, vocab: int = VOCAB_SIZE, hidden: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens (B,T) -> logits (B,T,vocab) for next-token prediction at each step."""
        x = self.embed(tokens)
        out, _ = self.gru(x)
        return self.head(out)

    def train_step(self, tokens: torch.Tensor) -> torch.Tensor:
        logits = self.forward(tokens[:, :-1])
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1))

    @torch.no_grad()
    def sample(self, n: int = 1, device: str = "cpu", max_len: int = 10) -> list[ArchSpec]:
        self.eval()
        specs: list[ArchSpec] = []
        for _ in range(n):
            seq = [TOK_BOS]
            for _ in range(max_len):
                t = torch.tensor([seq], device=device, dtype=torch.long)
                logits = self.forward(t)[0, -1]
                tok = int(torch.distributions.Categorical(logits=logits).sample())
                seq.append(tok)
                if tok == TOK_EOS: break
            try: specs.append(ArchSpec.from_tokens(seq))
            except ValueError: specs.append(random_spec(random.Random(len(specs) + 42)))
        return specs


def build_training_batch(specs: list[ArchSpec], device: str) -> torch.Tensor:
    rows = [s.to_tokens() for s in specs]
    max_len = max(len(r) for r in rows)
    padded = [r + [TOK_EOS] * (max_len - len(r)) for r in rows]
    return torch.tensor(padded, device=device, dtype=torch.long)


class ArchGeneratorTrainer:
    def __init__(self, model: ArchGenerator, device: str = "cpu", lr: float = 1e-3):
        self.model = model.to(device)
        self.device = device
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def step_batch(self, batch: torch.Tensor) -> float:
        self.model.train()
        loss = self.model.train_step(batch)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return loss.item()

    def fit(self, steps: int = 500, batch_size: int = 32) -> list[float]:
        specs = all_valid_specs()
        losses: list[float] = []
        for _ in range(steps):
            batch_specs = [random.choice(specs) for _ in range(batch_size)]
            losses.append(self.step_batch(build_training_batch(batch_specs, self.device)))
        return losses
