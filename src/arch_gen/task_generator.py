"""Task-conditioned architecture generator (GRU + task embedding + match-aware aux)."""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.arch_gen.spec import ArchSpec, TOK_BOS, TOK_EOS, VOCAB_SIZE, random_spec
from src.arch_gen.task_cond import ArchMatchHead, TaskEmbedder


class TaskArchGenerator(nn.Module):
    def __init__(self, vocab: int = VOCAB_SIZE, hidden: int = 64, task_dim: int = 32):
        super().__init__()
        self.hidden, self.task_dim = hidden, task_dim
        self.task_enc = TaskEmbedder(task_dim)
        self.match_head = ArchMatchHead(task_dim)
        self.tok = nn.Embedding(vocab, hidden)
        self.task_proj = nn.Linear(task_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, tokens: torch.Tensor, task_vec: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        t = self.task_proj(task_vec).unsqueeze(1).expand(B, tokens.shape[1], -1)
        out, _ = self.gru(self.tok(tokens) + t)
        return self.head(out)

    def train_step(self, tokens: torch.Tensor, task_vec: torch.Tensor, specs: list[ArchSpec], match_w: float = 0.5) -> torch.Tensor:
        logits = self.forward(tokens[:, :-1], task_vec)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1))
        if match_w <= 0 or not specs: return ce
        return ce + match_w * self.match_head.loss(task_vec, specs)

    @torch.no_grad()
    def sample(self, n: int, task_text: str, device: str, max_len: int = 10) -> list[ArchSpec]:
        self.eval()
        tv = self.task_enc.encode_text(task_text, device)
        specs: list[ArchSpec] = []
        for _ in range(n):
            seq = [TOK_BOS]
            for _ in range(max_len):
                t = torch.tensor([seq], device=device, dtype=torch.long)
                logits = self.forward(t, tv.unsqueeze(0))[0, -1]
                tok = int(torch.distributions.Categorical(logits=logits).sample())
                seq.append(tok)
                if tok == TOK_EOS: break
            try: specs.append(ArchSpec.from_tokens(seq))
            except ValueError: specs.append(random_spec(random.Random(len(specs) + 7)))
        return specs


def build_task_arch_batch(specs: list[ArchSpec], texts: list[str], device: str, enc: TaskEmbedder) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [s.to_tokens() for s in specs]
    max_len = max(len(r) for r in rows)
    padded = [r + [TOK_EOS] * (max_len - len(r)) for r in rows]
    tokens = torch.tensor(padded, device=device, dtype=torch.long)
    tasks = torch.stack([enc.encode_text(t, device) for t in texts])
    return tokens, tasks


class TaskArchTrainer:
    def __init__(self, model: TaskArchGenerator, device: str = "cpu", lr: float = 1e-3, match_weight: float = 0.5):
        self.model = model.to(device)
        self.device, self.match_weight = device, match_weight
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def step_batch(self, tokens: torch.Tensor, task_vecs: torch.Tensor, specs: list[ArchSpec]) -> float:
        self.model.train()
        loss = self.model.train_step(tokens, task_vecs, specs, self.match_weight)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return loss.item()

    def fit_paired(self, specs: list[ArchSpec], texts: list[str], steps: int = 1200, batch_size: int = 16) -> list[float]:
        n, losses = len(specs), []
        enc = self.model.task_enc
        for _ in range(steps):
            idx = torch.randint(0, n, (min(batch_size, n),)).tolist()
            batch_s = [specs[i] for i in idx]
            batch_t = [texts[i] for i in idx]
            tokens, tvecs = build_task_arch_batch(batch_s, batch_t, self.device, enc)
            losses.append(self.step_batch(tokens, tvecs, batch_s))
        return losses
