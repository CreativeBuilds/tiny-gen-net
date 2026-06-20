"""Task-conditioned transformer architecture generator."""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.arch_gen.task_cond import TaskEmbedder
from src.arch_gen.tx_spec import TX_VOCAB_SIZE, TOK_BOS, TOK_EOS, TxSpec, random_tx_spec
from src.arch_gen.tx_task_cond import TxMatchHead, describe_tx_spec, enrich_tx_task


class TaskTxArchGenerator(nn.Module):
    def __init__(self, vocab: int = TX_VOCAB_SIZE, hidden: int = 64, task_dim: int = 32):
        super().__init__()
        self.task_enc = TaskEmbedder(task_dim)
        self.match_head = TxMatchHead(task_dim)
        self.tok = nn.Embedding(vocab, hidden)
        self.task_proj = nn.Linear(task_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, vocab)

    def forward(self, tokens: torch.Tensor, task_vec: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        t = self.task_proj(task_vec).unsqueeze(1).expand(B, tokens.shape[1], -1)
        out, _ = self.gru(self.tok(tokens) + t)
        return self.head(out)

    def train_step(self, tokens: torch.Tensor, task_vec: torch.Tensor, specs: list[TxSpec], match_w: float = 0.5) -> torch.Tensor:
        logits = self.forward(tokens[:, :-1], task_vec)
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1))
        if match_w <= 0 or not specs: return ce
        return ce + match_w * self.match_head.loss(task_vec, specs)

    @torch.no_grad()
    def sample(self, n: int, task_text: str, device: str, max_len: int = 10) -> list[TxSpec]:
        self.eval()
        tv = self.task_enc.encode_text(enrich_tx_task(task_text), device)
        specs: list[TxSpec] = []
        for _ in range(n):
            seq = [TOK_BOS]
            for _ in range(max_len):
                t = torch.tensor([seq], device=device, dtype=torch.long)
                tok = int(torch.distributions.Categorical(logits=self.forward(t, tv.unsqueeze(0))[0, -1]).sample())
                seq.append(tok)
                if tok == TOK_EOS: break
            try: specs.append(TxSpec.from_tokens(seq))
            except ValueError: specs.append(random_tx_spec(random.Random(len(specs) + 11)))
        return specs


def build_tx_arch_batch(specs: list[TxSpec], texts: list[str], device: str, enc: TaskEmbedder) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [s.to_tokens() for s in specs]
    max_len = max(len(r) for r in rows)
    padded = [r + [TOK_EOS] * (max_len - len(r)) for r in rows]
    tokens = torch.tensor(padded, device=device, dtype=torch.long)
    tasks = torch.stack([enc.encode_text(enrich_tx_task(t), device) for t in texts])
    return tokens, tasks


class TaskTxArchTrainer:
    def __init__(self, model: TaskTxArchGenerator, device: str = "cpu", lr: float = 1e-3, match_weight: float = 0.5):
        self.model = model.to(device)
        self.device, self.match_weight = device, match_weight
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def fit_paired(self, specs: list[TxSpec], texts: list[str], steps: int = 800, batch_size: int = 8) -> list[float]:
        n, losses, enc = len(specs), [], self.model.task_enc
        for _ in range(steps):
            idx = torch.randint(0, n, (min(batch_size, n),)).tolist()
            batch_s = [specs[i] for i in idx]
            batch_t = [texts[i] for i in idx]
            tokens, tvecs = build_tx_arch_batch(batch_s, batch_t, self.device, enc)
            self.model.train()
            loss = self.model.train_step(tokens, tvecs, batch_s, self.match_weight)
            self.opt.zero_grad(); loss.backward(); self.opt.step()
            losses.append(loss.item())
        return losses
