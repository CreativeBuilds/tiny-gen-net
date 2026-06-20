"""Task-conditioned JEPA with sentence embedding in predictor context (Phase 5a)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.arch_gen.sentence_embedder import SENT_DIM
from src.jepa.chunks import split_chunks
from src.jepa.weight_jepa import WeightJEPA


class TaskCondWeightJEPA(WeightJEPA):
    """Extends WeightJEPA: task sentence vector biases pooled context before prediction."""

    def __init__(self, bounds: list[tuple[int, int]], task_dim: int = SENT_DIM, latent_dim: int = 64, hidden: int = 512):
        super().__init__(bounds, latent_dim, hidden)
        self.task_proj = nn.Linear(task_dim, latent_dim)

    def train_step(self, weights: torch.Tensor, task_vecs: torch.Tensor | None = None, mask_ratio: float = 0.4) -> dict[str, torch.Tensor | float]:
        B = weights.shape[0]
        chunks = split_chunks(weights, self.bounds)
        latents = self.encode_chunks(chunks, self.encoders)
        with torch.no_grad():
            target_latents = self.encode_chunks(chunks, self.target_encoders)

        jepa_loss, n_pred = 0.0, 0
        for b in range(B):
            n_mask = max(1, int(self.num_chunks * mask_ratio))
            masked = torch.randperm(self.num_chunks)[:n_mask].tolist()
            visible = [i for i in range(self.num_chunks) if i not in masked]
            ctx = self._pool_context([latents[i][b : b + 1] for i in range(self.num_chunks)], visible)
            if task_vecs is not None: ctx = ctx + self.task_proj(task_vecs[b : b + 1])
            for ti in masked:
                idx = torch.tensor([ti], device=weights.device)
                pred = self.predictor(torch.cat([ctx, self.chunk_embed(idx)], dim=-1))
                jepa_loss = jepa_loss + F.mse_loss(pred, target_latents[ti][b : b + 1])
                n_pred += 1

        recon_chunks = self.decode_latents(latents)
        recon_loss = sum(F.mse_loss(rc, c) for rc, c in zip(recon_chunks, chunks)) / self.num_chunks
        total = jepa_loss / max(n_pred, 1) + 0.5 * recon_loss
        return {"total": total, "jepa": jepa_loss / max(n_pred, 1), "recon": recon_loss}

    @torch.no_grad()
    def sample_cond(self, task_vec: torch.Tensor, device: str = "cpu") -> torch.Tensor:
        if self.latent_mean is None: raise RuntimeError("Call fit_latent_stats before sample_cond")
        bias = self.task_proj(task_vec.unsqueeze(0).to(device)).squeeze(0)
        latents = []
        for i in range(self.num_chunks):
            z = (self.latent_mean[i].to(device) + bias + self.latent_std[i].to(device) * torch.randn(self.latent_dim, device=device)).unsqueeze(0)
            latents.append(z)
        from src.jepa.chunks import merge_chunks
        out = merge_chunks(self.decode_latents(latents), self.bounds, self.weight_dim)
        return out.squeeze(0) if out.dim() > 1 else out


class TaskCondWeightJEPATrainer:
    def __init__(self, model: TaskCondWeightJEPA, device: str = "cpu", lr: float = 1e-3, ema: float = 0.99):
        self.model = model.to(device)
        self.device, self.ema = device, ema
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def step_batch(self, batch: torch.Tensor, task_vecs: torch.Tensor | None = None) -> dict[str, float]:
        self.model.train()
        losses = self.model.train_step(batch.to(self.device), task_vecs.to(self.device) if task_vecs is not None else None)
        self.opt.zero_grad()
        losses["total"].backward()
        self.opt.step()
        self.model.update_target_encoder(self.ema)
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}

    def fit(self, weights: list[torch.Tensor], texts: list[str], steps: int = 1200, batch_size: int = 4, enc=None) -> list[float]:
        from src.arch_gen.sentence_embedder import SentenceTaskEmbedder
        enc = enc or SentenceTaskEmbedder()
        stacked = torch.stack(weights)
        n, losses = stacked.shape[0], []
        for _ in range(steps):
            idx = torch.randint(0, n, (min(batch_size, n),)).tolist()
            batch = stacked[idx]
            tvecs = torch.stack([enc.encode_text(texts[i], self.device) for i in idx])
            losses.append(self.step_batch(batch, tvecs)["total"])
        self.model.fit_latent_stats(stacked.to(self.device))
        return losses
