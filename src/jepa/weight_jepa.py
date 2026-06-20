"""Minimal JEPA for global-normalized weight vectors (Phase 2).

Pattern: mask random layer chunks -> context encoder sees visible chunks ->
predictor predicts target chunk latents -> match EMA target encoder latents.
Sampling: fit Gaussian per chunk latent on training data, decode to weights.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.jepa.chunks import chunk_dims, merge_chunks, split_chunks


class MLP(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int = 256):
        super().__init__()
        h = min(hidden, max(d_in, d_out) * 2)
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WeightJEPA(nn.Module):
    """Layer-chunk JEPA with per-chunk encoders, EMA target, and decoders for generation."""

    def __init__(self, bounds: list[tuple[int, int]], latent_dim: int = 64, hidden: int = 256):
        super().__init__()
        self.bounds = bounds
        self.dims = chunk_dims(bounds)
        self.weight_dim = bounds[-1][1]
        self.latent_dim = latent_dim
        self.num_chunks = len(bounds)
        self.encoders = nn.ModuleList([MLP(d, latent_dim, hidden) for d in self.dims])
        self.decoders = nn.ModuleList([MLP(latent_dim, d, hidden) for d in self.dims])
        self.target_encoders = copy.deepcopy(self.encoders)
        for p in self.target_encoders.parameters(): p.requires_grad = False
        self.chunk_embed = nn.Embedding(self.num_chunks, latent_dim)
        # predictor: pooled context + target slot embedding -> target latent
        self.predictor = MLP(latent_dim * 2, latent_dim, hidden)
        self.latent_mean: torch.Tensor | None = None
        self.latent_std: torch.Tensor | None = None

    def encode_chunks(self, chunks: list[torch.Tensor], encoders: nn.ModuleList) -> list[torch.Tensor]:
        return [enc(c) for enc, c in zip(encoders, chunks)]

    def decode_latents(self, latents: list[torch.Tensor]) -> list[torch.Tensor]:
        return [dec(z) for dec, z in zip(self.decoders, latents)]

    @torch.no_grad()
    def update_target_encoder(self, momentum: float = 0.99):
        for p, pt in zip(self.encoders.parameters(), self.target_encoders.parameters()):
            pt.data.mul_(momentum).add_(p.data, alpha=1 - momentum)

    def _pool_context(self, latents: list[torch.Tensor], visible: list[int]) -> torch.Tensor:
        if not visible: return torch.zeros(latents[0].shape[0], self.latent_dim, device=latents[0].device)
        return torch.stack([latents[i] for i in visible], dim=0).mean(dim=0)

    def train_step(self, weights: torch.Tensor, mask_ratio: float = 0.4) -> dict[str, float]:
        """One JEPA step on batch of global-normalized flat weights (B, D)."""
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
    def fit_latent_stats(self, weights: torch.Tensor):
        """Store per-chunk latent mean/std from training set for sampling."""
        chunks = split_chunks(weights, self.bounds)
        latents = self.encode_chunks(chunks, self.encoders)
        means, stds = [], []
        for z in latents:
            means.append(z.mean(dim=0))
            stds.append(z.std(dim=0).clamp_min(1e-6))
        self.latent_mean = torch.stack(means)
        self.latent_std = torch.stack(stds)

    @torch.no_grad()
    def sample(self, n: int = 1, device: str = "cpu") -> torch.Tensor:
        """Sample n weight vectors in normalized space via latent Gaussian + decode."""
        if self.latent_mean is None: raise RuntimeError("Call fit_latent_stats before sample")
        latents = []
        for i in range(self.num_chunks):
            z = self.latent_mean[i] + self.latent_std[i] * torch.randn(n, self.latent_dim, device=device)
            latents.append(z)
        chunks = self.decode_latents(latents)
        return merge_chunks(chunks, self.bounds, self.weight_dim)

    def save(self, path: str):
        torch.save({"state_dict": self.state_dict(), "bounds": self.bounds, "latent_dim": self.latent_dim,
                    "latent_mean": self.latent_mean, "latent_std": self.latent_std}, path)

    def load(self, path: str, device: str = "cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=True)
        self.load_state_dict(ckpt["state_dict"])
        self.latent_mean = ckpt.get("latent_mean")
        self.latent_std = ckpt.get("latent_std")


class WeightJEPATrainer:
    """Training loop wrapper."""

    def __init__(self, model: WeightJEPA, device: str = "cpu", lr: float = 1e-3, ema: float = 0.99):
        self.model = model.to(device)
        self.device = device
        self.ema = ema
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def step_batch(self, batch: torch.Tensor) -> dict[str, float]:
        self.model.train()
        losses = self.model.train_step(batch.to(self.device))
        self.opt.zero_grad()
        losses["total"].backward()
        self.opt.step()
        self.model.update_target_encoder(self.ema)
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
