"""Minimal 2-level Hierarchical JEPA (H-JEPA) for weight vectors.

Level 0 (low):  per-layer-chunk encoders/decoders — same granularity as flat JEPA.
Level 1 (high): global embedding h = high_encoder(mean(z_i)) captures cross-layer structure.

Information flow:
  bottom-up: chunk latents z_i -> pooled -> h
  top-down:  h conditions low-level chunk prediction (predictor sees ctx + slot + h)
  high JEPA: mask chunks, predict z_target from partial pool via high_predictor

Sampling: draw h ~ N(h_mean, h_std), predict each z_i = high_predictor(h, slot_i) + jitter, decode.
Same sample() interface as WeightJEPA for hybrid pipeline compatibility.
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.jepa.chunks import chunk_dims, merge_chunks, split_chunks
from src.jepa.weight_jepa import MLP


class HierarchicalWeightJEPA(nn.Module):
    """2-level JEPA: low chunk encoders + high global structure + bidirectional coupling."""

    def __init__(self, bounds: list[tuple[int, int]], latent_dim: int = 64, hidden: int = 256):
        super().__init__()
        self.bounds, self.dims = bounds, chunk_dims(bounds)
        self.weight_dim, self.latent_dim, self.num_chunks = bounds[-1][1], latent_dim, len(bounds)
        # --- low level: per-chunk encode/decode (identical role to flat JEPA) ---
        self.encoders = nn.ModuleList([MLP(d, latent_dim, hidden) for d in self.dims])
        self.decoders = nn.ModuleList([MLP(latent_dim, d, hidden) for d in self.dims])
        self.target_encoders = copy.deepcopy(self.encoders)
        for p in self.target_encoders.parameters(): p.requires_grad = False
        self.chunk_embed = nn.Embedding(self.num_chunks, latent_dim)
        # top-down: low predictor sees visible ctx + target slot + global h
        self.low_predictor = MLP(latent_dim * 3, latent_dim, hidden)
        # --- high level: global structure from pooled chunk latents ---
        self.high_encoder = MLP(latent_dim, latent_dim, hidden)
        self.high_target = copy.deepcopy(self.high_encoder)
        for p in self.high_target.parameters(): p.requires_grad = False
        # high JEPA: partial pool embedding + slot -> target chunk latent
        self.high_predictor = MLP(latent_dim * 2, latent_dim, hidden)
        self.latent_mean: torch.Tensor | None = None
        self.latent_std: torch.Tensor | None = None
        self.h_mean: torch.Tensor | None = None
        self.h_std: torch.Tensor | None = None

    def encode_chunks(self, chunks: list[torch.Tensor], encoders: nn.ModuleList) -> list[torch.Tensor]:
        return [enc(c) for enc, c in zip(encoders, chunks)]

    def decode_latents(self, latents: list[torch.Tensor]) -> list[torch.Tensor]:
        return [dec(z) for dec, z in zip(self.decoders, latents)]

    def _pool(self, latents: list[torch.Tensor], idx: list[int]) -> torch.Tensor:
        if not idx: return torch.zeros(latents[0].shape[0], self.latent_dim, device=latents[0].device)
        return torch.stack([latents[i] for i in idx], dim=0).mean(dim=0)

    @torch.no_grad()
    def update_target_encoder(self, momentum: float = 0.99):
        for p, pt in zip(self.encoders.parameters(), self.target_encoders.parameters()):
            pt.data.mul_(momentum).add_(p.data, alpha=1 - momentum)
        for p, pt in zip(self.high_encoder.parameters(), self.high_target.parameters()):
            pt.data.mul_(momentum).add_(p.data, alpha=1 - momentum)

    def train_step(self, weights: torch.Tensor, mask_ratio: float = 0.4) -> dict[str, float]:
        B = weights.shape[0]
        chunks = split_chunks(weights, self.bounds)
        latents = self.encode_chunks(chunks, self.encoders)
        with torch.no_grad():
            target_latents = self.encode_chunks(chunks, self.target_encoders)
        # bottom-up global embedding for entire model (batch-level h)
        h = self.high_encoder(self._pool(latents, list(range(self.num_chunks))))
        low_loss, high_loss, n_pred = 0.0, 0.0, 0
        for b in range(B):
            n_mask = max(1, int(self.num_chunks * mask_ratio))
            masked = torch.randperm(self.num_chunks)[:n_mask].tolist()
            visible = [i for i in range(self.num_chunks) if i not in masked]
            ctx = self._pool([latents[i][b : b + 1] for i in range(self.num_chunks)], visible)
            hb = h[b : b + 1]
            h_ctx = self.high_encoder(ctx)  # high view of visible-only context
            for ti in masked:
                idx = torch.tensor([ti], device=weights.device)
                slot = self.chunk_embed(idx)
                low_pred = self.low_predictor(torch.cat([ctx, slot, hb], dim=-1))
                high_pred = self.high_predictor(torch.cat([h_ctx, slot], dim=-1))
                tgt = target_latents[ti][b : b + 1]
                low_loss = low_loss + F.mse_loss(low_pred, tgt)
                high_loss = high_loss + F.mse_loss(high_pred, tgt)
                n_pred += 1
        recon = sum(F.mse_loss(rc, c) for rc, c in zip(self.decode_latents(latents), chunks)) / self.num_chunks
        n = max(n_pred, 1)
        low_l, high_l = low_loss / n, high_loss / n
        total = low_l + high_l + 0.5 * recon
        return {"total": total, "low_jepa": low_l, "high_jepa": high_l, "recon": recon}

    @torch.no_grad()
    def fit_latent_stats(self, weights: torch.Tensor):
        chunks = split_chunks(weights, self.bounds)
        latents = self.encode_chunks(chunks, self.encoders)
        means, stds, hs = [], [], []
        for z in latents:
            means.append(z.mean(dim=0)); stds.append(z.std(dim=0).clamp_min(1e-6))
        for b in range(weights.shape[0]):
            hs.append(self.high_encoder(self._pool([latents[i][b : b + 1] for i in range(self.num_chunks)], list(range(self.num_chunks)))))
        self.latent_mean, self.latent_std = torch.stack(means), torch.stack(stds)
        h_stack = torch.cat(hs, dim=0)
        self.h_mean, self.h_std = h_stack.mean(dim=0), h_stack.std(dim=0).clamp_min(1e-6)

    @torch.no_grad()
    def sample(self, n: int = 1, device: str = "cpu") -> torch.Tensor:
        if self.h_mean is None: raise RuntimeError("Call fit_latent_stats before sample")
        h = self.h_mean + self.h_std * torch.randn(n, self.latent_dim, device=device)
        latents = []
        for i in range(self.num_chunks):
            idx = torch.full((n,), i, device=device, dtype=torch.long)
            z = self.high_predictor(torch.cat([h, self.chunk_embed(idx)], dim=-1))
            if self.latent_std is not None: z = z + 0.25 * self.latent_std[i] * torch.randn(n, self.latent_dim, device=device)
            latents.append(z)
        return merge_chunks(self.decode_latents(latents), self.bounds, self.weight_dim)

    def save(self, path: str):
        torch.save({"state_dict": self.state_dict(), "bounds": self.bounds, "latent_dim": self.latent_dim,
                    "latent_mean": self.latent_mean, "latent_std": self.latent_std, "h_mean": self.h_mean, "h_std": self.h_std}, path)

    def load(self, path: str, device: str = "cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=True)
        self.load_state_dict(ckpt["state_dict"])
        self.latent_mean, self.latent_std = ckpt.get("latent_mean"), ckpt.get("latent_std")
        self.h_mean, self.h_std = ckpt.get("h_mean"), ckpt.get("h_std")


class HierarchicalWeightJEPATrainer:
    def __init__(self, model: HierarchicalWeightJEPA, device: str = "cpu", lr: float = 1e-3, ema: float = 0.99):
        self.model, self.device, self.ema = model.to(device), device, ema
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def step_batch(self, batch: torch.Tensor) -> dict[str, float]:
        self.model.train()
        losses = self.model.train_step(batch.to(self.device))
        self.opt.zero_grad(); losses["total"].backward(); self.opt.step()
        self.model.update_target_encoder(self.ema)
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
