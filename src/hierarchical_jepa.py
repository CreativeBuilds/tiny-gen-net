"""Scale-aware hierarchical JEPA: global plan latent from task + target scale."""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.arch_gen.sentence_embedder import SENT_DIM
from src.jepa.weight_jepa import MLP
from src.scale_embedding import SCALE_DIM, ScaleEmbedding

LATENT_DIM = 64


class HighLevelScaleJEPA(nn.Module):
    """Level-1: (task, scale) → global plan h; JEPA predicts masked layer-summary latents."""

    def __init__(self, max_layers: int = 32, task_dim: int = SENT_DIM, latent_dim: int = LATENT_DIM, hidden: int = 256):
        super().__init__()
        self.latent_dim, self.max_layers = latent_dim, max_layers
        self.scale_emb = ScaleEmbedding(SCALE_DIM, task_dim, latent_dim)
        self.layer_pos = nn.Embedding(max_layers, latent_dim)
        self.enc = MLP(latent_dim * 2, latent_dim, hidden)
        self.target_enc = copy.deepcopy(self.enc)
        for p in self.target_enc.parameters(): p.requires_grad = False
        self.predictor = MLP(latent_dim * 3, latent_dim, hidden)
        self.scale_delta = nn.Sequential(nn.Linear(SCALE_DIM * 2 + latent_dim, hidden), nn.SiLU(), nn.Linear(hidden, latent_dim))
        self.cons_strong = True
        self.h_mean: torch.Tensor | None = None
        self.h_std: torch.Tensor | None = None

    @torch.no_grad()
    def update_target(self, ema: float = 0.99):
        for p, pt in zip(self.enc.parameters(), self.target_enc.parameters()):
            pt.data.mul_(ema).add_(p.data, alpha=1 - ema)

    def global_plan(self, task: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return self.scale_emb(scale, task)

    def layer_latent(self, flat_layer: torch.Tensor, layer_idx: int, enc: nn.Module) -> torch.Tensor:
        dev = self.layer_pos.weight.device
        if flat_layer.dim() == 1: flat_layer = flat_layer.unsqueeze(0)
        flat_layer = flat_layer.to(dev)
        B = flat_layer.shape[0]
        summary = flat_layer.mean(dim=-1, keepdim=True).expand(B, self.latent_dim)
        idx = torch.full((B,), min(layer_idx, self.max_layers - 1), device=self.layer_pos.weight.device)
        pos = self.layer_pos(idx)
        return enc(torch.cat([summary, pos], dim=-1))

    def train_step(self, task: torch.Tensor, scale: torch.Tensor, layer_flats: list[torch.Tensor], mask_ratio: float = 0.35) -> dict[str, torch.Tensor | float]:
        h = self.global_plan(task, scale)
        latents = [self.layer_latent(w, i, self.enc) for i, w in enumerate(layer_flats)]
        with torch.no_grad():
            targets = [self.layer_latent(w, i, self.target_enc) for i, w in enumerate(layer_flats)]
        n = len(latents)
        n_mask = max(1, int(n * mask_ratio))
        masked = torch.randperm(n)[:n_mask].tolist()
        jepa_loss, n_pred = 0.0, 0
        for ti in masked:
            visible = [i for i in range(n) if i not in masked]
            ctx = latents[visible[0]] if len(visible) == 1 else torch.stack([latents[i] for i in visible], dim=0).mean(dim=0)
            slot = self.layer_pos(torch.tensor([ti], device=h.device)).expand(h.shape[0], -1)
            pred = self.predictor(torch.cat([h, ctx, slot], dim=-1))
            jepa_loss = jepa_loss + F.mse_loss(pred, targets[ti])
            n_pred += 1
        recon = sum(F.mse_loss(l, t) for l, t in zip(latents, targets)) / n
        total = jepa_loss / max(n_pred, 1) + 0.25 * recon
        return {"total": total, "jepa": jepa_loss / max(n_pred, 1), "recon": recon, "plan": h}

    def scale_consistency(self, task: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor) -> torch.Tensor:
        """Same task at two scales → aligned plans + scale-delta prediction."""
        if task.dim() == 1: task = task.unsqueeze(0)
        if scale_a.dim() == 1: scale_a = scale_a.unsqueeze(0)
        if scale_b.dim() == 1: scale_b = scale_b.unsqueeze(0)
        ha, hb = self.global_plan(task, scale_a), self.global_plan(task, scale_b)
        cos = 1.0 - F.cosine_similarity(ha, hb, dim=-1).mean()
        if not self.cons_strong: return cos
        align = F.mse_loss(F.normalize(ha, dim=-1), F.normalize(hb, dim=-1))
        pred = self.scale_delta(torch.cat([scale_a, scale_b, ha], dim=-1))
        return cos + 0.4 * align + 0.35 * F.mse_loss(pred, hb)

    @torch.no_grad()
    def fit_plan_stats(self, plans: torch.Tensor):
        self.h_mean, self.h_std = plans.mean(dim=0), plans.std(dim=0).clamp_min(1e-6)

    @torch.no_grad()
    def sample_plan(self, task: torch.Tensor, scale: torch.Tensor, device: str) -> torch.Tensor:
        h = self.global_plan(task.to(device), scale.to(device))
        if self.h_mean is None: return h
        return h + 0.15 * self.h_std * torch.randn_like(h)


class HighLevelScaleJEPATrainer:
    def __init__(self, model: HighLevelScaleJEPA, device: str = "cpu", lr: float = 1e-3, ema: float = 0.99):
        self.model, self.device, self.ema = model.to(device), device, ema
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def step(self, task: torch.Tensor, scale: torch.Tensor, layer_flats: list[torch.Tensor]) -> dict[str, float]:
        self.model.train()
        losses = self.model.train_step(task, scale, layer_flats)
        self.opt.zero_grad(); losses["total"].backward(); self.opt.step()
        self.model.update_target(self.ema)
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
