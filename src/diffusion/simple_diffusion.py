"""Minimal DDPM for weight vectors (Phase 1).

Classic linear beta schedule + MLP epsilon predictor.
Educational, single-file-friendly — not SOTA diffusion.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


class WeightDenoiser(nn.Module):
    """Predict noise epsilon given noisy weights w_t and timestep t."""

    def __init__(self, dim: int, hidden: int = 512, time_dim: int = 128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(dim + hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )
        self.time_dim = time_dim

    def _time_embed(self, t: torch.Tensor) -> torch.Tensor:
        # Sinusoidal embedding (Transformer-style)
        half = self.time_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.time_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.time_mlp(emb)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self._time_embed(t)
        return self.net(torch.cat([x, te], dim=-1))


class WeightDiffusion:
    """DDPM wrapper: training step + ancestral sampling."""

    def __init__(self, dim: int, timesteps: int = 200, device: str = "cpu"):
        self.dim = dim
        self.timesteps = timesteps
        self.device = device
        betas = linear_beta_schedule(timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register = {
            "betas": betas,
            "alphas": alphas,
            "alphas_cumprod": alphas_cumprod,
            "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
            "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
        }
        self.model = WeightDenoiser(dim).to(device)

    def _buf(self, name: str) -> torch.Tensor:
        return self.register[name].to(self.device)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None: noise = torch.randn_like(x0)
        sa = self._buf("sqrt_alphas_cumprod")[t].unsqueeze(-1)
        so = self._buf("sqrt_one_minus_alphas_cumprod")[t].unsqueeze(-1)
        return sa * x0 + so * noise

    def train_step(self, x0: torch.Tensor, opt: torch.optim.Optimizer) -> float:
        B = x0.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=self.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred = self.model(xt, t)
        loss = F.mse_loss(pred, noise)
        opt.zero_grad()
        loss.backward()
        opt.step()
        return loss.item()

    @torch.no_grad()
    def sample(self, n: int = 1) -> torch.Tensor:
        x = torch.randn(n, self.dim, device=self.device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((n,), i, device=self.device, dtype=torch.long)
            beta = self._buf("betas")[i]
            alpha = self._buf("alphas")[i]
            ac = self._buf("alphas_cumprod")[i]
            eps = self.model(x, t)
            mean = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - ac)) * eps)
            if i > 0:
                x = mean + torch.sqrt(beta) * torch.randn_like(x)
                continue
            x = mean
        return x

    @torch.no_grad()
    def sample_refine(self, x0: torch.Tensor, start_t: int | None = None) -> torch.Tensor:
        """Reverse diffusion from q(x_start_t | x0) — img2img-style refinement of an init (e.g. JEPA)."""
        x0 = x0.to(self.device)
        n = x0.shape[0]
        start_t = self.timesteps - 1 if start_t is None else min(max(start_t, 0), self.timesteps - 1)
        t0 = torch.full((n,), start_t, device=self.device, dtype=torch.long)
        x = self.q_sample(x0, t0)
        for i in reversed(range(start_t + 1)):
            t = torch.full((n,), i, device=self.device, dtype=torch.long)
            beta = self._buf("betas")[i]
            alpha = self._buf("alphas")[i]
            ac = self._buf("alphas_cumprod")[i]
            eps = self.model(x, t)
            mean = (1 / torch.sqrt(alpha)) * (x - (beta / torch.sqrt(1 - ac)) * eps)
            if i > 0:
                x = mean + torch.sqrt(beta) * torch.randn_like(x)
                continue
            x = mean
        return x

    def save(self, path: str):
        torch.save({"model": self.model.state_dict(), "dim": self.dim, "timesteps": self.timesteps}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
