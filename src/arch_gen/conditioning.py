"""Architecture embedding + conditioned weight generator (hypernet-style)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.synthetic_text import VOCAB_SIZE
from src.arch_gen.spec import DEPTH_CHOICES, HIDDEN_CHOICES, ArchSpec, REF_DEPTH, REF_HIDDEN, all_valid_specs
from src.models.variable_mlp import build_from_spec
from src.utils.weights import load_flat_into_model

def _max_params() -> int:
    return max(build_from_spec(s).num_parameters() for s in all_valid_specs())

MAX_PARAMS = _max_params()


class ArchEmbedder(nn.Module):
    """Embed discrete ArchSpec (hidden, depth, skip) -> conditioning vector."""

    def __init__(self, dim: int = 64):
        super().__init__()
        self.h_emb = nn.Embedding(len(HIDDEN_CHOICES), dim // 2)
        self.d_emb = nn.Embedding(len(DEPTH_CHOICES), dim // 4)
        self.s_emb = nn.Embedding(2, dim // 4)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def _idx(self, spec: ArchSpec) -> tuple[int, int, int]:
        return HIDDEN_CHOICES.index(spec.hidden_dim), DEPTH_CHOICES.index(spec.depth), int(spec.skip)

    def forward(self, spec: ArchSpec, device: str = "cpu") -> torch.Tensor:
        hi, di, si = self._idx(spec)
        x = torch.cat([self.h_emb(torch.tensor([hi], device=device)), self.d_emb(torch.tensor([di], device=device)),
                       self.s_emb(torch.tensor([si], device=device))], dim=-1)
        return self.proj(x).squeeze(0)

    def batch(self, specs: list[ArchSpec], device: str) -> torch.Tensor:
        hi = torch.tensor([self._idx(s)[0] for s in specs], device=device)
        di = torch.tensor([self._idx(s)[1] for s in specs], device=device)
        si = torch.tensor([self._idx(s)[2] for s in specs], device=device)
        x = torch.cat([self.h_emb(hi), self.d_emb(di), self.s_emb(si)], dim=-1)
        return self.proj(x)


def ref_similarity(spec: ArchSpec) -> float:
    h_span = max(HIDDEN_CHOICES) - min(HIDDEN_CHOICES)
    h_sim = 1.0 - abs(spec.hidden_dim - REF_HIDDEN) / h_span
    d_sim = 1.0 - abs(spec.depth - REF_DEPTH) / max(DEPTH_CHOICES)
    s_sim = 1.0 if not spec.skip else 0.5
    return 0.5 * h_sim + 0.3 * d_sim + 0.2 * s_sim


class CondWeightGenerator(nn.Module):
    """Sample flat weights for any ArchSpec: ctx = arch_emb + latent -> universal head -> trim."""

    def __init__(self, embed_dim: int = 64, latent_dim: int = 32, hidden: int = 256):
        super().__init__()
        self.embed = ArchEmbedder(embed_dim)
        self.latent_dim = latent_dim
        self.trunk = nn.Sequential(nn.Linear(embed_dim + latent_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.head = nn.Linear(hidden, MAX_PARAMS)

    def _ctx(self, spec: ArchSpec, device: str, z: torch.Tensor | None = None) -> torch.Tensor:
        e = self.embed(spec, device)
        if z is None: z = torch.randn(self.latent_dim, device=device)
        return self.trunk(torch.cat([e, z], dim=-1))

    def forward_flat(self, spec: ArchSpec, z: torch.Tensor, device: str) -> torch.Tensor:
        n = build_from_spec(spec).num_parameters()
        return self.head(self._ctx(spec, device, z))[:n]

    @torch.no_grad()
    def sample(self, spec: ArchSpec, device: str = "cpu", noise_scale: float = 1.0) -> torch.Tensor:
        self.eval()
        z = torch.randn(self.latent_dim, device=device) * noise_scale
        return self.forward_flat(spec, z, device)

    @torch.no_grad()
    def load_into(self, model: nn.Module, spec: ArchSpec, device: str = "cpu") -> torch.Tensor:
        flat = self.sample(spec, device)
        load_flat_into_model(flat, model)
        return flat


class CondWeightTrainer:
    def __init__(self, model: CondWeightGenerator, device: str = "cpu", lr: float = 1e-3):
        self.model = model.to(device)
        self.device = device
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def step_batch(self, specs: list[ArchSpec], targets: list[torch.Tensor]) -> float:
        self.model.train()
        loss = 0.0
        for spec, tgt in zip(specs, targets):
            z = torch.randn(self.model.latent_dim, device=self.device)
            pred = self.model.forward_flat(spec, z, self.device)
            loss = loss + F.mse_loss(pred, tgt.to(self.device))
        loss = loss / len(specs)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return loss.item()

    def fit(self, specs: list[ArchSpec], weights: list[torch.Tensor], steps: int = 2000, batch_size: int = 16) -> list[float]:
        n = len(specs)
        losses: list[float] = []
        for _ in range(steps):
            idx = torch.randint(0, n, (min(batch_size, n),)).tolist()
            losses.append(self.step_batch([specs[i] for i in idx], [weights[i] for i in idx]))
        return losses


def normalize_multi_arch(weights: list[torch.Tensor]) -> tuple[list[torch.Tensor], dict]:
    flat = torch.cat(weights)
    mean, std = flat.mean(), flat.std().clamp_min(1e-6)
    return [(w - mean) / std for w in weights], {"mean": mean, "std": std, "norm_mode": "global"}

def denorm_multi_arch(flat: torch.Tensor, meta: dict) -> torch.Tensor:
    return flat * meta["std"] + meta["mean"]

@torch.no_grad()
def sample_denorm(model: CondWeightGenerator, spec: ArchSpec, meta: dict, device: str, noise_scale: float = 1.0) -> torch.Tensor:
    return denorm_multi_arch(model.sample(spec, device, noise_scale=noise_scale), meta)

def save_cond_checkpoint(path: str, model: CondWeightGenerator, norm_meta: dict):
    torch.save({"state_dict": model.state_dict(), "norm_meta": norm_meta}, path)

def load_cond_checkpoint(path: str, device: str) -> tuple[CondWeightGenerator, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = CondWeightGenerator()
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["norm_meta"]
