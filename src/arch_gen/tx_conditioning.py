"""Transformer architecture embedding + conditioned weight generator."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.arch_gen.tx_spec import D_MODEL_CHOICES, N_HEAD_CHOICES, N_LAYER_CHOICES, REF_D_MODEL, REF_N_HEAD, REF_N_LAYER, TxSpec, all_valid_tx_specs
from src.models.variable_transformer import build_from_tx_spec
from src.utils.weights import load_flat_into_model


def _max_tx_params() -> int:
    return max(build_from_tx_spec(s).num_parameters() for s in all_valid_tx_specs())

MAX_TX_PARAMS = _max_tx_params()


class TxEmbedder(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.d_emb = nn.Embedding(len(D_MODEL_CHOICES), dim // 2)
        self.l_emb = nn.Embedding(len(N_LAYER_CHOICES), dim // 4)
        self.h_emb = nn.Embedding(len(N_HEAD_CHOICES), dim // 4)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def _idx(self, spec: TxSpec) -> tuple[int, int, int]:
        return D_MODEL_CHOICES.index(spec.d_model), N_LAYER_CHOICES.index(spec.n_layer), N_HEAD_CHOICES.index(spec.n_head)

    def forward(self, spec: TxSpec, device: str = "cpu") -> torch.Tensor:
        di, li, hi = self._idx(spec)
        x = torch.cat([self.d_emb(torch.tensor([di], device=device)), self.l_emb(torch.tensor([li], device=device)),
                       self.h_emb(torch.tensor([hi], device=device))], dim=-1)
        return self.proj(x).squeeze(0)


def tx_ref_similarity(spec: TxSpec) -> float:
    d_span = max(D_MODEL_CHOICES) - min(D_MODEL_CHOICES)
    d_sim = 1.0 - abs(spec.d_model - REF_D_MODEL) / d_span
    l_sim = 1.0 - abs(spec.n_layer - REF_N_LAYER) / max(N_LAYER_CHOICES)
    h_sim = 1.0 - abs(spec.n_head - REF_N_HEAD) / max(N_HEAD_CHOICES)
    return 0.5 * d_sim + 0.3 * l_sim + 0.2 * h_sim


class TxCondWeightGenerator(nn.Module):
    def __init__(self, embed_dim: int = 64, latent_dim: int = 32, hidden: int = 256):
        super().__init__()
        self.embed = TxEmbedder(embed_dim)
        self.latent_dim = latent_dim
        self.trunk = nn.Sequential(nn.Linear(embed_dim + latent_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.head = nn.Linear(hidden, MAX_TX_PARAMS)

    def forward_flat(self, spec: TxSpec, z: torch.Tensor, device: str) -> torch.Tensor:
        e = self.embed(spec, device)
        return self.head(self.trunk(torch.cat([e, z], dim=-1)))[: build_from_tx_spec(spec).num_parameters()]

    @torch.no_grad()
    def sample(self, spec: TxSpec, device: str = "cpu", noise_scale: float = 1.0) -> torch.Tensor:
        self.eval()
        z = torch.randn(self.latent_dim, device=device) * noise_scale
        return self.forward_flat(spec, z, device)


class TxCondWeightTrainer:
    def __init__(self, model: TxCondWeightGenerator, device: str = "cpu", lr: float = 1e-3):
        self.model = model.to(device)
        self.device = device
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def step_batch(self, specs: list[TxSpec], targets: list[torch.Tensor]) -> float:
        self.model.train()
        loss = 0.0
        for spec, tgt in zip(specs, targets):
            z = torch.randn(self.model.latent_dim, device=self.device)
            loss = loss + F.mse_loss(self.model.forward_flat(spec, z, self.device), tgt.to(self.device))
        loss = loss / len(specs)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return loss.item()

    def fit(self, specs: list[TxSpec], weights: list[torch.Tensor], steps: int = 2000, batch_size: int = 8) -> list[float]:
        n, losses = len(specs), []
        for _ in range(steps):
            idx = torch.randint(0, n, (min(batch_size, n),)).tolist()
            losses.append(self.step_batch([specs[i] for i in idx], [weights[i] for i in idx]))
        return losses


def normalize_tx_weights(weights: list[torch.Tensor]) -> tuple[list[torch.Tensor], dict]:
    flat = torch.cat(weights)
    mean, std = flat.mean(), flat.std().clamp_min(1e-6)
    return [(w - mean) / std for w in weights], {"mean": mean, "std": std, "norm_mode": "global"}


def denorm_tx(flat: torch.Tensor, meta: dict) -> torch.Tensor:
    return flat * meta["std"] + meta["mean"]


@torch.no_grad()
def sample_denorm_tx(model: TxCondWeightGenerator, spec: TxSpec, meta: dict, device: str, noise_scale: float = 0.5) -> torch.Tensor:
    return denorm_tx(model.sample(spec, device, noise_scale), meta)


def save_tx_cond_ckpt(path: str, model: TxCondWeightGenerator, norm_meta: dict):
    torch.save({"state_dict": model.state_dict(), "norm_meta": norm_meta}, path)


def load_tx_cond_ckpt(path: str, device: str) -> tuple[TxCondWeightGenerator, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = TxCondWeightGenerator().to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt["norm_meta"]


@torch.no_grad()
def init_tx_weights(model, spec: TxSpec, wgen: TxCondWeightGenerator, norm_meta: dict, device: str, noise_scale: float = 0.5):
    load_flat_into_model(sample_denorm_tx(wgen, spec, norm_meta, device, noise_scale), model)
