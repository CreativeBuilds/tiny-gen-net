"""Task-conditioned transformer weight generator."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.arch_gen.task_cond import TASK_DIM, TaskEmbedder
from src.arch_gen.tx_conditioning import MAX_TX_PARAMS, TxEmbedder, denorm_tx
from src.arch_gen.tx_spec import TxSpec
from src.arch_gen.tx_task_cond import TxMatchHead, enrich_tx_task
from src.models.variable_transformer import build_from_tx_spec
from src.utils.weights import load_flat_into_model


class TaskTxWeightGenerator(nn.Module):
    def __init__(self, task_dim: int = TASK_DIM, embed_dim: int = 64, latent_dim: int = 32, hidden: int = 256):
        super().__init__()
        self.task_enc = TaskEmbedder(task_dim)
        self.match_head = TxMatchHead(task_dim)
        self.embed = TxEmbedder(embed_dim)
        self.latent_dim = latent_dim
        d_in = embed_dim + latent_dim + task_dim
        self.trunk = nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.head = nn.Linear(hidden, MAX_TX_PARAMS)

    def forward_flat(self, spec: TxSpec, task_text: str, z: torch.Tensor, device: str) -> torch.Tensor:
        te = self.task_enc.encode_text(enrich_tx_task(task_text), device)
        ae = self.embed(spec, device)
        return self.head(self.trunk(torch.cat([ae, te, z], dim=-1)))[: build_from_tx_spec(spec).num_parameters()]

    @torch.no_grad()
    def sample(self, spec: TxSpec, task_text: str, device: str, noise_scale: float = 0.5) -> torch.Tensor:
        self.eval()
        z = torch.randn(self.latent_dim, device=device) * noise_scale
        return self.forward_flat(spec, task_text, z, device)


class TaskTxWeightTrainer:
    def __init__(self, model: TaskTxWeightGenerator, device: str = "cpu", lr: float = 1e-3, match_weight: float = 0.3):
        self.model = model.to(device)
        self.device, self.match_weight = device, match_weight
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def step_batch(self, specs: list[TxSpec], texts: list[str], targets: list[torch.Tensor]) -> float:
        self.model.train()
        loss, tvecs = 0.0, []
        for spec, text, tgt in zip(specs, texts, targets):
            z = torch.randn(self.model.latent_dim, device=self.device)
            loss = loss + F.mse_loss(self.model.forward_flat(spec, text, z, self.device), tgt.to(self.device))
            tvecs.append(self.model.task_enc.encode_text(enrich_tx_task(text), self.device))
        loss = loss / len(specs)
        if self.match_weight > 0:
            loss = loss + self.match_weight * self.model.match_head.loss(torch.stack(tvecs), specs)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return loss.item()

    def fit(self, specs: list[TxSpec], texts: list[str], weights: list[torch.Tensor], steps: int = 1500, batch_size: int = 8) -> list[float]:
        n, losses = len(specs), []
        for _ in range(steps):
            idx = torch.randint(0, n, (min(batch_size, n),)).tolist()
            losses.append(self.step_batch([specs[i] for i in idx], [texts[i] for i in idx], [weights[i] for i in idx]))
        return losses


@torch.no_grad()
def sample_denorm_task_tx(model: TaskTxWeightGenerator, spec: TxSpec, task_text: str, meta: dict, device: str, noise_scale: float = 0.5) -> torch.Tensor:
    return denorm_tx(model.sample(spec, task_text, device, noise_scale), meta)


def save_task_tx_ckpt(path: str, arch_gen, weight_gen, norm_meta: dict):
    torch.save({"arch": arch_gen.state_dict(), "weights": weight_gen.state_dict(), "norm_meta": norm_meta}, path)


def load_task_tx_ckpt(path: str, device: str):
    from src.arch_gen.tx_task_generator import TaskTxArchGenerator
    ckpt = torch.load(path, map_location=device, weights_only=True)
    arch = TaskTxArchGenerator().to(device)
    wgen = TaskTxWeightGenerator().to(device)
    arch.load_state_dict(ckpt["arch"])
    wgen.load_state_dict(ckpt["weights"])
    return arch.eval(), wgen.eval(), ckpt["norm_meta"]


@torch.no_grad()
def init_task_tx_weights(model, spec: TxSpec, wgen: TaskTxWeightGenerator, task_text: str, norm_meta: dict, device: str, noise_scale: float = 0.5):
    load_flat_into_model(sample_denorm_task_tx(wgen, spec, task_text, norm_meta, device, noise_scale), model)
