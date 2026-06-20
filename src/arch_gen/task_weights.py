"""Task-conditioned weight generator: arch + task text -> flat weights."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.arch_gen.conditioning import ArchEmbedder, MAX_PARAMS, denorm_multi_arch, normalize_multi_arch
from src.arch_gen.spec import ArchSpec
from src.arch_gen.task_cond import TASK_DIM, ArchMatchHead, TaskEmbedder
from src.models.variable_mlp import build_from_spec
from src.utils.weights import load_flat_into_model


class TaskCondWeightGenerator(nn.Module):
    """ctx = arch_emb + task_emb + latent -> hypernet head."""

    def __init__(self, task_dim: int = TASK_DIM, embed_dim: int = 64, latent_dim: int = 32, hidden: int = 256):
        super().__init__()
        self.task_enc = TaskEmbedder(task_dim)
        self.match_head = ArchMatchHead(task_dim)
        self.embed = ArchEmbedder(embed_dim)
        self.latent_dim, self.task_dim = latent_dim, task_dim
        d_in = embed_dim + latent_dim + task_dim
        self.trunk = nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.head = nn.Linear(hidden, MAX_PARAMS)

    def _ctx(self, spec: ArchSpec, task_text: str, device: str, z: torch.Tensor | None = None) -> torch.Tensor:
        te = self.task_enc.encode_text(task_text, device)
        ae = self.embed(spec, device)
        if z is None: z = torch.randn(self.latent_dim, device=device)
        return self.trunk(torch.cat([ae, te, z], dim=-1))

    def forward_flat(self, spec: ArchSpec, task_text: str, z: torch.Tensor, device: str) -> torch.Tensor:
        n = build_from_spec(spec).num_parameters()
        return self.head(self._ctx(spec, task_text, device, z))[:n]

    @torch.no_grad()
    def sample(self, spec: ArchSpec, task_text: str, device: str = "cpu", noise_scale: float = 0.5) -> torch.Tensor:
        self.eval()
        z = torch.randn(self.latent_dim, device=device) * noise_scale
        return self.forward_flat(spec, task_text, z, device)


class TaskCondWeightTrainer:
    def __init__(self, model: TaskCondWeightGenerator, device: str = "cpu", lr: float = 1e-3, match_weight: float = 0.3):
        self.model = model.to(device)
        self.device, self.match_weight = device, match_weight
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def step_batch(self, specs: list[ArchSpec], texts: list[str], targets: list[torch.Tensor]) -> float:
        self.model.train()
        loss, tvecs = 0.0, []
        for spec, text, tgt in zip(specs, texts, targets):
            z = torch.randn(self.model.latent_dim, device=self.device)
            pred = self.model.forward_flat(spec, text, z, self.device)
            loss = loss + F.mse_loss(pred, tgt.to(self.device))
            tvecs.append(self.model.task_enc.encode_text(text, self.device))
        loss = loss / len(specs)
        if self.match_weight > 0:
            tv = torch.stack(tvecs)
            loss = loss + self.match_weight * self.model.match_head.loss(tv, specs)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return loss.item()

    def fit(self, specs: list[ArchSpec], texts: list[str], weights: list[torch.Tensor], steps: int = 2000, batch_size: int = 16) -> list[float]:
        n, losses = len(specs), []
        for _ in range(steps):
            idx = torch.randint(0, n, (min(batch_size, n),)).tolist()
            losses.append(self.step_batch([specs[i] for i in idx], [texts[i] for i in idx], [weights[i] for i in idx]))
        return losses


@torch.no_grad()
def sample_denorm_task(model: TaskCondWeightGenerator, spec: ArchSpec, task_text: str, meta: dict, device: str, noise_scale: float = 0.5) -> torch.Tensor:
    return denorm_multi_arch(model.sample(spec, task_text, device, noise_scale), meta)


def save_task_cond_ckpt(path: str, arch_gen, weight_gen, norm_meta: dict):
    torch.save({"arch": arch_gen.state_dict(), "weights": weight_gen.state_dict(), "norm_meta": norm_meta}, path)


def load_task_cond_ckpt(path: str, device: str):
    from src.arch_gen.task_generator import TaskArchGenerator
    ckpt = torch.load(path, map_location=device, weights_only=True)
    arch = TaskArchGenerator().to(device)
    wgen = TaskCondWeightGenerator().to(device)
    arch.load_state_dict(ckpt["arch"])
    wgen.load_state_dict(ckpt["weights"])
    return arch.eval(), wgen.eval(), ckpt["norm_meta"]
