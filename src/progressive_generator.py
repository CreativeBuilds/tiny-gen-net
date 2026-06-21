"""Progressive layer-by-layer weight generator with scale-aware conditioning."""

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.arch_gen.nano_spec import NanoSpec
from src.arch_gen.sentence_embedder import SENT_DIM, SentenceTaskEmbedder
from src.hierarchical_jepa import HighLevelScaleJEPA, LATENT_DIM
from src.nano_layers import layer_dims, logical_layer_bounds
from src.scale_embedding import SCALE_DIM, scale_features
from src.utils.weights import load_flat_into_model

MAX_LAYER_DEC = 65536
SUB = 8192


def _batch2(t: torch.Tensor) -> torch.Tensor:
    return t if t.dim() >= 2 else t.unsqueeze(0)


class LayerWeightDecoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, SUB))

    def forward(self, h: torch.Tensor, n: int) -> torch.Tensor:
        return self.net(h)[:n]


class ProgressiveWeightGenerator(nn.Module):
    """Mid+low: global plan h → block hints → layer decode; optional strong scale+frac+plan residual."""

    def __init__(self, vocab: int = 65, task_dim: int = SENT_DIM, latent_dim: int = LATENT_DIM, hidden: int = 256,
                 plan_scale: float = 0.35, strong_plan: bool = False):
        super().__init__()
        self.vocab, self.latent_dim, self.plan_scale, self.strong_plan = vocab, latent_dim, plan_scale, strong_plan
        self.high = HighLevelScaleJEPA(task_dim=task_dim, latent_dim=latent_dim, hidden=hidden)
        bp_in = latent_dim * 4 if strong_plan else latent_dim * 2
        self.block_plan = nn.Sequential(nn.Linear(bp_in, latent_dim), nn.SiLU(), nn.Linear(latent_dim, latent_dim))
        gru_in = latent_dim * 5 if strong_plan else latent_dim * 3
        self.mid = nn.GRU(gru_in, latent_dim, batch_first=True)
        dec_in = latent_dim * 5 if strong_plan else latent_dim * 3
        self.dec = LayerWeightDecoder(dec_in, hidden)
        if strong_plan:
            self.scale_proj = nn.Linear(SCALE_DIM, latent_dim)
            self.frac_emb = nn.Linear(1, latent_dim)
            self.plan_mix = nn.Sequential(nn.Linear(latent_dim * 2, latent_dim), nn.SiLU(), nn.Linear(latent_dim, latent_dim))

    def _scale_frac(self, scale: torch.Tensor, i: int, n: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        sv = scale if scale.dim() == 1 else scale.squeeze(0)
        se = self.scale_proj(sv.to(device))
        fe = self.frac_emb(torch.tensor([i / max(n - 1, 1)], device=device, dtype=sv.dtype))
        return se, fe

    def _block_hint(self, h: torch.Tensor, pos: torch.Tensor, se: torch.Tensor | None = None, fe: torch.Tensor | None = None) -> torch.Tensor:
        if se is None: return self.block_plan(torch.cat([h, pos], dim=-1))
        return self.block_plan(torch.cat([h, pos, se, fe], dim=-1))

    def _plan_z(self, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        if not self.strong_plan: return z + self.plan_scale * h
        return z + self.plan_scale * self.plan_mix(torch.cat([z, h], dim=-1))

    def _decode_layer(self, z: torch.Tensor, h: torch.Tensor, dim: int, se: torch.Tensor | None = None, fe: torch.Tensor | None = None) -> torch.Tensor:
        parts, si = [], 0
        for off in range(0, dim, SUB):
            n = min(SUB, dim - off)
            idx = torch.tensor([si % 32], device=z.device, dtype=torch.long)
            pos = self.high.layer_pos(idx).squeeze(0)
            ctx = torch.cat([z, pos, h, se, fe], dim=-1) if se is not None else torch.cat([z, pos, h], dim=-1)
            parts.append(self.dec(ctx, n))
            si += 1
        return torch.cat(parts)

    def _layer_step(self, h: torch.Tensor, scale: torch.Tensor, i: int, n: int, prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dev = h.device
        idx = torch.tensor([min(i, 31)], device=self.high.layer_pos.weight.device)
        pos = self.high.layer_pos(idx).squeeze(0)
        se, fe = (None, None)
        if self.strong_plan: se, fe = self._scale_frac(scale, i, n, dev)
        bh = self._block_hint(h, pos, se, fe)
        if se is None:
            inp = torch.stack([torch.cat([h, pos, bh]), torch.cat([prev, pos, bh])]).unsqueeze(0)
        else:
            inp = torch.stack([torch.cat([h, pos, bh, se, fe]), torch.cat([prev, pos, bh, se, fe])]).unsqueeze(0)
        out, _ = self.mid(inp)
        z = self._plan_z(out[0, -1], h)
        return z, (se, fe)

    def forward_progressive(self, task: torch.Tensor, scale: torch.Tensor, layer_targets: list[torch.Tensor]) -> tuple[list[torch.Tensor], torch.Tensor]:
        task, scale = _batch2(task), _batch2(scale)
        h = self.high.global_plan(task, scale)
        h_vec = h.squeeze(0) if h.shape[0] == 1 else h[0]
        scale_v = scale.squeeze(0) if scale.dim() > 1 else scale
        prev = torch.zeros(self.latent_dim, device=h_vec.device)
        preds, n = [], len(layer_targets)
        for i, tgt in enumerate(layer_targets):
            z, (se, fe) = self._layer_step(h_vec, scale_v, i, n, prev)
            preds.append(self._decode_layer(z, h_vec, tgt.numel(), se, fe))
            prev = z.detach()
        return preds, h

    @torch.no_grad()
    def generate(self, task: torch.Tensor, spec: NanoSpec, device: str) -> torch.Tensor:
        self.eval()
        scale = scale_features(spec, self.vocab).to(device)
        task_b, scale_b = _batch2(task.to(device)), _batch2(scale)
        h = self.high.sample_plan(task_b, scale_b, device).squeeze(0)
        prev = torch.zeros(self.latent_dim, device=device)
        dims, parts = layer_dims(spec, self.vocab), []
        n = len(dims)
        for i, dim in enumerate(dims):
            z, (se, fe) = self._layer_step(h, scale, i, n, prev)
            parts.append(self._decode_layer(z, h, dim, se, fe))
            prev = z
        return torch.cat(parts)


def autocast_ctx(device: str):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if str(device).startswith("cuda") else nullcontext()


class ProgressiveWeightTrainer:
    def __init__(self, model: ProgressiveWeightGenerator, device: str = "cpu", lr: float = 1e-3):
        self.model, self.device = model.to(device), device
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def step(self, task: torch.Tensor, scale: torch.Tensor, layer_targets: list[torch.Tensor]) -> dict[str, float]:
        self.model.train()
        with autocast_ctx(self.device):
            preds, h = self.model.forward_progressive(task, scale, layer_targets)
            prog = sum(F.mse_loss(p, t.to(self.device)) for p, t in zip(preds, layer_targets)) / len(preds)
            loss = prog
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.model.high.update_target(0.99)
        return {"total": loss.item(), "prog": prog.item()}

    def fit(self, tasks: list[torch.Tensor], scales: list[torch.Tensor], layer_batches: list[list[torch.Tensor]], steps: int = 800) -> list[float]:
        n, losses = len(tasks), []
        for _ in tqdm(range(steps), desc="prog", leave=False):
            i = torch.randint(0, n, (1,)).item()
            losses.append(self.step(tasks[i].unsqueeze(0), scales[i].unsqueeze(0), layer_batches[i])["total"])
        return losses


class ScaleAwareProgressiveTrainer(ProgressiveWeightTrainer):
    def __init__(self, model: ProgressiveWeightGenerator, device: str = "cpu", lr: float = 1e-3, jepa_w: float = 0.2, cons_w: float = 0.5):
        super().__init__(model, device, lr)
        self.jepa_w, self.cons_w = jepa_w, cons_w

    def step(self, task: torch.Tensor, scale: torch.Tensor, layer_targets: list[torch.Tensor],
             cons_task: torch.Tensor | None = None, cons_scale: torch.Tensor | None = None) -> dict[str, float]:
        self.model.train()
        with autocast_ctx(self.device):
            preds, _ = self.model.forward_progressive(task, scale, layer_targets)
            prog = sum(F.mse_loss(p, t.to(self.device)) for p, t in zip(preds, layer_targets)) / len(preds)
            jepa = self.model.high.train_step(_batch2(task), _batch2(scale), layer_targets)
            loss = prog + self.jepa_w * jepa["total"]
            cons_v = torch.tensor(0.0, device=self.device)
            if cons_task is not None and cons_scale is not None:
                cons_v = self.model.high.scale_consistency(cons_task, _batch2(scale).squeeze(0), cons_scale)
                loss = loss + self.cons_w * cons_v
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        self.model.high.update_target(0.99)
        return {"total": loss.item(), "prog": prog.item(), "jepa": jepa["jepa"].item() if isinstance(jepa["jepa"], torch.Tensor) else jepa["jepa"], "cons": cons_v.item()}

    def fit(self, tasks: list[torch.Tensor], scales: list[torch.Tensor], layer_batches: list[list[torch.Tensor]],
            cons_tasks: list[torch.Tensor], steps: int = 800) -> list[dict[str, float]]:
        n, losses = len(tasks), []
        for _ in tqdm(range(steps), desc="5c", leave=False):
            i = torch.randint(0, n, (1,)).item()
            j = torch.randint(0, n, (1,)).item()
            while j == i and n > 1: j = torch.randint(0, n, (1,)).item()
            ct = cons_tasks[torch.randint(0, len(cons_tasks), (1,)).item()]
            losses.append(self.step(tasks[i].unsqueeze(0), scales[i].unsqueeze(0), layer_batches[i], ct, scales[j]))
        return losses


def split_layers(flat: torch.Tensor, spec: NanoSpec, vocab: int) -> list[torch.Tensor]:
    bounds = logical_layer_bounds(spec, vocab)
    return [flat[s:e].clone() for s, e in bounds]


def init_progressive_weights(model: nn.Module, spec: NanoSpec, gen: ProgressiveWeightGenerator, task_text: str, enc: SentenceTaskEmbedder, device: str):
    task = enc.encode_text(task_text, device)
    flat = gen.generate(task, spec, device)
    load_flat_into_model(flat, model)
