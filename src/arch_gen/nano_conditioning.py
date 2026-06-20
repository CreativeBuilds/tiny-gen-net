"""Nano architecture embedding + sub-chunk weight generators (1M–10M params)."""

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.arch_gen.nano_spec import N_EMBD_CHOICES, N_HEAD_CHOICES, N_LAYER_CHOICES, BLOCK_SIZE_CHOICES, REF_N_EMBD, REF_N_HEAD, REF_N_LAYER, NanoSpec
from src.models.nano_transformer import build_from_nano_spec
from src.utils.weights import load_flat_into_model, param_slices

MAX_SUBCHUNK = 32768
_LAYOUT_CACHE: dict[tuple[int, int, int, int, int], list[int]] = {}


def chunk_layout(spec: NanoSpec, vocab: int) -> list[int]:
    """Sub-chunk sizes for spec (cached — avoids building probe model every forward)."""
    key = (spec.n_embd, spec.n_layer, spec.n_head, spec.block_size, vocab)
    if key in _LAYOUT_CACHE: return _LAYOUT_CACHE[key]
    probe = build_from_nano_spec(spec, vocab)
    sizes: list[int] = []
    for sl in param_slices(probe):
        dim = sl["end"] - sl["start"]
        for off in range(0, dim, MAX_SUBCHUNK):
            sizes.append(min(MAX_SUBCHUNK, dim - off))
    _LAYOUT_CACHE[key] = sizes
    return sizes


def prewarm_chunk_layouts(specs: list[NanoSpec], vocab: int) -> None:
    for s in specs:
        chunk_layout(s, vocab)


class NanoEmbedder(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.e_emb = nn.Embedding(len(N_EMBD_CHOICES), dim // 2)
        self.l_emb = nn.Embedding(len(N_LAYER_CHOICES), dim // 4)
        self.h_emb = nn.Embedding(len(N_HEAD_CHOICES), dim // 8)
        self.b_emb = nn.Embedding(len(BLOCK_SIZE_CHOICES), dim // 8)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def _idx(self, spec: NanoSpec) -> tuple[int, int, int, int]:
        return (N_EMBD_CHOICES.index(spec.n_embd), N_LAYER_CHOICES.index(spec.n_layer),
                N_HEAD_CHOICES.index(spec.n_head), BLOCK_SIZE_CHOICES.index(spec.block_size))

    def forward(self, spec: NanoSpec, device: str = "cpu") -> torch.Tensor:
        ei, li, hi, bi = self._idx(spec)
        x = torch.cat([self.e_emb(torch.tensor([ei], device=device)), self.l_emb(torch.tensor([li], device=device)),
                       self.h_emb(torch.tensor([hi], device=device)), self.b_emb(torch.tensor([bi], device=device))], dim=-1)
        return self.proj(x).squeeze(0)


class SubChunkDecoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, MAX_SUBCHUNK))

    def forward(self, h: torch.Tensor, n: int) -> torch.Tensor:
        return self.net(h)[:n]


class NanoWeightGenCore(nn.Module):
    """Shared sub-chunk decode: cond vector -> full flat weight for any NanoSpec."""

    def __init__(self, cond_dim: int, latent_dim: int = 64, hidden: int = 256):
        super().__init__()
        self.latent_dim = latent_dim
        self.trunk = nn.Sequential(nn.Linear(cond_dim + latent_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.sub_idx = nn.Embedding(2048, hidden // 4)
        self.sub_dec = SubChunkDecoder(hidden + hidden // 4, hidden)

    def decode_flat(self, cond: torch.Tensor, spec: NanoSpec, vocab: int, z: torch.Tensor) -> torch.Tensor:
        h0 = self.trunk(torch.cat([cond, z], dim=-1))
        sizes = chunk_layout(spec, vocab)
        if not sizes: return torch.empty(0, device=h0.device)
        dev = h0.device
        idx = torch.arange(len(sizes), device=dev) % 2048
        idx_emb = self.sub_idx(idx)
        h_exp = h0.unsqueeze(0).expand(len(sizes), -1)
        h_in = torch.cat([h_exp, idx_emb], dim=-1)
        return torch.cat([self.sub_dec(h_in[i], sizes[i]) for i in range(len(sizes))])


def autocast_ctx(device: str):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if str(device).startswith("cuda") else nullcontext()


class NanoCondWeightGenerator(NanoWeightGenCore):
    def __init__(self, vocab: int = 65, embed_dim: int = 128, latent_dim: int = 64, hidden: int = 256):
        super().__init__(embed_dim, latent_dim, hidden)
        self.vocab = vocab
        self.embed = NanoEmbedder(embed_dim)

    def forward_flat(self, spec: NanoSpec, z: torch.Tensor, device: str) -> torch.Tensor:
        return self.decode_flat(self.embed(spec, device), spec, self.vocab, z)

    @torch.no_grad()
    def sample(self, spec: NanoSpec, device: str = "cpu", noise_scale: float = 1.0) -> torch.Tensor:
        self.eval()
        z = torch.randn(self.latent_dim, device=device) * noise_scale
        return self.forward_flat(spec, z, device)


class NanoCondWeightTrainer:
    def __init__(self, model: NanoCondWeightGenerator, device: str = "cpu", lr: float = 1e-3):
        self.model = model.to(device)
        self.device = device
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def step_batch(self, specs: list[NanoSpec], targets: list[torch.Tensor]) -> float:
        self.model.train()
        loss = 0.0
        with autocast_ctx(self.device):
            for spec, tgt in zip(specs, targets):
                z = torch.randn(self.model.latent_dim, device=self.device)
                loss = loss + F.mse_loss(self.model.forward_flat(spec, z, self.device), tgt.to(self.device))
        loss = loss / len(specs)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return loss.item()

    def fit(self, specs: list[NanoSpec], weights: list[torch.Tensor], steps: int = 2000, batch_size: int = 4, ckpt_path: str | None = None, norm_meta: dict | None = None) -> list[float]:
        prewarm_chunk_layouts(specs, self.model.vocab)
        n, losses = len(specs), []
        for i in tqdm(range(steps), desc="cond", leave=False):
            idx = torch.randint(0, n, (min(batch_size, n),)).tolist()
            losses.append(self.step_batch([specs[i] for i in idx], [weights[i] for i in idx]))
            if ckpt_path and norm_meta and (i + 1) % 200 == 0:
                save_nano_cond_ckpt(ckpt_path, self.model, norm_meta)
        return losses


def normalize_nano_weights(weights: list[torch.Tensor]) -> tuple[list[torch.Tensor], dict]:
    flat = torch.cat(weights)
    mean, std = flat.mean(), flat.std().clamp_min(1e-6)
    return [(w - mean) / std for w in weights], {"mean": mean, "std": std, "norm_mode": "global"}


def denorm_nano(flat: torch.Tensor, meta: dict) -> torch.Tensor:
    return flat * meta["std"] + meta["mean"]


@torch.no_grad()
def sample_denorm_nano(model: NanoCondWeightGenerator, spec: NanoSpec, meta: dict, device: str, noise_scale: float = 0.5) -> torch.Tensor:
    return denorm_nano(model.sample(spec, device, noise_scale), meta)


def save_nano_cond_ckpt(path: str, model: NanoCondWeightGenerator, norm_meta: dict):
    torch.save({"state_dict": model.state_dict(), "norm_meta": norm_meta, "vocab": model.vocab}, path)


def load_nano_cond_ckpt(path: str, device: str) -> tuple[NanoCondWeightGenerator, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = NanoCondWeightGenerator(vocab=ckpt.get("vocab", 65)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), ckpt["norm_meta"]


@torch.no_grad()
def init_nano_weights(model, spec: NanoSpec, wgen: NanoCondWeightGenerator, norm_meta: dict, device: str, noise_scale: float = 0.5):
    load_flat_into_model(sample_denorm_nano(wgen, spec, norm_meta, device, noise_scale), model)
