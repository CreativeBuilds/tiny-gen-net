"""Task-conditioned nano weight generator (sentence embeddings + sub-chunk decode)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.arch_gen.nano_conditioning import NanoEmbedder, NanoWeightGenCore, autocast_ctx, denorm_nano, prewarm_chunk_layouts
from src.arch_gen.nano_spec import NanoSpec
from src.arch_gen.nano_task_cond import NanoMatchHead, normalize_nano_task
from src.arch_gen.sentence_embedder import SENT_DIM, SentenceTaskEmbedder
from src.utils.weights import load_flat_into_model


class TaskNanoWeightGenerator(NanoWeightGenCore):
    def __init__(self, vocab: int = 65, task_dim: int = SENT_DIM, embed_dim: int = 128, latent_dim: int = 64, hidden: int = 256):
        super().__init__(embed_dim + task_dim, latent_dim, hidden)
        self.vocab = vocab
        self.task_enc = SentenceTaskEmbedder()
        self.match_head = NanoMatchHead(task_dim)
        self.embed = NanoEmbedder(embed_dim)

    def forward_flat(self, spec: NanoSpec, task_text: str, z: torch.Tensor, device: str) -> torch.Tensor:
        te = self.task_enc.encode_text(normalize_nano_task(task_text), device)
        ae = self.embed(spec, device)
        return self.decode_flat(torch.cat([ae, te], dim=-1), spec, self.vocab, z)

    @torch.no_grad()
    def sample(self, spec: NanoSpec, task_text: str, device: str, noise_scale: float = 0.5) -> torch.Tensor:
        self.eval()
        z = torch.randn(self.latent_dim, device=device) * noise_scale
        return self.forward_flat(spec, task_text, z, device)


class TaskNanoWeightTrainer:
    def __init__(self, model: TaskNanoWeightGenerator, device: str = "cpu", lr: float = 1e-3, match_weight: float = 0.3):
        self.model = model.to(device)
        self.device, self.match_weight = device, match_weight
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def step_batch(self, specs: list[NanoSpec], texts: list[str], targets: list[torch.Tensor]) -> float:
        self.model.train()
        loss, tvecs = 0.0, []
        with autocast_ctx(self.device):
            for spec, text, tgt in zip(specs, texts, targets):
                z = torch.randn(self.model.latent_dim, device=self.device)
                loss = loss + F.mse_loss(self.model.forward_flat(spec, text, z, self.device), tgt.to(self.device))
                tvecs.append(self.model.task_enc.encode_text(normalize_nano_task(text), self.device))
        loss = loss / len(specs)
        if self.match_weight > 0:
            loss = loss + self.match_weight * self.model.match_head.loss(torch.stack(tvecs), specs)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        return loss.item()

    def fit(self, specs: list[NanoSpec], texts: list[str], weights: list[torch.Tensor], steps: int = 1500, batch_size: int = 4, ckpt_path: str | None = None, arch_gen=None, norm_meta: dict | None = None) -> list[float]:
        prewarm_chunk_layouts(specs, self.model.vocab)
        n, losses = len(specs), []
        for i in tqdm(range(steps), desc="task_w", leave=False):
            idx = torch.randint(0, n, (min(batch_size, n),)).tolist()
            losses.append(self.step_batch([specs[i] for i in idx], [texts[i] for i in idx], [weights[i] for i in idx]))
            if ckpt_path and arch_gen is not None and norm_meta is not None and (i + 1) % 200 == 0:
                save_task_nano_ckpt(ckpt_path, arch_gen, self.model, norm_meta)
        return losses


@torch.no_grad()
def sample_denorm_task_nano(model: TaskNanoWeightGenerator, spec: NanoSpec, task_text: str, meta: dict, device: str, noise_scale: float = 0.5) -> torch.Tensor:
    return denorm_nano(model.sample(spec, task_text, device, noise_scale), meta)


def save_task_nano_ckpt(path: str, arch_gen, weight_gen, norm_meta: dict):
    torch.save({"arch": arch_gen.state_dict(), "weights": weight_gen.state_dict(), "norm_meta": norm_meta, "vocab": weight_gen.vocab}, path)


def load_task_nano_ckpt(path: str, device: str):
    from src.arch_gen.nano_task_generator import TaskNanoArchGenerator
    ckpt = torch.load(path, map_location=device, weights_only=True)
    vocab = ckpt.get("vocab", 65)
    arch = TaskNanoArchGenerator().to(device)
    wgen = TaskNanoWeightGenerator(vocab=vocab).to(device)
    arch.load_state_dict(ckpt["arch"])
    wgen.load_state_dict(ckpt["weights"])
    return arch.eval(), wgen.eval(), ckpt["norm_meta"]


@torch.no_grad()
def init_task_nano_weights(model, spec: NanoSpec, wgen: TaskNanoWeightGenerator, task_text: str, norm_meta: dict, device: str, noise_scale: float = 0.5):
    load_flat_into_model(sample_denorm_task_nano(wgen, spec, task_text, norm_meta, device, noise_scale), model)
