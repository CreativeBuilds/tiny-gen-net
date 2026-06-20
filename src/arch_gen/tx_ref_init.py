"""Fixed ref-tx weight init: JEPA, hybrid, cond, task-cond."""

import torch
from tqdm import tqdm

from src.arch_gen.tx_eval import eval_tx_flat, finetune_tx
from src.arch_gen.tx_spec import TxSpec
from src.diffusion.simple_diffusion import WeightDiffusion
from src.hybrid.jepa_diffusion import JEPADiffusionHybrid
from src.jepa.weight_jepa import WeightJEPA, WeightJEPATrainer
from src.models.variable_transformer import build_from_tx_spec
from src.utils.weights import denormalize_weights, flatten_state_dict, normalize_weights, norm_meta_to_tensors, param_slices


def collect_ref_tx_weights(spec: TxSpec, n: int, steps: int, seed: int, device: str) -> list[torch.Tensor]:
    rows = []
    for i in range(n):
        m = build_from_tx_spec(spec).to(device)
        finetune_tx(m, steps, seed=seed + i * 13, device=device)
        rows.append(flatten_state_dict(m))
    return rows


def train_ref_tx_jepa(weights: torch.Tensor, spec: TxSpec, device: str, steps: int = 1200, batch_size: int = 8) -> tuple[WeightJEPA, dict]:
    probe = build_from_tx_spec(spec)
    bounds = [(s["start"], s["end"]) for s in param_slices(probe)]
    norm_w, meta = normalize_weights(weights, "global", bounds)
    meta = norm_meta_to_tensors(meta)
    jepa = WeightJEPA(bounds, latent_dim=64)
    trainer = WeightJEPATrainer(jepa, device=device)
    bs = min(batch_size, norm_w.shape[0])
    for _ in tqdm(range(steps), desc="tx jepa"):
        idx = torch.randint(0, norm_w.shape[0], (bs,))
        trainer.step_batch(norm_w[idx])
    jepa.eval(); jepa.fit_latent_stats(norm_w.to(device))
    return jepa, meta


def train_ref_tx_diffusion(weights: torch.Tensor, spec: TxSpec, device: str, steps: int = 800, batch_size: int = 8) -> tuple[WeightDiffusion, dict]:
    probe = build_from_tx_spec(spec)
    bounds = [(s["start"], s["end"]) for s in param_slices(probe)]
    norm_w, meta = normalize_weights(weights, "global", bounds)
    meta = norm_meta_to_tensors(meta)
    diff = WeightDiffusion(norm_w.shape[1], timesteps=200, device=device)
    opt = torch.optim.AdamW(diff.model.parameters(), lr=1e-3)
    bs = min(batch_size, norm_w.shape[0])
    for _ in tqdm(range(steps), desc="tx diff"):
        idx = torch.randint(0, norm_w.shape[0], (bs,))
        diff.train_step(norm_w[idx].to(device), opt)
    diff.model.eval()
    return diff, meta


def eval_ref_tx_inits(spec: TxSpec, methods: dict[str, torch.Tensor], device: str, ft_steps: list[int], seed: int) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {k: [] for k in methods}
    for name, flats in methods.items():
        for i, flat in enumerate(flats):
            out[name].append(eval_tx_flat(flat, spec, device, ft_steps, seed + i))
    return out
