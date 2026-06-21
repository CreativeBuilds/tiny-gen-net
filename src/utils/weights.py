"""Weight flatten/unflatten, normalization variants, and collection I/O."""

import json
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

NormMode = Literal["global", "layer", "perdim"]
NORM_MODES: tuple[NormMode, ...] = ("global", "layer", "perdim")


def flatten_state_dict(model: nn.Module) -> torch.Tensor:
    parts = [p.detach().cpu().flatten() for p in model.parameters()]
    return torch.cat(parts)


def unflatten_to_state_dict(flat: torch.Tensor, model: nn.Module) -> dict:
    state, offset = {}, 0
    for name, p in model.named_parameters():
        n = p.numel()
        state[name] = flat[offset : offset + n].view_as(p).clone()
        offset += n
    if offset != flat.numel(): raise ValueError(f"Flat size {flat.numel()} != model params {offset}")
    return state


def load_flat_into_model(flat: torch.Tensor, model: nn.Module) -> nn.Module:
    model.load_state_dict(unflatten_to_state_dict(flat, model))
    return model


def weight_dim(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def param_slices(model: nn.Module) -> list[dict]:
    """Layer groups = one entry per parameter tensor (embed, fc1.w, fc1.b, ...)."""
    slices, offset = [], 0
    for name, p in model.named_parameters():
        n = p.numel()
        slices.append({"name": name, "start": offset, "end": offset + n})
        offset += n
    return slices


def _layer_bounds(meta_or_model) -> list[tuple[int, int]]:
    if isinstance(meta_or_model, nn.Module): return [(s["start"], s["end"]) for s in param_slices(meta_or_model)]
    if "layer_slices" in meta_or_model: return [tuple(x) for x in meta_or_model["layer_slices"]]
    raise ValueError("Need model or meta with layer_slices for layer normalization")


def normalize_weights(
    weights: torch.Tensor,
    mode: NormMode = "perdim",
    layer_slices: list[tuple[int, int]] | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict]:
    """Normalize (N, D) weight matrix. Returns normalized weights + meta for denormalize."""
    if mode not in NORM_MODES: raise ValueError(f"Unknown norm mode: {mode}")
    if mode == "global":
        mean, std = weights.mean(), weights.std().clamp_min(eps)
        return (weights - mean) / std, {"norm_mode": "global", "mean": mean, "std": std}
    if mode == "perdim":
        mean, std = weights.mean(dim=0), weights.std(dim=0).clamp_min(eps)
        return (weights - mean) / std, {"norm_mode": "perdim", "mean": mean, "std": std}
    # layer-wise: one mean/std per parameter tensor
    if not layer_slices: raise ValueError("layer_slices required for layer normalization")
    norm = weights.clone()
    means, stds = [], []
    for s, e in layer_slices:
        chunk = weights[:, s:e]
        m, sd = chunk.mean(), chunk.std().clamp_min(eps)
        means.append(m)
        stds.append(sd)
        norm[:, s:e] = (chunk - m) / sd
    return norm, {"norm_mode": "layer", "mean": torch.stack(means), "std": torch.stack(stds), "layer_slices": [list(x) for x in layer_slices]}


def denormalize_weights(norm: torch.Tensor, meta: dict) -> torch.Tensor:
    """Map normalized weights back to raw scale using meta from normalize_weights."""
    mode = meta.get("norm_mode", "perdim")
    mean, std = meta["mean"], meta["std"]
    if not isinstance(mean, torch.Tensor): mean = torch.tensor(mean)
    if not isinstance(std, torch.Tensor): std = torch.tensor(std)
    if mode == "global": return norm * std + mean
    if mode == "perdim": return norm * std + mean
    if mode == "layer":
        out = norm.clone()
        for i, (s, e) in enumerate(_layer_bounds(meta)):
            out[..., s:e] = norm[..., s:e] * std[i] + mean[i]
        return out
    raise ValueError(f"Unknown norm_mode: {mode}")


def norm_meta_to_tensors(meta: dict) -> dict:
    out = dict(meta)
    for k in ("mean", "std"):
        if k not in out: continue
        v = out[k]
        if isinstance(v, torch.Tensor): continue
        if isinstance(v, (int, float)): out[k] = torch.tensor(v)
        elif isinstance(v, list): out[k] = torch.tensor(v)
    return out


def dataset_variance(weights: torch.Tensor) -> dict:
    """Spread of a (N, D) weight collection across models.

    total_var: mean over dims of the per-dim variance across the N models
               (how much the dataset moves per coordinate). Lower after a good
               symmetry alignment if permutation slop was a real noise source.
    mean_pairwise_l2: mean L2 distance between distinct model pairs (estimated
               on up to 64 models to stay cheap).
    """
    w = weights.float()
    n = w.shape[0]
    per_dim_var = w.var(dim=0, unbiased=False)
    total_var = per_dim_var.mean().item()
    m = min(n, 64)
    sub = w[:m]
    d = torch.cdist(sub, sub)
    if m > 1:
        mask = ~torch.eye(m, dtype=torch.bool)
        mean_pairwise_l2 = d[mask].mean().item()
    else:
        mean_pairwise_l2 = 0.0
    return {"total_var": total_var, "mean_pairwise_l2": mean_pairwise_l2, "num_models": n}


def save_weight_collection(weights: torch.Tensor, meta: dict, out_dir: str | Path, filename: str = "weights.pt"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(weights, out_dir / filename)
    serial = {}
    for k, v in meta.items():
        if isinstance(v, torch.Tensor): serial[k] = v.tolist() if v.ndim > 0 or k in ("mean", "std") else v.item()
        else: serial[k] = v
    (out_dir / "meta.json").write_text(json.dumps(serial, indent=2))


def load_weight_collection(path: str | Path) -> tuple[torch.Tensor, dict]:
    path = Path(path)
    weights = torch.load(path, weights_only=True)
    meta_path = path.parent / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return weights, norm_meta_to_tensors(meta)


def load_raw_weights(raw_path: str | Path) -> torch.Tensor:
    return torch.load(Path(raw_path), weights_only=True)
