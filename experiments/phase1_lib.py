"""Shared Phase 1 diffusion train + eval (used by phase1_weight_diffusion and norm ablation)."""

from pathlib import Path

import torch
from tqdm import tqdm

from src.diffusion.simple_diffusion import WeightDiffusion
from src.models.tiny_mlp import TinyMLP
from src.utils.device import device_str, get_device
from src.utils.train_loop import eval_model_loss, finetune_steps, set_seed
from src.utils.viz import plot_bar_comparison, plot_loss_curve, plot_multi_loss_curves, plot_weight_histograms
from src.utils.weights import (
    NormMode,
    denormalize_weights,
    flatten_state_dict,
    load_flat_into_model,
    load_raw_weights,
    normalize_weights,
    norm_meta_to_tensors,
)


def eval_flat(flat: torch.Tensor, hidden: int, device: str, ft_steps: list[int], seed: int, model_factory=None) -> dict:
    m = (model_factory() if model_factory else TinyMLP(hidden_dim=hidden)).to(device)
    load_flat_into_model(flat, m)
    out = {}
    z = eval_model_loss(m, seed=seed, device=device)
    out["zero_loss"], out["zero_acc"] = z["eval_loss"], z["eval_acc"]
    for n in ft_steps:
        if n <= 0: continue
        mc = (model_factory() if model_factory else TinyMLP(hidden_dim=hidden)).to(device)
        load_flat_into_model(flat, mc)
        curve = finetune_steps(mc, n, seed=seed, device=device)
        a = eval_model_loss(mc, seed=seed, device=device)
        out[f"ft{n}_loss"], out[f"ft{n}_acc"] = a["eval_loss"], a["eval_acc"]
        out[f"ft{n}_curve"] = curve
    return out


def avg_metric(rows: list[dict], key: str) -> float:
    return sum(r[key] for r in rows) / max(len(rows), 1)


def run_phase1_experiment(
    weights_norm: torch.Tensor,
    norm_meta: dict,
    *,
    tag: str,
    hidden: int = 64,
    diffusion_steps: int = 2000,
    timesteps: int = 200,
    batch_size: int = 32,
    lr: float = 1e-4,
    num_samples: int = 30,
    num_collected_eval: int = 20,
    ft_steps: list[int] | None = None,
    seed: int = 42,
    plot_dir: str | Path = "checkpoints/plots",
    device: str | None = None,
    model_factory=None,
) -> dict:
    """Train diffusion on normalized weights, sample, eval vs random/collected. Returns metrics dict."""
    ft_steps = ft_steps or [100, 200]
    norm_meta = norm_meta_to_tensors(norm_meta)
    set_seed(seed)
    device = device_str(device or get_device())
    dim = weights_norm.shape[1]
    print(f"Phase 1 [{tag}] norm={norm_meta.get('norm_mode')} | weights {weights_norm.shape} | device={device}")

    diff = WeightDiffusion(dim=dim, timesteps=timesteps, device=device)
    opt = torch.optim.AdamW(diff.model.parameters(), lr=lr)
    train_losses: list[float] = []
    diff.model.train()
    bs = min(batch_size, weights_norm.shape[0])
    for _ in tqdm(range(diffusion_steps), desc=f"diffusion [{tag}]"):
        idx = torch.randint(0, weights_norm.shape[0], (bs,))
        train_losses.append(diff.train_step(weights_norm[idx].to(device), opt))

    ckpt_dir = Path("checkpoints/diffusion")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    diff.save(str(ckpt_dir / f"weight_denoiser_{tag}.pt"))
    plot_dir = Path(plot_dir)
    plot_loss_curve(train_losses[:: max(1, len(train_losses) // 500)], out_path=plot_dir / f"{tag}_diffusion_train.png", title=f"Diffusion loss ({tag})")

    diff.model.eval()
    sampled_raw = denormalize_weights(diff.sample(n=num_samples).cpu(), norm_meta)
    collected_raw = denormalize_weights(weights_norm, norm_meta)
    random_rows = torch.stack([flatten_state_dict((model_factory() if model_factory else TinyMLP(hidden_dim=hidden))) for _ in range(num_samples)])
    plot_weight_histograms(
        {"collected": collected_raw, "sampled": sampled_raw, "random_init": random_rows},
        out_path=plot_dir / f"{tag}_weight_hist.png",
        title=f"Weight dist ({tag})",
    )

    n_col = min(num_collected_eval, weights_norm.shape[0])
    col_idx = list(range(weights_norm.shape[0] - n_col, weights_norm.shape[0]))
    gen_rows, rnd_rows, col_rows = [], [], []
    for i in range(num_samples):
        gen_rows.append(eval_flat(sampled_raw[i], hidden, device, ft_steps, seed + i, model_factory=model_factory))
        set_seed(1000 + i)
        rnd_rows.append(eval_flat(random_rows[i], hidden, device, ft_steps, seed + i, model_factory=model_factory))
    for j, ci in enumerate(col_idx):
        col_rows.append(eval_flat(collected_raw[ci], hidden, device, ft_steps, seed + 10000 + j, model_factory=model_factory))

    metrics = {
        "tag": tag,
        "norm_mode": norm_meta.get("norm_mode"),
        "num_train_weights": weights_norm.shape[0],
        "weight_dim": dim,
        "diffusion_steps": diffusion_steps,
        "diffusion_final_loss": train_losses[-1],
        "num_samples_eval": num_samples,
        "generated_zero_loss": avg_metric(gen_rows, "zero_loss"),
        "generated_zero_acc": avg_metric(gen_rows, "zero_acc"),
        "random_zero_loss": avg_metric(rnd_rows, "zero_loss"),
        "random_zero_acc": avg_metric(rnd_rows, "zero_acc"),
        "collected_zero_loss": avg_metric(col_rows, "zero_loss"),
        "collected_zero_acc": avg_metric(col_rows, "zero_acc"),
    }
    for n in ft_steps:
        if n <= 0: continue
        metrics[f"generated_ft{n}_loss"] = avg_metric(gen_rows, f"ft{n}_loss")
        metrics[f"generated_ft{n}_acc"] = avg_metric(gen_rows, f"ft{n}_acc")
        metrics[f"random_ft{n}_loss"] = avg_metric(rnd_rows, f"ft{n}_loss")
        metrics[f"random_ft{n}_acc"] = avg_metric(rnd_rows, f"ft{n}_acc")
        metrics[f"collected_ft{n}_loss"] = avg_metric(col_rows, f"ft{n}_loss")
        metrics[f"collected_ft{n}_acc"] = avg_metric(col_rows, f"ft{n}_acc")

    metrics["zero_shot_delta_vs_random"] = metrics["random_zero_loss"] - metrics["generated_zero_loss"]
    for n in ft_steps:
        if n <= 0: continue
        metrics[f"ft{n}_delta_vs_random"] = metrics[f"random_ft{n}_loss"] - metrics[f"generated_ft{n}_loss"]

    loss_labels = ["gen_zero", "rnd_zero", "col_zero"]
    loss_vals = [metrics["generated_zero_loss"], metrics["random_zero_loss"], metrics["collected_zero_loss"]]
    acc_labels, acc_vals = list(loss_labels), [metrics["generated_zero_acc"], metrics["random_zero_acc"], metrics["collected_zero_acc"]]
    for n in ft_steps:
        if n <= 0: continue
        loss_labels += [f"gen_ft{n}", f"rnd_ft{n}", f"col_ft{n}"]
        loss_vals += [metrics[f"generated_ft{n}_loss"], metrics[f"random_ft{n}_loss"], metrics[f"collected_ft{n}_loss"]]
        acc_labels += [f"gen_ft{n}", f"rnd_ft{n}", f"col_ft{n}"]
        acc_vals += [metrics[f"generated_ft{n}_acc"], metrics[f"random_ft{n}_acc"], metrics[f"collected_ft{n}_acc"]]
    plot_bar_comparison(loss_labels, loss_vals, out_path=plot_dir / f"{tag}_comparison_loss.png", title=f"Loss ({tag})", ylabel="Loss")
    plot_bar_comparison(acc_labels, acc_vals, out_path=plot_dir / f"{tag}_comparison_acc.png", title=f"Acc ({tag})", ylabel="Accuracy")

    ft_curves = {}
    for i in range(min(3, len(gen_rows))):
        for n in ft_steps:
            if n <= 0 or f"ft{n}_curve" not in gen_rows[i]: continue
            ft_curves[f"gen{i}_ft{n}"] = gen_rows[i][f"ft{n}_curve"]
            ft_curves[f"rnd{i}_ft{n}"] = rnd_rows[i][f"ft{n}_curve"]
    if ft_curves:
        plot_multi_loss_curves(ft_curves, out_path=plot_dir / f"{tag}_finetune_curves.png", title=f"FT curves ({tag})")

    return metrics


def prepare_normalized_weights(raw_path: str | Path, mode: NormMode, layer_slices: list[tuple[int, int]]) -> tuple[torch.Tensor, dict]:
    raw = load_raw_weights(raw_path)
    norm_w, norm_meta = normalize_weights(raw, mode=mode, layer_slices=layer_slices)
    meta = {"layer_slices": [list(x) for x in layer_slices], **norm_meta}
    return norm_w, meta
