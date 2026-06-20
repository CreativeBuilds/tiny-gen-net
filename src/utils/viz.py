"""Visualization helpers — losses, weight distributions, comparisons."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def _ensure_dir(path: str | Path | None) -> Path | None:
    if path is None: return None
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def plot_loss_curve(losses: list[float], out_path: str | Path | None = None, title: str = "Training Loss"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if out_path:
        fig.savefig(_ensure_dir(out_path), dpi=120)
        plt.close(fig)
        return
    plt.show()


def plot_multi_loss_curves(curves: dict[str, list[float]], out_path: str | Path | None = None, title: str = "Loss Curves"):
    fig, ax = plt.subplots(figsize=(9, 4))
    for label, losses in curves.items():
        ax.plot(losses, label=label, linewidth=1.2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if out_path:
        fig.savefig(_ensure_dir(out_path), dpi=120)
        plt.close(fig)
        return
    plt.show()


def plot_weight_histograms(
    tensors: dict[str, torch.Tensor | np.ndarray],
    out_path: str | Path | None = None,
    title: str = "Weight Distributions",
    bins: int = 80,
):
    fig, ax = plt.subplots(figsize=(9, 4))
    for label, w in tensors.items():
        arr = w.detach().cpu().numpy().flatten() if isinstance(w, torch.Tensor) else np.asarray(w).flatten()
        ax.hist(arr, bins=bins, alpha=0.5, density=True, label=label)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if out_path:
        fig.savefig(_ensure_dir(out_path), dpi=120)
        plt.close(fig)
        return
    plt.show()


def plot_bar_comparison(labels: list[str], values: list[float], out_path: str | Path | None = None, title: str = "Comparison", ylabel: str = "Metric"):
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"][: len(labels)]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for i, v in enumerate(values):
        ax.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    if out_path:
        fig.savefig(_ensure_dir(out_path), dpi=120)
        plt.close(fig)
        return
    plt.show()


def plot_scatter(x: list[float], y: list[float], out_path: str | Path | None = None, title: str = "Scatter",
                 xlabel: str = "x", ylabel: str = "y", labels: list[str] | None = None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x, y, alpha=0.75)
    if labels:
        for xi, yi, lb in zip(x, y, labels):
            ax.annotate(lb, (xi, yi), fontsize=7, alpha=0.7)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if out_path:
        fig.savefig(_ensure_dir(out_path), dpi=120); plt.close(fig); return
    plt.show()
