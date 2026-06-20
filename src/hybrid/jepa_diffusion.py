"""JEPA-init + diffusion-refine hybrid for weight generation.

Pipeline: JEPA samples a plausible normalized weight vector (strong zero-shot),
then diffusion denoises from q(x_t | x_jepa) for t=refine_t..0 (img2img style).
Goal: keep JEPA's zero-shot quality while moving toward diffusion's FT-friendly manifold.
"""

import torch

from src.diffusion.simple_diffusion import WeightDiffusion
from src.jepa.hierarchical_jepa import HierarchicalWeightJEPA
from src.jepa.weight_jepa import WeightJEPA


def load_hjepa(path: str, device: str = "cpu") -> HierarchicalWeightJEPA:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    bounds = [tuple(x) for x in ckpt["bounds"]]
    model = HierarchicalWeightJEPA(bounds, latent_dim=ckpt["latent_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.latent_mean, model.latent_std = ckpt.get("latent_mean"), ckpt.get("latent_std")
    model.h_mean, model.h_std = ckpt.get("h_mean"), ckpt.get("h_std")
    return model.to(device).eval()


def load_jepa(path: str, device: str = "cpu") -> WeightJEPA:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    bounds = [tuple(x) for x in ckpt["bounds"]]
    model = WeightJEPA(bounds, latent_dim=ckpt["latent_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.latent_mean, model.latent_std = ckpt.get("latent_mean"), ckpt.get("latent_std")
    return model.to(device).eval()


def load_diffusion(path: str, dim: int, device: str = "cpu", timesteps: int = 200) -> WeightDiffusion:
    diff = WeightDiffusion(dim=dim, timesteps=timesteps, device=device)
    diff.load(path)
    diff.model.eval()
    return diff


class JEPADiffusionHybrid:
    """Sample: JEPA x0 -> forward noise to refine_t -> reverse diffuse to 0. Works with flat or H-JEPA (duck-typed sample())."""

    def __init__(self, jepa, diffusion: WeightDiffusion, refine_t: int = 50):
        self.jepa = jepa
        self.diffusion = diffusion
        self.refine_t = refine_t

    @torch.no_grad()
    def sample(self, n: int = 1, device: str = "cpu") -> torch.Tensor:
        x0 = self.jepa.sample(n=n, device=device)
        return self.diffusion.sample_refine(x0, start_t=self.refine_t).cpu()
