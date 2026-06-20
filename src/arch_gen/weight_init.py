"""Initialize weights for generated architectures — hybrid (ref arch), conditioned, or random."""

import torch

from src.arch_gen.conditioning import CondWeightGenerator, load_cond_checkpoint, sample_denorm
from src.arch_gen.spec import ArchSpec
from src.arch_gen.task_weights import TaskCondWeightGenerator, sample_denorm_task
from src.hybrid.jepa_diffusion import JEPADiffusionHybrid, load_diffusion, load_jepa
from src.models.variable_mlp import VariableTinyMLP
from src.utils.weights import denormalize_weights, load_flat_into_model, norm_meta_to_tensors


def load_hybrid(jepa_ckpt: str, diffusion_ckpt: str, dim: int, device: str, refine_t: int = 50) -> JEPADiffusionHybrid:
    return JEPADiffusionHybrid(load_jepa(jepa_ckpt, device), load_diffusion(diffusion_ckpt, dim, device), refine_t=refine_t)


@torch.no_grad()
def sample_reference_weights(hybrid: JEPADiffusionHybrid, norm_meta: dict, device: str) -> torch.Tensor:
    norm_meta = norm_meta_to_tensors(norm_meta)
    return denormalize_weights(hybrid.sample(n=1, device=device)[0], norm_meta)


@torch.no_grad()
def init_task_weights(model: VariableTinyMLP, spec: ArchSpec, wgen: TaskCondWeightGenerator, task_text: str,
                      norm_meta: dict, device: str = "cpu", noise_scale: float = 0.5) -> str:
    load_flat_into_model(sample_denorm_task(wgen, spec, task_text, norm_meta, device, noise_scale), model)
    return "task_cond"


@torch.no_grad()
def init_model_weights(model: VariableTinyMLP, spec: ArchSpec, *, hybrid: JEPADiffusionHybrid | None = None,
                       norm_meta: dict | None = None, cond_gen: CondWeightGenerator | None = None,
                       cond_norm: dict | None = None, device: str = "cpu", method: str = "auto",
                       noise_scale: float = 1.0) -> str:
    if method == "random": return "random"
    if method == "conditioned" or (method == "auto" and cond_gen is not None and cond_norm is not None):
        load_flat_into_model(sample_denorm(cond_gen, spec, cond_norm, device, noise_scale=noise_scale), model)
        return "conditioned"
    if method != "random" and spec.is_jepa_compatible and hybrid is not None and norm_meta is not None:
        load_flat_into_model(sample_reference_weights(hybrid, norm_meta, device), model)
        return "hybrid"
    return "random"
