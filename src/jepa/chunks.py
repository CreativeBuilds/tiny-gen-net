"""Split/merge flat weight vectors into layer-aligned chunks."""

import torch


def split_chunks(flat: torch.Tensor, bounds: list[tuple[int, int]]) -> list[torch.Tensor]:
    """flat: (B, D) or (D,) -> list of chunk tensors."""
    if flat.dim() == 1: return [flat[s:e] for s, e in bounds]
    return [flat[:, s:e] for s, e in bounds]


def merge_chunks(chunks: list[torch.Tensor], bounds: list[tuple[int, int]], dim: int) -> torch.Tensor:
    """Concatenate chunks back to (B, D) or (D,)."""
    if chunks[0].dim() == 1:
        out = torch.zeros(dim, device=chunks[0].device, dtype=chunks[0].dtype)
        for c, (s, e) in zip(chunks, bounds): out[s:e] = c
        return out
    B = chunks[0].shape[0]
    out = torch.zeros(B, dim, device=chunks[0].device, dtype=chunks[0].dtype)
    for c, (s, e) in zip(chunks, bounds): out[:, s:e] = c
    return out


def chunk_dims(bounds: list[tuple[int, int]]) -> list[int]:
    return [e - s for s, e in bounds]
