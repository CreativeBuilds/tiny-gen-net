"""Frozen sentence-transformer task embeddings (Phase 5a)."""

import torch
import torch.nn as nn

SENT_DIM = 384
_MODEL = None


def _load_model():
    global _MODEL
    if _MODEL is not None: return _MODEL
    from sentence_transformers import SentenceTransformer
    _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    _MODEL.eval()
    for p in _MODEL.parameters(): p.requires_grad = False
    return _MODEL


@torch.no_grad()
def encode_sentence(text: str, device: str = "cpu") -> torch.Tensor:
    m = _load_model()
    emb = m.encode(text, convert_to_tensor=True, device=device)
    return emb.float().view(-1)


class SentenceTaskEmbedder(nn.Module):
    """Drop-in replacement for keyword TaskEmbedder — frozen MiniLM."""

    DIM = SENT_DIM

    def encode_text(self, text: str, device: str) -> torch.Tensor:
        return encode_sentence(text, device)

    def forward(self, texts: list[str], device: str) -> torch.Tensor:
        return torch.stack([self.encode_text(t, device) for t in texts])
