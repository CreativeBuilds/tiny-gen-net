"""Tiny Shakespeare char corpus for Phase 5a real-text training."""

import urllib.request
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "text"
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def _ensure_corpus() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "shakespeare.txt"
    if not path.exists():
        urllib.request.urlretrieve(URL, path)
    return path


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


def load_shakespeare(val_frac: float = 0.1, seed: int = 42) -> tuple[str, str, CharTokenizer]:
    text = _ensure_corpus().read_text(encoding="utf-8")
    n = len(text)
    split = int(n * (1 - val_frac))
    tok = CharTokenizer(text)
    return text[:split], text[split:], tok


def get_batch(data: str, tok: CharTokenizer, batch_size: int, block_size: int, device: str, rng) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    xs, ys = [], []
    for i in ix:
        chunk = tok.encode(data[i : i + block_size + 1])
        xs.append(chunk[:-1])
        ys.append(chunk[1:])
    x = torch.tensor(xs, dtype=torch.long, device=device)
    y = torch.tensor(ys, dtype=torch.long, device=device)
    return x, y
