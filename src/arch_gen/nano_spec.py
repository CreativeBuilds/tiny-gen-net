"""NanoGPT-scale architecture spec (500k–256M params). Supports continuous scale features."""

import math
from dataclasses import dataclass

N_EMBD_CHOICES = (128, 192, 256, 384, 512, 768)
N_LAYER_CHOICES = (4, 6, 8, 10, 12, 16, 24)
N_HEAD_CHOICES = (4, 8, 16)
BLOCK_SIZE_CHOICES = (128, 256)
MIN_NANO_PARAMS = 500_000
MAX_NANO_PARAMS = 256_000_000
REF_N_EMBD, REF_N_LAYER, REF_N_HEAD, REF_BLOCK = 256, 4, 8, 128

TOK_BOS, TOK_EOS = 0, 1
TOK_E = {e: 2 + i for i, e in enumerate(N_EMBD_CHOICES)}
TOK_L = {l: 2 + len(N_EMBD_CHOICES) + i for i, l in enumerate(N_LAYER_CHOICES)}
TOK_H = {h: 2 + len(N_EMBD_CHOICES) + len(N_LAYER_CHOICES) + i for i, h in enumerate(N_HEAD_CHOICES)}
TOK_B = {b: 2 + len(N_EMBD_CHOICES) + len(N_LAYER_CHOICES) + len(N_HEAD_CHOICES) + i for i, b in enumerate(BLOCK_SIZE_CHOICES)}
E_FROM_TOK = {v: k for k, v in TOK_E.items()}
L_FROM_TOK = {v: k for k, v in TOK_L.items()}
H_FROM_TOK = {v: k for k, v in TOK_H.items()}
B_FROM_TOK = {v: k for k, v in TOK_B.items()}
NANO_VOCAB_SIZE = 4 + len(N_EMBD_CHOICES) + len(N_LAYER_CHOICES) + len(N_HEAD_CHOICES) + len(BLOCK_SIZE_CHOICES)

PRESETS = {
    "smoke": (128, 4, 4, 128),
    "nano_1m": (128, 6, 4, 128),
    "nano_3m": (256, 4, 8, 128),
    "nano_8m": (256, 8, 8, 256),
    "nano_12m": (384, 6, 8, 256),
    "nano_16m": (384, 8, 16, 256),
    "nano_30m": (512, 12, 16, 256),
    "nano_100m": (768, 16, 16, 256),
}


def estimate_nano_params(n_embd: int, n_layer: int, n_head: int, vocab: int = 65, block_size: int = 128) -> int:
    per = 12 * n_embd * n_embd + 13 * n_embd  # attn + mlp(4x) + ln approx
    return vocab * n_embd + block_size * n_embd + n_layer * per + n_embd * vocab


@dataclass(frozen=True)
class NanoSpec:
    n_embd: int
    n_layer: int
    n_head: int
    block_size: int = 128

    def validate(self) -> "NanoSpec":
        if self.n_embd not in N_EMBD_CHOICES: raise ValueError(f"n_embd {self.n_embd} not in {N_EMBD_CHOICES}")
        if self.n_layer not in N_LAYER_CHOICES: raise ValueError(f"n_layer {self.n_layer} not in {N_LAYER_CHOICES}")
        if self.n_head not in N_HEAD_CHOICES: raise ValueError(f"n_head {self.n_head} not in {N_HEAD_CHOICES}")
        if self.block_size not in BLOCK_SIZE_CHOICES: raise ValueError(f"block_size {self.block_size} not in {BLOCK_SIZE_CHOICES}")
        if self.n_embd % self.n_head != 0: raise ValueError(f"n_embd {self.n_embd} not divisible by n_head {self.n_head}")
        p = estimate_nano_params(self.n_embd, self.n_layer, self.n_head, block_size=self.block_size)
        if p < MIN_NANO_PARAMS or p > MAX_NANO_PARAMS: raise ValueError(f"params {p:,} outside [{MIN_NANO_PARAMS:,}, {MAX_NANO_PARAMS:,}]")
        return self

    @staticmethod
    def preset(name: str) -> "NanoSpec":
        if name not in PRESETS: raise ValueError(f"Unknown preset {name}; choose from {list(PRESETS)}")
        return NanoSpec(*PRESETS[name]).validate()

    def to_tokens(self) -> list[int]:
        return [TOK_BOS, TOK_E[self.n_embd], TOK_L[self.n_layer], TOK_H[self.n_head], TOK_B[self.block_size], TOK_EOS]

    @staticmethod
    def from_tokens(tokens: list[int]) -> "NanoSpec":
        if len(tokens) < 5 or tokens[0] != TOK_BOS: raise ValueError(f"Invalid nano tokens: {tokens}")
        body = tokens[1:-1] if tokens[-1] == TOK_EOS else tokens[1:]
        if len(body) < 4: raise ValueError(f"Invalid nano tokens: {tokens}")
        return NanoSpec(E_FROM_TOK[body[0]], L_FROM_TOK[body[1]], H_FROM_TOK[body[2]], B_FROM_TOK[body[3]]).validate()

    @staticmethod
    def from_label(label: list) -> "NanoSpec":
        return NanoSpec(label[0], label[1], label[2], label[3] if len(label) > 3 else 128).validate()

    def param_count(self, vocab: int = 65) -> int:
        return estimate_nano_params(self.n_embd, self.n_layer, self.n_head, vocab, self.block_size)

    def log_params(self, vocab: int = 65) -> float:
        return math.log10(max(self.param_count(vocab), 1))

    def diagram(self) -> str:
        p = estimate_nano_params(self.n_embd, self.n_layer, self.n_head, block_size=self.block_size)
        return "\n".join([f"NanoSpec(d={self.n_embd}, L={self.n_layer}, H={self.n_head}, ctx={self.block_size})",
            f"  tok+pos[vocab->{self.n_embd}]",
            *[f"  block{i+1}: causal-MHA({self.n_head}h) + mlp4x" for i in range(self.n_layer)],
            f"  lm_head[{self.n_embd}->vocab]  params≈{p:,}"])


def all_valid_nano_specs() -> list[NanoSpec]:
    out = []
    for e in N_EMBD_CHOICES:
        for l in N_LAYER_CHOICES:
            for h in N_HEAD_CHOICES:
                for b in BLOCK_SIZE_CHOICES:
                    if e % h != 0: continue
                    try: out.append(NanoSpec(e, l, h, b).validate())
                    except ValueError: pass
    return out


def random_nano_spec(rng) -> NanoSpec:
    return rng.choice(all_valid_nano_specs())


def extrapolation_holdout_presets() -> list[str]:
    """Larger presets for OOD extrapolation eval (may exceed training grid)."""
    return ["nano_12m", "nano_16m", "nano_30m"]


def anchor_presets() -> list[str]:
    return ["nano_12m", "nano_16m"]


def training_presets() -> list[str]:
    return ["nano_1m", "nano_3m", "nano_8m"]


def pick_diverse_nano_specs(n: int, rng) -> list[NanoSpec]:
    valid = all_valid_nano_specs()
    rng.shuffle(valid)
    seen, out = set(), []
    for s in valid:
        key = (s.n_embd, s.n_layer)
        if key in seen: continue
        seen.add(key); out.append(s)
        if len(out) >= n: break
    while len(out) < n and len(out) < len(valid): out.append(valid[len(out)])
    return out[:n]
