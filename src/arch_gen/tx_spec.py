"""Transformer architecture spec: discrete tokens -> VariableTinyTransformer configs."""

from dataclasses import dataclass

from data.synthetic_text import VOCAB_SIZE

D_MODEL_CHOICES = (32, 48, 64, 96, 128)
N_LAYER_CHOICES = (2, 3, 4, 5, 6)
N_HEAD_CHOICES = (2, 4, 8, 16)
CTX_LEN = 8
MAX_TX_PARAMS = 100_000
REF_D_MODEL, REF_N_LAYER, REF_N_HEAD = 48, 3, 4

TOK_BOS, TOK_EOS = 0, 1
TOK_D = {d: 2 + i for i, d in enumerate(D_MODEL_CHOICES)}
TOK_L = {l: 2 + len(D_MODEL_CHOICES) + i for i, l in enumerate(N_LAYER_CHOICES)}
TOK_H = {h: 2 + len(D_MODEL_CHOICES) + len(N_LAYER_CHOICES) + i for i, h in enumerate(N_HEAD_CHOICES)}
D_FROM_TOK = {v: k for k, v in TOK_D.items()}
L_FROM_TOK = {v: k for k, v in TOK_L.items()}
H_FROM_TOK = {v: k for k, v in TOK_H.items()}
TX_VOCAB_SIZE = 4 + len(D_MODEL_CHOICES) + len(N_LAYER_CHOICES) + len(N_HEAD_CHOICES)


def estimate_tx_params(d_model: int, n_layer: int, n_head: int, vocab: int = VOCAB_SIZE, ctx_len: int = CTX_LEN) -> int:
    ff = max(d_model // 2, 16)
    per = 4 * d_model * d_model + 2 * d_model * ff + 4 * d_model
    return vocab * d_model + ctx_len * d_model + n_layer * per + d_model * vocab


@dataclass(frozen=True)
class TxSpec:
    d_model: int
    n_layer: int
    n_head: int
    ctx_len: int = CTX_LEN

    def validate(self) -> "TxSpec":
        if self.d_model not in D_MODEL_CHOICES: raise ValueError(f"d_model {self.d_model} not in {D_MODEL_CHOICES}")
        if self.n_layer not in N_LAYER_CHOICES: raise ValueError(f"n_layer {self.n_layer} not in {N_LAYER_CHOICES}")
        if self.n_head not in N_HEAD_CHOICES: raise ValueError(f"n_head {self.n_head} not in {N_HEAD_CHOICES}")
        if self.d_model % self.n_head != 0: raise ValueError(f"d_model {self.d_model} not divisible by n_head {self.n_head}")
        if estimate_tx_params(self.d_model, self.n_layer, self.n_head, ctx_len=self.ctx_len) > MAX_TX_PARAMS:
            raise ValueError(f"params exceed {MAX_TX_PARAMS}")
        return self

    def to_tokens(self) -> list[int]:
        return [TOK_BOS, TOK_D[self.d_model], TOK_L[self.n_layer], TOK_H[self.n_head], TOK_EOS]

    @staticmethod
    def from_tokens(tokens: list[int]) -> "TxSpec":
        if len(tokens) < 4 or tokens[0] != TOK_BOS: raise ValueError(f"Invalid tx tokens: {tokens}")
        body = tokens[1:-1] if tokens[-1] == TOK_EOS else tokens[1:]
        if len(body) < 3 or body[0] not in D_FROM_TOK or body[1] not in L_FROM_TOK or body[2] not in H_FROM_TOK:
            raise ValueError(f"Invalid tx tokens: {tokens}")
        return TxSpec(D_FROM_TOK[body[0]], L_FROM_TOK[body[1]], H_FROM_TOK[body[2]]).validate()

    @staticmethod
    def from_label(label: list) -> "TxSpec":
        return TxSpec(label[0], label[1], label[2], label[3] if len(label) > 3 else CTX_LEN).validate()

    def diagram(self) -> str:
        p = estimate_tx_params(self.d_model, self.n_layer, self.n_head, ctx_len=self.ctx_len)
        return "\n".join([f"TxSpec(d={self.d_model}, L={self.n_layer}, H={self.n_head}, ctx={self.ctx_len})",
            f"  tok+pos[vocab->{self.d_model}]",
            *[f"  block{i+1}: causal-MHA({self.n_head}h) + ff" for i in range(self.n_layer)],
            f"  head[{self.d_model}->vocab]  params={p:,}"])


def all_valid_tx_specs() -> list[TxSpec]:
    out = []
    for d in D_MODEL_CHOICES:
        for l in N_LAYER_CHOICES:
            for h in N_HEAD_CHOICES:
                if d % h != 0: continue
                try: out.append(TxSpec(d, l, h).validate())
                except ValueError: pass
    return out


def random_tx_spec(rng) -> TxSpec:
    valid = all_valid_tx_specs()
    return rng.choice(valid)


def pick_diverse_tx_specs(n: int, rng) -> list[TxSpec]:
    valid = all_valid_tx_specs()
    rng.shuffle(valid)
    seen, out = set(), []
    for s in valid:
        key = (s.d_model, s.n_layer)
        if key in seen: continue
        seen.add(key); out.append(s)
        if len(out) >= n: break
    while len(out) < n and len(out) < len(valid): out.append(valid[len(out)])
    return out[:n]
