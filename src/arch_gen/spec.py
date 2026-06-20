"""Architecture spec: discrete tokens -> valid VariableTinyMLP configs."""

from dataclasses import dataclass

HIDDEN_CHOICES = (24, 32, 48, 64, 96, 128, 160)
DEPTH_CHOICES = (1, 2, 3)
REF_HIDDEN, REF_DEPTH = 64, 1

TOK_BOS, TOK_EOS = 0, 1
TOK_H = {h: 2 + i for i, h in enumerate(HIDDEN_CHOICES)}
TOK_D = {d: 2 + len(HIDDEN_CHOICES) + i for i, d in enumerate(DEPTH_CHOICES)}
TOK_S = {False: 2 + len(HIDDEN_CHOICES) + len(DEPTH_CHOICES), True: 3 + len(HIDDEN_CHOICES) + len(DEPTH_CHOICES)}
H_FROM_TOK = {v: k for k, v in TOK_H.items()}
D_FROM_TOK = {v: k for k, v in TOK_D.items()}
S_FROM_TOK = {v: k for k, v in TOK_S.items()}
VOCAB_SIZE = 4 + len(HIDDEN_CHOICES) + len(DEPTH_CHOICES)


@dataclass(frozen=True)
class ArchSpec:
    hidden_dim: int
    depth: int
    skip: bool = False  # residual add h_prev after first block (depth >= 2)

    def validate(self) -> "ArchSpec":
        if self.hidden_dim not in HIDDEN_CHOICES: raise ValueError(f"hidden_dim {self.hidden_dim} not in {HIDDEN_CHOICES}")
        if self.depth not in DEPTH_CHOICES: raise ValueError(f"depth {self.depth} not in {DEPTH_CHOICES}")
        if self.skip and self.depth < 2: raise ValueError("skip requires depth >= 2")
        return self

    @property
    def is_jepa_compatible(self) -> bool:
        return self.hidden_dim == REF_HIDDEN and self.depth == REF_DEPTH and not self.skip

    def to_tokens(self) -> list[int]:
        return [TOK_BOS, TOK_H[self.hidden_dim], TOK_D[self.depth], TOK_S[self.skip], TOK_EOS]

    @staticmethod
    def from_tokens(tokens: list[int]) -> "ArchSpec":
        if len(tokens) < 4 or tokens[0] != TOK_BOS: raise ValueError(f"Invalid arch token sequence: {tokens}")
        body = tokens[1:-1] if tokens[-1] == TOK_EOS else tokens[1:]
        if len(body) < 2: raise ValueError(f"Invalid arch token sequence: {tokens}")
        if body[0] not in H_FROM_TOK or body[1] not in D_FROM_TOK: raise ValueError(f"Invalid arch token sequence: {tokens}")
        skip = S_FROM_TOK[body[2]] if len(body) >= 3 and body[2] in S_FROM_TOK else False
        return ArchSpec(H_FROM_TOK[body[0]], D_FROM_TOK[body[1]], skip).validate()

    @staticmethod
    def from_legacy(h: int, d: int, skip: bool = False) -> "ArchSpec":
        return ArchSpec(h, d, skip).validate()

    @staticmethod
    def from_label(label: list) -> "ArchSpec":
        if len(label) == 2: return ArchSpec(label[0], label[1], False).validate()
        return ArchSpec(label[0], label[1], bool(label[2])).validate()

    def diagram(self) -> str:
        h, d, sk = self.hidden_dim, self.depth, self.skip
        lines = [f"ArchSpec(h={h}, depth={d}, skip={sk})", f"  embed[vocab->{h}]"]
        lines.append(f"  fc1[{h * 2}->{h}]  # emb + h_prev")
        if sk: lines.append("  + h_prev residual")
        for i in range(2, d + 1): lines.append(f"  fc{i}[{h}->{h}]")
        lines.append(f"  head[{h}->vocab]")
        return "\n".join(lines)


def all_valid_specs() -> list[ArchSpec]:
    out = []
    for h in HIDDEN_CHOICES:
        for d in DEPTH_CHOICES:
            out.append(ArchSpec(h, d, False).validate())
            if d >= 2: out.append(ArchSpec(h, d, True).validate())
    return out


def random_spec(rng) -> ArchSpec:
    d = rng.choice(DEPTH_CHOICES)
    sk = rng.choice([False, True]) if d >= 2 else False
    return ArchSpec(rng.choice(HIDDEN_CHOICES), d, sk).validate()
