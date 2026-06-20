# tiny-gen-net

**Generative models for neural network weights — and architectures.**

Can diffusion / JEPA-style models learn weight distributions for tiny networks, then sample better-than-random initializations — and eventually full architectures? This repo explores that **incrementally and verifiably**, Karpathy-style.

---

## MVP Achieved (2026-06-19)

**The core idea works at tiny scale:** generate variable architectures + conditioned weights → runnable models that **beat random init**.

| Phase | Task | Zero-shot Δ vs random |
|-------|------|------------------------|
| 3 | Synthetic char, variable MLP | **+0.37** |
| 4b | Synthetic char, tiny transformer | **+0.31** |
| **5a** | **Real Shakespeare, 1–6M nanoGPT** | **+0.73** (task-nano) |

Phase 5a cloud run (`metrics_phase5a_v1.json`): 15 collected nets, task-conditioned weight hypernet, H100. See [EXPERIMENTS.md](EXPERIMENTS.md).

### Earlier MVP (Phase 3)

| Result | Value |
|--------|-------|
| Zero-shot loss (50 diverse archs) | **1.71** |
| Δ vs random | **+0.37** |
| FT100 loss | **1.07** (ref hybrid: 1.05) |

### Try the demo

```bash
source .venv/bin/activate
python demo_generate_model.py --arch transformer --task "small efficient transformer char predictor" --num 3
python demo_generate_model.py --arch mlp --task "efficient lightweight char predictor" --num 3
```

Phase 4 MLP: `task_cond_phase4_v2.pt` · Phase 4b transformer: `task_tx_phase4b_v1.pt` (from `experiments/phase4b_tiny_transformer.py`).

### What works

- **Tiny transformer (Phase 4b v2)** — task → tx arch + weights; match **0.73**; ref init zero=**0.31**
- **Task-conditioned MLP (Phase 4 v2)** — text → arch + weights with **~92% topology alignment**
- **Variable architecture generation** — token DSL + GRU sampler → valid `VariableTinyMLP`
- **Architecture-conditioned weights** — `CondWeightGenerator` hypernet assigns per-topology inits
- **Fixed-arch specialists** — flat JEPA (zero-shot), hybrid JEPA→diffusion (FT), global norm

### Limitations

- Tiny models only (~3k–105k params) for Phase 0–4b synthetic task
- Phase 5a: real Shakespeare char LM at 1M–10M params — **full training requires RunPod GPU**
- Cond generator is MSE hypernet (sub-chunk decode at nano scale), not a full generative weight manifold
- Phase 4 task conditioning uses keyword bag-of-words; Phase 5a uses sentence transformers

Learnings: [MVP_LEARNINGS.md](MVP_LEARNINGS.md) · Full log: [EXPERIMENTS.md](EXPERIMENTS.md)

---

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Demo (after refine checkpoints exist)
python demo_generate_model.py --diversity high --num 5

# Full MVP pipeline (train checkpoints ~15 min on MPS)
python experiments/phase3_conditioned_refine.py --num_eval 50 --skip_collect  # if weights collected
```

### Phase 5 — real text on RunPod

```bash
python experiments/phase5a_nano_text.py --smoke     # local wiring (~10s)
./scripts/runpod_launch.sh                          # H100 SXM (default, ~$2.69/hr)
RUNPOD_PULL=1 RUNPOD_STOP=1 ./scripts/runpod_train.sh --tag phase5a_v1
./scripts/runpod_gpus.sh                              # GPU tiers + $300/day budget
```

### Cursor IDE

| Command | Purpose |
|---------|---------|
| `/continue` | Advance one research increment |
| `/status` | Read-only project summary |
| `/run-phase3-refine` | Reproduce MVP checkpoints |

See [AGENTS.md](AGENTS.md)

---

## Repository Layout

```
tiny-gen-net/
├── demo_generate_model.py   ← MVP demo (arch + weights → eval)
├── MVP_LEARNINGS.md         ← what worked / what didn't
├── STATE.md                 ← current phase (read first for /continue)
├── VISION.md                ← roadmap post-MVP
├── EXPERIMENTS.md           ← experiment log
├── COMMANDS.md              ← CLI reference
├── src/
│   ├── models/              ← TinyMLP, VariableTinyMLP
│   ├── diffusion/           ← DDPM on weight vectors
│   ├── jepa/                ← flat + hierarchical JEPA
│   ├── hybrid/              ← JEPA + diffusion refine
│   ├── arch_gen/            ← ArchSpec, generator, conditioning
│   └── utils/
├── experiments/             ← phase0–3 scripts
├── scripts/                 ← collect_weights, multi_arch collection
└── checkpoints/             ← artifacts (gitignored *.pt)
```

---

## Phased Roadmap

| Phase | Status | Outcome |
|-------|--------|---------|
| **0** | ✅ | Baselines + weight collection |
| **1** | ✅ | Diffusion beats random after FT; **global norm** wins |
| **2** | ✅ | JEPA zero-shot; hybrid FT; H-JEPA negative |
| **3** | ✅ **MVP** | Variable arch + conditioned weights |
| **4** | ✅ initial | Task text → arch + weights (keyword conditioning) |
| **4b+** | 🔜 | Scale-up, richer task embed, tiny transformer |

Details: [VISION.md](VISION.md)

---

## The `/continue` Protocol

1. Read `STATE.md`, `EXPERIMENTS.md`
2. Propose one increment aligned with roadmap
3. Implement minimally, run experiment, update docs

---

## License

Research / educational use.
