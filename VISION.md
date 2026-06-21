# Vision — tiny-gen-net

## The Core Question (Answered at Tiny Scale)

Can generative models produce neural network **weights** and **architectures** such that the output is a runnable model that beats random initialization?

**MVP answer (2026-06-19): Yes** — on a synthetic char task with tiny MLPs (3k–105k params), architecture-conditioned weight generation yields **Δ +0.37 zero-shot** vs random across diverse topologies. See [MVP_LEARNINGS.md](MVP_LEARNINGS.md).

## What We Built

```
Architecture tokens (ArchSpec)
           │
           ▼
   ┌───────────────────┐
   │  ArchGenerator    │  GRU over discrete tokens
   └─────────┬─────────┘
             │
             ▼
   ┌───────────────────┐
   │ CondWeightGenerator│  arch embed + latent → hypernet weights
   └─────────┬─────────┘
             │
             ▼
   VariableTinyMLP → eval on task

Fixed-arch path (specialists):
  JEPA (zero-shot) · Hybrid JEPA→diffusion (FT) · DDPM (weight manifold)
```

## Completed Phases

| Phase | Goal | Result |
|-------|------|--------|
| 0 | Baselines + weight collection | ✅ |
| 1 | Diffusion on weights | ✅ FT signal; global norm critical |
| 2 | JEPA + hybrid | ✅ JEPA zero-shot; hybrid FT; H-JEPA negative |
| 3 | Variable arch + conditioned weights | ✅ **MVP** |

## Post-MVP Roadmap

### Phase 4 — task conditioning ✅ (refined)

- ✅ **v2 refine** — MLP task conditioning (92% match)
- ✅ **4b tiny transformer** — decoder-only tx; task-tx ref init zero=**0.31**, FT100=**0.28**
- ✅ **4b refine** — tx steering match **0.73**; fixed-ref init comparison (task-tx > tx-JEPA)

### Phase 5 — real text + scale

- ✅ **5a** — Shakespeare + sentence embeddings + nanoGPT (1M–10M); task-nano Δ **+0.73** on H100
- ✅ **5b** — hierarchical progressive scaffold measured; **negative** (in-grid −0.22 Δ, extrap broken)
- **5c** — scale-consistency + block planning + 12m/16m anchor warm-start; target extrap gap

---

- End-to-end: task description → trainable model without cold start
- Open-source as educational reference (Karpathy-style research codebase)
- Hooks for external trainers (HF, Lightning) to consume generated inits

## Design Principles (Validated)

1. **Staged complexity** — fixed arch → weights → variable arch → conditioning
2. **Global normalization** — essential for diffusion/JEPA on flat vectors
3. **Always compare to random** — Δ vs random is the honest metric
4. **Negative results count** — H-JEPA, per-dim norm, unconditioned Phase 3 v1
5. **Minimal code, heavy experiments log** — `EXPERIMENTS.md` is the lab notebook

## Open Questions (Post-MVP)

- Does cond hypernet generalize OOD (archs outside 35-spec grid)?
- Can task conditioning replace multi-arch collection? (partially — steering works; loss still trails uncond)
- At what scale does hierarchy become necessary again?
- Joint vs staged generation — when does two-stage break down?

Track in [EXPERIMENTS.md](EXPERIMENTS.md) and [STATE.md](STATE.md).
