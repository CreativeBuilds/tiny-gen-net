# Project State — tiny-gen-net

*Last updated: 2026-06-20*

## Current Phase

**Phase 5c — fine aux sweep failed; mid remains best strong_plan balance**

`phase5c_fine` (jepa=0.11, cons=0.20) landed in-grid near neutral (−0.017) but extrap collapsed (−0.623). Aux tuning between light and mid is **not** monotonic — `mid` still best balanced strong_plan config.

## Phase 5c Variant Comparison (H100)

| Variant | Warm-start | Loss profile | Planning | In-grid Δ | Extrap Δ |
|---------|------------|--------------|----------|-----------|----------|
| v1 | ❌ | v1 cosine | default | −0.17 | −0.065 |
| v2 refine | ✅ | strong aux | h-inject | −0.89 | −0.11 |
| ws ablation | ✅ | v1 cosine | default | −0.41 | **−0.023** |
| **light** | ✅ | jepa=0.1, cons=0.15 | **strong_plan** | **+0.042** | −0.149 |
| **fine** | ✅ | jepa=0.11, cons=0.20 | **strong_plan** | −0.017 | −0.623 |
| **mid** | ✅ | jepa=0.12, cons=0.25 | **strong_plan** | −0.072 | **−0.115** |
| light-noplan | ✅ | jepa=0.1, cons=0.15 | default | −0.69 | −0.214 |
| hybrid | ✅ | jepa=0.15, cons=0.35 | **strong_plan** | −0.029 | −0.564 |

Random CE ≈ 4.28. Phase 5a task-nano ref: in-grid Δ **+0.73**.

## Phase 5c Status

| Item | Status |
|------|--------|
| Beat random in-grid | ✅ **+0.042** (`phase5c_light` only) |
| Beat random extrap | ❌ best: ws ablation −0.023 |
| Best strong_plan balance | ✅ **mid** (fine sweep rejected) |
| Fine sweep (light↔mid) | ❌ failed — extrap collapse |

## Key Results (phase5c_fine, seed 42, H100)

- **In-grid zero-shot Δ:** −0.017 (vs light +0.042, mid −0.072)
- **Extrap zero-shot Δ:** −0.623 (vs light −0.149, mid −0.115)
- **Config:** `jepa_w=0.11`, `cons_w=0.20`, `strong_plan=True`, `plan_scale=0.1`
- **Worst extrap preset:** nano_16m holdout CE 5.49 vs random 4.31

## Open Questions

- Aux weight axis is **not** smoothly monotonic between light/mid/fine on extrap — `cons_w=0.20` may hit an unstable region.
- `fine` in-grid near-neutral (−0.017) suggests the in-grid/extrap tradeoff is sharp, not a continuous Pareto curve.
- Extrap gains may require anchor scale (20–30M) rather than further aux micro-tuning.

## Immediate Next Steps

1. **[NEW 06-23] Width-growth FP gap is LayerNorm, not the copy.** Transformer
   zero-pad width growth is only approx function-preserving (1.26 logit diff on a
   trained d32→d64 model) because full-width LayerNorm's variance shrinks when new
   dims are zero. MLP growth stays exact (no norm). See EXPERIMENTS.md 06-23.
2. **Local fix before any pod:** add an RMSNorm/segment-norm-over-original-dims
   variant of TxBlock, re-run `phase_grow_width_realtext_smoke.py`, confirm FP
   diff < 1e-3. Only THEN re-rent ONE H100 (≤$3.29/hr) for the scale arm.
3. `plans/width_scaleup_v1.md` assumed exact preservation — revise its premise.

### Prior (Phase 5c, paused)

1. **Stop aux micro-sweeps** — light and mid bracket the useful region; fine is a dead end.
2. Add one **20–30M warm-start anchor** on **`mid`** profile (separate ablation) — test whether larger anchor closes extrap gap vs ws ablation.
3. Keep **`light`** as in-grid reference; use **`mid`** as strong_plan balance baseline.
