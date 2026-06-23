# Project State — tiny-gen-net

*Last updated: 2026-06-23 21:02 UTC — Scale run COMPLETE*

## Current Phase

**Width-Growth scale validation — COMPLETE. Function preservation + training advantage verified at d512→d1024.**

The function-preserving width-growth operator (exact zero-pad + ActiveLayerNorm +
zero-init low-rank correction) has been validated end-to-end at scale:

- **FP holds**: d512→d1024, 8 layers, 10.66M→42.30M params, FP max-abs logit diff
  **8.58e-06 < 1e-3** on H100 (toy d32→d64 was 4.77e-07).
- **Grown+corr beats random**: 3-arm matched-FLOP comparison at d1024:
  - Random: CE 0.2673
  - Grown: CE 0.2371 (−11.3% vs random)
  - Grown+corr (rank32): CE **0.2180** (−18.5% vs random, −8.1% vs grown)
- **Knowledge transfers**: grown arm starts at source CE (0.2427), already below
  random's final CE (0.2673) — source model's knowledge transfers perfectly through
  width growth.
- Correction layer actively learns (B norm 0→1.086).

Pod `nj54tli4w9xpdj` STOPPED (training complete, no idle burn). Budget: $0/hr.

**Next decision points:**
1. Push the scale results to origin, commit EXPERIMENTS.md + STATE.md updates.
2. Next experiment direction: (a) larger growth ratio (d256→d1024, 4x), (b) depth
   growth (add layers), (c) longer training to see if gap widens or narrows, (d)
   real TinyStories corpus instead of synthetic fallback, (e) iterative growth
   (d512→d768→d1024) to test multi-step preservation.

---

## (prior) Phase 5c — fine aux sweep failed; mid remains best strong_plan balance

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

## Key Results (width-growth scale run, 2026-06-23)

| Metric | Value |
|--------|-------|
| Source (d512) params | 10,663,944 |
| Target (d1024) params | 42,299,400 |
| FP max-abs diff @ init | 8.583e-06 (< 1e-3) |
| Source val CE (pretrain) | 0.2427 |
| Random final CE | 0.2673 |
| Grown final CE | 0.2371 (−11.3%) |
| Grown+corr final CE | 0.2180 (−18.5%) |
| Correction B norm | 1.086 |
| Elapsed | 1002 sec (~16.7 min H100) |

## Immediate Next Steps

1. **[DONE 06-23] Width-growth EXACTLY function-preserving** — Task 004 (ActiveLayerNorm, FP 4.77e-07 at toy).
2. **[DONE 06-23] 3-arm scale runner** — Task 005 (CPU smoke passed).
3. **[DONE 06-23] Scale run d512→d1024** — Task 006 COMPLETE. FP 8.58e-06, grown+corr beats random by 18.5%.
4. **[NEXT] Commit + push results. Decide next experiment direction.**
