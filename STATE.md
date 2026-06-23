# Project State — tiny-gen-net

*Last updated: 2026-06-23 22:00 UTC — Task 008 (4x growth ratio) COMPLETE*

## Current Phase

**Iterative width growth — scaling to larger growth ratios. COMPLETE through 4x.**

Function-preserving width growth has been validated for both single-step and
iterative/multi-step at scale, across 2x and 4x growth ratios:

### Task 006 (single-step d512→d1024, 2x)
- FP holds: 8.58e-06 < 1e-3 at 42M params
- Grown+corr beats random by 18.5% (CE 0.218 vs 0.267, 2000 steps)
- Correction layer helps (+8.1% vs grown alone)

### Task 007 (iterative d512→d768→d1024, 2x)
- FP holds at both steps: step1=7.63e-06, step2=6.68e-06 (both < 1e-3)
- Iterative beats single-step by 15.4%: CE 0.2255 vs 0.2667
- Iterative beats random by 36.2%
- Correction HURTS in iterative case (0.2389 vs 0.2255)

### Task 008 (iterative d256→d512→d1024, 4x)
- FP holds at 4x growth: step1=0.0 (exact), step2=9.54e-06 (both < 1e-3)
- Iterative beats single-step by 10.5%: CE 0.2696 vs 0.3011
- Iterative beats random by 26.6%
- Correction STILL hurts (0.2877 vs 0.2696)
- Intermediate d512 training with active_dim=256 WORSENED CE (0.284→0.308)
- Iterative advantage diminishes with growth ratio (15.4% at 2x → 10.5% at 4x)

Pod `c8ys2brrllnzad` STOPPED. Budget $0/hr.

**Key cross-task insight:** Iterative growth consistently beats single-step across
growth ratios, but the advantage shrinks at larger ratios (weaker source, active_dim
constraint limits intermediate training). Correction consistently hurts iterative.

**Next decision points:**
1. Longer training (does the gap widen or narrow at 5000+ steps?)
2. 3-step iterative growth (d512→d640→d768→d1024) — more granular steps
3. Real TinyStories corpus (synthetic may limit generalization signal)
4. Depth growth (add layers — prior work showed limited results, revisit?)
5. Width+depth joint growth
6. Correction ablation (when does correction help vs hurt? Helps single-step, hurts iterative)

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

## Key Results Summary (width-growth, all scale runs)

### Task 006 — Single-step d512→d1024 (2000 steps/arm)
| Metric | Value |
|--------|-------|
| Source (d512) params | 10,663,944 |
| Target (d1024) params | 42,299,400 |
| FP max-abs diff | 8.58e-06 (< 1e-3) |
| Random final CE | 0.2673 |
| Grown final CE | 0.2371 (−11.3%) |
| Grown+corr final CE | 0.2180 (−18.5%) |
| Correction B norm | 1.086 |
| Elapsed | 1002 sec (~16.7 min) |

### Task 007 — Iterative d512→d768→d1024 (1000 d1024 steps/arm)
| Metric | Value |
|--------|-------|
| Source (d512) params | 10,663,944 |
| Mid (d768) params | 23,860,232 |
| Target (d1024) params | 42,299,400 |
| FP step1 (d512→d768) | 7.63e-06 (< 1e-3) |
| FP step2 (d768→d1024) | 6.68e-06 (< 1e-3) |
| Random final CE | 0.3533 |
| Single-step final CE | 0.2667 (−24.5%) |
| **Iterative final CE** | **0.2255 (−36.2%)** |
| Iterative+corr final CE | 0.2389 (−32.4%) |
| Elapsed | 813 sec (~13.6 min) |

### Task 008 — Iterative d256→d512→d1024 (1000 d1024 steps/arm, 4x growth)
| Metric | Value |
|--------|-------|
| Source (d256) params | 2,710,536 |
| Mid (d512) params | 10,663,944 |
| Target (d1024) params | 42,299,400 |
| FP step1 (d256→d512) | 0.0 (exact) |
| FP step2 (d512→d1024) | 9.54e-06 (< 1e-3) |
| FP single-step | 3.81e-06 (< 1e-3) |
| Random final CE | 0.3672 |
| Single-step final CE | 0.3011 (−18.0%) |
| **Iterative final CE** | **0.2696 (−26.6%)** |
| Iterative+corr final CE | 0.2877 (−21.6%, WORSE than iterative) |
| Elapsed | 717 sec (~12 min) |

## Immediate Next Steps

1. **[DONE 06-23] Width-growth EXACTLY function-preserving** — Task 004.
2. **[DONE 06-23] 3-arm scale runner** — Task 005.
3. **[DONE 06-23] Single-step scale run d512→d1024** — Task 006.
4. **[DONE 06-23] Iterative growth d512→d768→d1024 (2x)** — Task 007.
5. **[DONE 06-23] 4x growth ratio d256→d512→d1024** — Task 008.
6. **[NEXT] Decide next experiment direction** — longer training, 3-step iterative, real corpus, or correction ablation.
