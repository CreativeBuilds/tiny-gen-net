# Width-Growth + Low-Rank Correction — 1-100M Scale-up Plan (v1)

**Author (bootstrap)**: Ze (temporary until ForgeCritic reviews)
**Date**: 2026-06-23
**Trajectory**: knowledge_base/trajectories/width_growth_correction.md
**Status**: REVISED v1.1 (2026-06-23) — exact function-preservation is now
DEMONSTRATED (Task 004: ActiveLayerNorm, FP diff 4.77e-07 at d32→d64 trained).
Ready for the scale arm on ONE ≤$3.29/hr H100.

> **v1.1 update.** The original draft assumed zero-pad width growth was exactly
> function-preserving. Task 003 showed it was NOT (full-width LayerNorm variance
> shrink, 1.26 logit diff). Task 004 fixed it with `ActiveLayerNorm` (normalize
> over original Ds dims only) — growth is now exact to float epsilon. The plan
> below stands; "naive zero-pad width growth" in Arm 2 now means the
> ActiveLayerNorm-corrected operator (exact at init), and Arm 2 vs Arm 3 isolates
> the value of the *learned* low-rank correction on top of an already-exact start.

## 1. Experiment Design (Minimal)

Goal: Measure whether the 4× FLOP savings observed at toy scale (H=16→64) survives at 1-100M scale on real text.

### Models
- Source: ~25M param TinyStories-style decoder-only transformer (d_model=512, n_layer=8, n_head=8)
- Grown target: ~80-100M (grow d_model 512→1024 at layer 4, keep depth fixed)
- Correction: rank-32 LoRA-style delta on the grown linear projections (Q/K/V/O/FF)

### Data
- TinyStories (or first 2B tokens of SlimPajama) — ~2-4B tokens total training budget
- Tokenizer: GPT-2 50k vocab (reuse existing)

### Training Budget (FLOPs, not steps)
- Source training: 25M model × 4e18 FLOPs (approx 80k steps @ bs=256, ctx=1024)
- Grown model training: target ~1.6e19 FLOPs ceiling (allows direct comparison)
- Evaluation: fixed 10M token validation slice, report CE every 500 steps

### Arms (matched total FLOPs)
1. Random init 100M baseline
2. Naive zero-pad width growth + continue pretrain
3. Grown + rank-32 correction (learned on source then frozen or lightly tuned)
4. µP re-init baseline (if easy to add)
5. Optional: continued pretrain of source then naive growth (ablation)

### Metrics
- CE vs total training FLOPs (primary)
- Effective FLOP reduction factor at target CE
- Wall time / GPU-hours (secondary)
- Correction norm and dead-unit revival statistics (instrumentation)

## 2. Files to Change / Add

- `experiments/phase_grow_width_realtext.py` — new single-file runner (copy style from phase_grow_gate_sweep.py)
- `src/growth/width_growth_realtext.py` — thin wrapper around existing width_growth.py with real-text dataset + larger model factory
- `src/models/variable_transformer.py` — (if missing) variable-width decoder-only (priority 1)
- Update `EXPERIMENTS.md` + `STATE.md` after first validation run

## 3. Rough FLOP Math

Toy result: naive eventually negative delta; grown_corr stays positive.
At scale the same dead-unit mechanism should apply if residual width growth still creates unused capacity. Expected win: 3-5× reduction in total FLOPs to reach a given CE on the 100M model.

## 4. Risks & Instrumentation
- Risk: correction rank too small → instrument correction norm per layer
- Risk: dataset shift (synthetic→real) washes out signal → keep source and target on identical data distribution
- Risk: H100 rental cost → start with 1× H100 validation run (1-2h) before full sweep

## 5. Proposed Next Prompt (Task 002)
"Implement the variable transformer and the minimal experiment runner. Run a 1-seed smoke test on CPU/GPU (tiny sizes) to verify code path, then report any blockers. Do not rent pods yet."

## Acceptance Criteria for This Plan
- ForgeCritic must read and either accept or produce a revised v2.
- Ze then merges the accepted plan into knowledge_base/trajectories/.
- CB (or autonomous budget) authorizes first $X of H100 spend before any pod creation.