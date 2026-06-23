# Experiment Log — tiny-gen-net

Record every experiment here. One row per run (or per meaningful variant).

## Template

Copy this block for new entries:

```markdown
### YYYY-MM-DD — [Phase N] Short title

| Field | Value |
|-------|-------|
| **Phase** | 0 / 1 / 2 / ... |
| **Script** | `experiments/...` or `scripts/...` |
| **Seed** | 42 |
| **Key metrics** | final_loss=..., val_acc=..., vs_random=... |
| **Outcome** | success / inconclusive / failed |
| **Notes** | What worked, what didn't, follow-ups |
| **Artifacts** | `checkpoints/...` |
```

---

## Experiments

### 2026-06-23 — [Phase 6 / Growth] EXACT width growth via ActiveLayerNorm (Task 004) ✅

| Field | Value |
|-------|-------|
| **Phase** | 6 (Growth) |
| **Script** | `experiments/phase_grow_width_realtext_smoke.py` |
| **Seed** | 1234 |
| **Source → Target** | d32/h4/L3 (17,048 p) → d64/h8/L3 (64,808 p), head_dim=8 const |
| **FP max-abs logit diff BEFORE** | **1.255** (full-width LayerNorm — the Task 003 bug) |
| **FP max-abs logit diff AFTER** | **4.77e-07** (tol 1e-3) → **exactly function-preserving** |
| **Correction** | identity@step0 = 0.0; B-norm 0 → 0.201 over 50 steps; no NaN |
| **Outcome** | **success** — re-rent gate PASSED |
| **Author** | ze-main (ForgeCritic stalled on Task 004 across cycles; local CPU, $0) |

**Fix.** Task 003 isolated the function-preservation gap to `nn.LayerNorm` over the
full grown width: zero-padded new dims shrink the per-token variance, re-scaling
the copied dims (1.26 logit diff on a trained source). Static gamma rescales could
not fix it. Task 004 adds **`ActiveLayerNorm`** to `TxBlock` — a LayerNorm that
computes its mean/variance over only the first `active_dim` (= original Ds) channels
while keeping per-channel affine across the full width. `grow_tx_width` sets each
grown norm's `active_dim = Ds`. With new dims zero (gamma=beta=0 there), this
reproduces the source's normalization **exactly** → growth is now exact (4.77e-07,
float epsilon). Backward-compatible: a fresh model has `active_dim == d_model`,
numerically identical to `nn.LayerNorm` (verified diff 4.77e-07), so prior phases
do not regress. The MLP growth operator was already exact (no norm).

**Significance.** This completes the exact-function-preserving width-growth operator
for transformers — the prerequisite for the 1–100M scale arm. Combined with the
prior depth-growth + GATE-GROW results, the growth toolkit now has an exact width
primitive on the residual stream. Next: revise `plans/width_scaleup_v1.md` (drop the
old "exact preservation assumed" premise — it is now *demonstrated*) and run the
scale arm on one H100.

---

### 2026-06-23 — [Phase 6 / Growth] Transformer WIDTH growth + low-rank correction (CPU smoke)

| Field | Value |
|-------|-------|
| **Phase** | 6 (architecture growth) |
| **Script** | `experiments/phase_grow_width_realtext_smoke.py`, operator `src/growth/tx_width_growth.py` |
| **Seed** | 1234 |
| **Hardware** | CPU (local, $0 — pod `vgt6p4yoar259l` was lost 404 before Task 002 reported; re-rent gated on this smoke) |
| **Setup** | VariableTinyTransformer d=32→64 (n_head 4→8, head_dim=8 const, L=3, ctx=8). Net2Wider zero-pad copy of tok/pos/MHA(in_proj QKV, out_proj)/FF/head; LayerNorm copied with new gamma=beta=0. Source lightly pre-trained (20 steps) before growth. |
| **Key metrics** | src=17,048p → tgt=64,808p; function-preservation max-abs logit diff **1.26** (tol 5e-2 → **NOT** preserved on trained source; 0.014 on *untrained* source); correction identity diff @step0 = 0.0; B-norm 0→0.073 (grad 0.016) → correction learns; no NaN over 50 steps |
| **Outcome** | inconclusive — operator runs clean, but **NOT exactly function-preserving** |
| **Notes** | **Key finding:** unlike the MLP width-growth (`src/growth/width_growth.py`, which IS exact — no normalization), the transformer is only *approximately* function-preserving under naive zero-pad. Root cause isolated by diagnostic: embeddings copy exactly (first Ds identical, new dims=0), but **full-width LayerNorm** breaks it — its denominator `sqrt(var+eps)` is computed over all Dt dims; padding with zeros shrinks per-token variance, and the residual **scales with activation magnitude** (0.014 untrained → 1.26 trained). No static per-channel gamma rescale fixes it (a sqrt(Dt/Ds) attempt made it *worse*). Exact transformer width growth needs an architectural change: segment/sub-vector LayerNorm or RMSNorm restricted to the original dims, or LN-free residual injection. LowRankCorrection (zero-init B = identity at step0) is verified working and is the natural vehicle to *learn out* the LN-induced gap. |
| **Decision** | Do **NOT** re-rent H100 yet. The scale run from `plans/width_scaleup_v1.md` assumed function preservation; that premise is false for this architecture. Next increment is a cheap local fix (RMSNorm-on-original-dims variant), re-measure FP diff, THEN re-rent for scale. |
| **Artifacts** | `~/.hermes-chats/tiny-gen-net/status/forgecritic.json` (task 003, status=complete_with_caveats) |

### 2026-06-20 — [Phase 5c] Fine aux sweep (light↔mid) + strong_plan — H100

| Field | Value |
|-------|-------|
| **Phase** | 5c |
| **Script** | `PROFILE=fine RETRAIN_PROG=1 RETRAIN_ANCHORS=1 ./scripts/runpod_phase5c.sh phase5c_fine` |
| **Seed** | 42 |
| **Setup** | Fine aux (`jepa_w=0.11`, `cons_w=0.20`) + `strong_plan=True`; warm 12m/16m anchors |
| **Key metrics** | in-grid Δ=**−0.017**; extrap Δ=**−0.623** (CE ~4.90); ft100 in-grid Δ=+0.027 |
| **Outcome** | **failed** — in-grid near neutral but extrap collapsed; not a better operating point |
| **Notes** | vs light: in-grid +0.042→−0.017 (still close), extrap −0.149→−0.623 (much worse). vs mid: in-grid −0.072→−0.017 (better), extrap −0.115→−0.623 (catastrophic). Aux axis non-monotonic in this band; `mid` remains best strong_plan balance |
| **Artifacts** | `metrics_phase5c_fine.json`, `prog_phase5c_fine.pt` |

### 2026-06-20 — [Phase 5c] Mid aux weights + strong_plan — H100

| Field | Value |
|-------|-------|
| **Phase** | 5c |
| **Script** | `PROFILE=mid RETRAIN_PROG=1 RETRAIN_ANCHORS=1 ./scripts/runpod_phase5c.sh phase5c_mid` |
| **Seed** | 42 |
| **Setup** | Intermediate aux (`jepa_w=0.12`, `cons_w=0.25`) + `strong_plan=True`; warm 12m/16m anchors |
| **Key metrics** | in-grid Δ=**−0.072**; extrap Δ=**−0.115** (CE ~4.39); ft100 in-grid Δ=−0.015 |
| **Outcome** | **partial success** — best extrap among strong_plan variants; slight combined-balance win vs light |
| **Notes** | vs light: extrap −0.149→−0.115 (+23%), in-grid +0.042→−0.072. vs ws ablation: extrap still worse (−0.023). \|Δ\| sum: mid 0.19 vs light 0.19 |
| **Artifacts** | `metrics_phase5c_mid.json`, `prog_phase5c_mid.pt` |

### 2026-06-20 — [Phase 5c] Hybrid (v1 aux + strong_plan) — H100

| Field | Value |
|-------|-------|
| **Phase** | 5c |
| **Script** | `PROFILE=hybrid RETRAIN_PROG=1 RETRAIN_ANCHORS=1 ./scripts/runpod_phase5c.sh phase5c_hybrid` |
| **Seed** | 42 |
| **Setup** | v1 aux (`jepa_w=0.15`, `cons_w=0.35`) + `strong_plan=True`, `plan_scale=0.1`; warm 12m/16m anchors |
| **Key metrics** | in-grid Δ=**−0.029**; extrap Δ=**−0.564** (CE ~4.84); ft100 in-grid Δ=+0.038 |
| **Outcome** | **failed** — stronger aux + strong_plan hurt both vs light; extrap worst in table |
| **Notes** | Hypothesis rejected: v1 aux weights do not recover extrap when paired with strong_plan. Best in-grid remains `light`; best extrap remains `ws ablation` |
| **Artifacts** | `metrics_phase5c_hybrid.json`, `prog_phase5c_hybrid.pt` |

### 2026-06-20 — [Phase 5c] Light losses, no strong planning — H100

| Field | Value |
|-------|-------|
| **Phase** | 5c |
| **Script** | `PROFILE=light-noplan RETRAIN_PROG=1 RETRAIN_ANCHORS=1 ./scripts/runpod_phase5c.sh phase5c_light_noplan` |
| **Seed** | 42 |
| **Setup** | Same light aux as `phase5c_light` but **`strong_plan=False`** (only variable changed) |
| **Key metrics** | in-grid Δ=**−0.69**; extrap Δ=**−0.214** (CE ~4.49) |
| **Outcome** | **failed / hypothesis rejected** — removing strong_plan did not recover extrap; in-grid collapsed |
| **Notes** | vs light: in-grid +0.042→−0.69, extrap −0.149→−0.214. strong_plan is required for in-grid win; extrap gap likely from lighter aux weights, not plan injection |
| **Artifacts** | `metrics_phase5c_light_noplan.json`, `prog_phase5c_light_noplan.pt` |

### 2026-06-20 — [Phase 5c] Light losses + strong planning — H100

| Field | Value |
|-------|-------|
| **Phase** | 5c |
| **Script** | `PROFILE=light RETRAIN_PROG=1 RETRAIN_ANCHORS=1 ./scripts/runpod_phase5c.sh phase5c_light` |
| **Seed** | 42 |
| **Setup** | Light aux (`jepa_w=0.1`, `cons_w=0.15`, cosine cons); `strong_plan=True` (scale+layer-frac+plan_mix residual in every progressive step); warm 12m/16m anchors |
| **Key metrics** | in-grid Δ=**+0.042**; extrap Δ=**−0.149** (CE ~4.42); ft100 in-grid Δ=+0.041 |
| **Outcome** | **partial success** — first variant beating random in-grid; extrap worse than ws ablation |
| **Notes** | Lighter losses + direct plan conditioning fixed in-grid. Extrap regression suggests strong_plan may overfit training-scale cues. Next: ablate strong_plan with same light losses |
| **Artifacts** | `metrics_phase5c_light.json`, `prog_phase5c_light.pt`, `checkpoints/plots/phase5c_scale/` |

### 2026-06-20 — [Phase 5c] Variant comparison (H100, updated)

| Variant | Warm-start | Loss profile | Planning | In-grid Δ | Extrap Δ |
|---------|------------|--------------|----------|-----------|----------|
| v1 | 0 tensors | v1 cosine | default | −0.17 | −0.065 |
| v2 refine | 77–101 tensors | strong aux | h-inject | −0.89 | −0.11 |
| ws ablation | 77–101 tensors | v1 cosine | default | −0.41 | **−0.023** |
| **light** | 77–101 tensors | jepa=0.1, cons=0.15 | **strong_plan** | **+0.042** | −0.149 |
| **fine** | 77–101 tensors | jepa=0.11, cons=0.20 | **strong_plan** | −0.017 | −0.623 |
| **mid** | 77–101 tensors | jepa=0.12, cons=0.25 | **strong_plan** | −0.072 | **−0.115** |
| light-noplan | 77–101 tensors | jepa=0.1, cons=0.15 | default | −0.69 | −0.214 |
| hybrid | 77–101 tensors | jepa=0.15, cons=0.35 | **strong_plan** | −0.029 | −0.564 |

Phase 5a task-nano baseline for reference: in-grid Δ **+0.73**.

### 2026-06-20 — [Phase 5c] Warm-start-only ablation — H100

| Field | Value |
|-------|-------|
| **Phase** | 5c |
| **Script** | `ABLATION=warmstart-only ./scripts/runpod_phase5c.sh` → `phase5c_ws_ablation` |
| **Seed** | 42 |
| **Setup** | Fixed partial warm-start (12m/16m anchors) + **v1 loss weights** (cosine cons, jepa_w=0.15, cons_w=0.35, plan_scale=0); isolates warm-start vs v2 strong losses |
| **Key metrics** | in-grid Δ=**−0.41**; extrap Δ=**−0.023** (CE ~4.30) |
| **Outcome** | **partial success** — best extrap so far; confirms v2 loss regression |
| **Notes** | vs v1: extrap improved (−0.065→−0.023), in-grid worse (−0.17→−0.41). Warm-start helps holdout when reconstruction not dominated by strong aux losses |
| **Artifacts** | `metrics_phase5c_ws_ablation.json`, `prog_phase5c_ws_ablation.pt` |

### 2026-06-20 — [Phase 5c] v2 refine — H100 (strong losses)

| Field | Value |
|-------|-------|
| **Phase** | 5c |
| **Script** | `RETRAIN_PROG=1 RETRAIN_ANCHORS=1 ./scripts/runpod_phase5c.sh phase5c_v2` |
| **Seed** | 42 |
| **Setup** | Fixed warm-start + stronger scale-consistency (align+Δ-pred) + h GRU inject + cons_w=0.5 |
| **Key metrics** | in-grid Δ=**−0.89**; extrap Δ=**−0.11** |
| **Outcome** | **failed** — regression vs v1 and ws ablation |
| **Notes** | Warm-start worked (4.8M/6.4M params transferred) but aux losses hurt reconstruction |
| **Artifacts** | `metrics_phase5c_v2.json`, `prog_phase5c_v2.pt` |

### 2026-06-20 — [Phase 5c] Scale-consistency + anchors — H100 cloud

| Field | Value |
|-------|-------|
| **Phase** | 5c |
| **Script** | `experiments/phase5c_scale_hierarchy.py --cloud-resume --retrain-prog --tag phase5c_v1` |
| **Seed** | 42 |
| **Setup** | 15 Phase 5a weights + 2 anchors (12m/16m); 1000 train steps (prog+JEPA+scale-cons); eval in-grid + holdout |
| **Key metrics** | in-grid zero Δ=**−0.17**; extrap zero Δ=**−0.065** (CE ~4.34); vs 5b extrap Δ −26.8 |
| **Outcome** | **partial success** — extrap no longer broken; still below random in-grid |
| **Notes** | Anchor warm-start matched **0 tensors** (256-d donor → 384-d target); fix needed for true bootstrap |
| **Artifacts** | `metrics_phase5c_v1.json`, `prog_phase5c_v1.pt`, `weights_with_anchors.pt` |

### 2026-06-20 — [Phase 5c] Scale-consistency + block planning + anchors (smoke)

| Field | Value |
|-------|-------|
| **Phase** | 5c |
| **Script** | `experiments/phase5c_scale_hierarchy.py --smoke` |
| **Seed** | 42 |
| **Setup** | ScaleAwareProgressiveTrainer (prog + JEPA + cosine scale-consistency); block_plan MLP; global h in every sub-chunk decode; anchor warm-start helper |
| **Key metrics** | 60 train steps ~8 it/s MPS; total loss wiring OK |
| **Outcome** | **infrastructure success** — H100 cloud run next |
| **Notes** | Fixes layer_latent device for MPS; `--collect-anchors` adds 12m/16m warm-start on cloud |
| **Artifacts** | `src/nano_warmstart.py`, `prog_phase5c_smoke.pt`, `checkpoints/plots/phase5c_scale/` |

### 2026-06-20 — [Phase 5b] Hierarchical progressive — H100 full run

| Field | Value |
|-------|-------|
| **Phase** | 5b |
| **Script** | `experiments/phase5b_hierarchical_progressive.py --cloud-resume --tag phase5b_v1` |
| **Seed** | 42 |
| **Setup** | 15 Phase 5a weights; 800 prog steps; eval in-grid (1m–8m) + holdout (12m/16m/30m) |
| **Key metrics** | in-grid zero Δ=**−0.22**; extrap zero Δ=**−26.8** (CE ~31); random CE≈4.28 |
| **Outcome** | **failed** (negative signal — scaffolding insufficient) |
| **Notes** | Fixed batch-dim + `logical_layer_bounds` double-count bug before final run; confirms need for 5c |
| **Artifacts** | `metrics_phase5b_v1.json`, `prog_phase5b_v1.pt` |

### 2026-06-19 — [Phase 5a] Real text + sentence embeddings + nano scale (cloud-first)

| Field | Value |
|-------|-------|
| **Phase** | 5a |
| **Script** | `experiments/phase5a_nano_text.py` |
| **Seed** | 42 |
| **Setup** | Tiny Shakespeare char LM; MiniLM sentence task embeds; NanoTransformer 1M–10M; sub-chunk weight hypernet |
| **Key metrics (partial local smoke)** | task_nano Δ=**+0.43**; nano_jepa Δ=+0.35; random CE≈4.23 |
| **Outcome** | **infrastructure success** — pipeline wired; full run pending on RunPod |
| **Notes** | Local weight-gen training is 30+ min on MPS; use `--smoke` locally, `--cloud` on RunPod |
| **Artifacts** | `scripts/runpod_*.sh`, `metrics_phase5a_smoke.json`, `checkpoints/weights/nano_multi_v1/` |

### 2026-06-20 — [Phase 5b] Hierarchical progressive generator (smoke)

| Field | Value |
|-------|-------|
| **Phase** | 5b |
| **Script** | `experiments/phase5b_hierarchical_progressive.py --smoke` |
| **Seed** | 42 |
| **Setup** | ScaleEmbedding + HighLevelScaleJEPA + ProgressiveWeightGenerator; nano_spec grid to 256M; 2 collect + 40 prog steps |
| **Key metrics** | wiring OK; prog train ~5 it/s MPS |
| **Outcome** | **infrastructure success** — full H100 run pending |
| **Notes** | Reuses Phase 5a flat hypernet path for comparison; progressive path for extrapolation eval on 12m/16m/30m holdouts |
| **Artifacts** | `src/hierarchical_jepa.py`, `src/progressive_generator.py`, `prog_phase5b_smoke.pt` |

---

### 2026-06-20 — [Phase 5a] Cloud resume — Shakespeare H100

| Field | Value |
|-------|-------|
| **Phase** | 5a |
| **Script** | `experiments/phase5a_nano_text.py --cloud-resume --tag phase5a_v1` |
| **Seed** | 42 |
| **Setup** | 15 collected nano specs (reuse weights_raw.pt); cloud-first hypernet steps; ft100 eval |
| **Key metrics** | task_nano zero Δ=**+0.73**; nano_cond Δ=+0.19; nano_jepa Δ=+0.04; arch_match=0.69; random CE=4.24 |
| **Outcome** | **success** |
| **Notes** | RunPod H100; optimized sub-chunk decode + profile fix; cond ~3 it/s after resume |
| **Artifacts** | `checkpoints/arch_gen/metrics_phase5a_v1.json` |

---

| Field | Value |
|-------|-------|
| **Phase** | 4b refine |
| **Script** | `experiments/phase4b_tx_refine.py` |
| **Seed** | 42 |
| **Hyperparameters** | 81 augmented train pairs; match_w=0.7; ref tx d=48/L=3/H=4; 30 ref weights; tx-JEPA 1200 + tx-diff 800 steps |
| **Key metrics** | match=**0.73** (+0.21 vs v1); task_tx zero=**0.49**; ref task-tx zero=**0.31** FT100=**0.28** |
| **Outcome** | **success** — steering improved; task-tx init best on fixed ref |
| **Artifacts** | `task_tx_phase4b_v2.pt`, `checkpoints/plots/phase4b_refine/` |

#### Steering v2 vs v1

| Metric | v2 | v1 | Δ |
|--------|----|----|---|
| Arch-task match | **0.73** | 0.52 | **+0.21** |
| Task-tx zero (25 tasks) | **0.49** | 1.51 | −1.02 |
| Large/capacity hit rate | d=96 ✓ | d=48 ✗ | fixed |

#### Fixed ref tx init (20 samples, ref d=48/L=3/H=4)

| Init | Zero | FT100 | vs MLP ref |
|------|------|-------|------------|
| **Task-tx** | **0.31** | **0.28** | beats MLP task (1.76) |
| Tx hybrid | 2.06 | 0.35 | hybrid FT100=0.99 (MLP) |
| Tx JEPA | 2.06 | 0.88 | JEPA zero=1.84 (MLP) |
| Random | 2.08 | — | — |

#### Analysis

**Steering fix works** — tx-calibrated tiers (`large`→d≥96), `enrich_tx_task()`, and 81 augmented synonym pairs raised match from 0.52→0.73. "Large wide capacity" now hits d=96,L=2,H=16 (match=1.0).

**Task-tx dominates fixed ref** — canonical task string + task weight hypernet yields zero=0.31, FT100=0.28. Best overall init path for transformers so far.

**Tx-JEPA immature** — 30 ref samples insufficient for JEPA zero-shot (≈random). Hybrid FT100=0.35 promising but zero-shot weak. MLP JEPA/hybrid still better on fixed MLP; tx needs more collection.

**Remaining gap** — "large deep reasoning" sometimes picks d=32,L=5 (match=0.5). Deep/large keyword conflict under param cap.

---

### 2026-06-19 — [Phase 4b] Tiny transformer scale-up

| Field | Value |
|-------|-------|
| **Phase** | 4b |
| **Script** | `experiments/phase4b_tiny_transformer.py` |
| **Seed** | 42 |
| **Hyperparameters** | 30 diverse tx specs collected (350 train steps each); TxCond + TaskTx arch/weights; 25 eval tasks; ctx=8, ff=d/2, grid=56 specs (<100k params) |
| **Key metrics** | task_tx zero=**1.51**, Δ=**+0.57**, FT100=**0.31**, match=0.52 |
| **Outcome** | **success** — pipeline works on richer arch; steering weaker than MLP |
| **Artifacts** | `checkpoints/arch_gen/task_tx_phase4b_v1.pt`, `checkpoints/plots/phase4b_tx/` |

#### Comparison table

| Metric | Task-tx | Tx cond (in-dist) | MLP task v2 | MLP random | Tx random |
|--------|---------|-------------------|-------------|------------|-----------|
| Zero-shot loss | **1.51** | 0.61 | 1.76 | 2.08 | 2.08 |
| Zero Δ vs random | **+0.57** | +1.47 | +0.32 | — | — |
| FT100 loss | **0.31** | — | 1.11 | — | — |
| Arch-task match | 0.52 | — | **0.92** | — | — |

#### Example task → transformer configs

| Task | Generated (d, L, H) | Match |
|------|---------------------|-------|
| small efficient lightweight transformer | (64, 3, 2) | 1.0 |
| efficient shallow transformer | (64, 2, 8) | 1.0 |
| medium deep char predictor transformer | (32, 6, 4) | 1.0 |
| large deep reasoning transformer | (48, 4, 2) | 0.0 ← "large" maps to d=48 |
| deep balanced reasoning transformer | (48, 6, 8) | 0.5 |

#### Analysis

**Pipeline generalizes** — same cond hypernet + task conditioning pattern works for causal transformers. Task-tx zero-shot beats MLP task conditioning despite weaker keyword match.

**Cond hypernet strong in-distribution** — tx_cond zero=0.61 on collected specs (weights seen during training). Held-out eval needed for honest cond baseline.

**Steering gap** — 100k param cap shrinks d_model range; "large/capacity" keywords can't map to d≥128 with deep layers. Arch generator defaults to d=48–64 for most tasks.

**FT signal** — FT100=0.31 vs MLP 1.11 suggests transformers benefit more from good weight inits on this task.

**Challenges** — sequence eval (ctx=8 windows), larger flat weight vectors, no JEPA/hybrid for variable tx yet.

---

### 2026-06-19 — [Phase 4] Task conditioning refine (match-aware + rich vocab)

| Field | Value |
|-------|-------|
| **Phase** | 4 refine |
| **Script** | `experiments/phase4_task_cond_refine.py` |
| **Seed** | 42 |
| **Hyperparameters** | 19 keywords; `ArchMatchHead` aux (match_w=0.5 arch, 0.3 weights); 1200/2000 train steps; 25 eval tasks |
| **Key metrics** | match=**0.92**, zero=1.76, Δ=+0.32, FT100=1.11 |
| **Outcome** | **success** — steering quality up; minor zero-shot regression vs v1 |
| **Artifacts** | `checkpoints/arch_gen/task_cond_phase4_v2.pt`, `checkpoints/plots/phase4_refine/` |

#### v2 vs v1 comparison

| Metric | v2 refine | v1 | Δ |
|--------|-----------|-----|---|
| Eval tasks | 25 | 10 | — |
| Arch-task match | **0.92** | 0.73 | **+0.19** |
| Zero-shot loss | 1.76 | 1.69 | +0.07 (worse) |
| Zero Δ vs random | +0.32 | +0.39 | −0.07 |
| FT100 loss | 1.11 | 1.08 | +0.02 |
| Uncond zero (same run) | 1.67 | — | task still trails uncond on loss |

#### Steering examples (v2)

| Task | Generated (h, d, skip) | Match |
|------|------------------------|-------|
| large deep skip connections char predictor | (160, 3, true) | 1.0 |
| efficient shallow char predictor | (48, 1, false) | 1.0 |
| compact shallow simple char predictor | (32, 1, false) | 1.0 |
| deep reasoning capacity char predictor | (160, 3, false) | 1.0 |
| lightweight fast char predictor | (64, 2, false) | 0.0 ← conflicting keywords |

#### Analysis

**Match-aware aux loss works** — `ArchMatchHead` predicts h/depth/skip from task embedding during arch + weight training. Skip connections now hit reliably (v1 often missed skip).

**Expanded vocab closes v1 gaps** — "efficient", "lightweight", "reasoning", "capacity", "skip connections" all map via `describe_spec()` + `spec_matches_task()`.

**Tradeoff** — stronger steering slightly hurts zero-shot (−0.07 vs v1). Uncond baseline still ~0.09 lower loss. Acceptable for controllable generation use case.

**Remaining failures** — contradictory phrases ("lightweight" + "fast" with medium h) and compound "medium wide balanced" (0.5). Need synonym disambiguation or sentence embedder.

---

### 2026-06-19 — [Phase 4] Task conditioning (text → arch + weights)

| Field | Value |
|-------|-------|
| **Phase** | 4 |
| **Script** | `experiments/phase4_task_cond.py` |
| **Seed** | 42 |
| **Hyperparameters** | Keyword TaskEmbedder; TaskArchGenerator 800 steps; TaskCondWeightGenerator 2000 steps on 140 multi-arch pairs; 10 eval task strings |
| **Key metrics** | See table below |
| **Outcome** | **partial success** — strong arch steering; loss parity with Phase 3 |
| **Artifacts** | `checkpoints/arch_gen/task_cond_phase4_v1.pt`, `checkpoints/plots/phase4_task/` |

#### Comparison table

| Metric | Task-conditioned | Phase 3 uncond | Phase 3 ref (50-sample) | Random |
|--------|------------------|--------------|---------------------------|--------|
| Zero-shot loss | 1.69 | 1.66 | 1.71 | 2.08 |
| Zero Δ vs random | +0.39 | +0.42 | +0.37 | — |
| FT100 loss | 1.08 | 1.08 | — | — |
| Arch-task match | **0.73** | — | — | — |

#### Example task → architecture mappings

| Task | Generated (h, d, skip) | Match |
|------|------------------------|-------|
| small narrow char predictor | (24, 3, false) | 1.0 |
| shallow fast char predictor | (48, 1, false) | 1.0 |
| large deep char predictor | (128, 3, false) | 1.0 |
| large wide char predictor | (128, 2, true) | 1.0 |
| large deep skip char predictor | (96, 3, false) | 0.67 (skip missed) |

#### Analysis

**Steering works** — avg match 0.73 vs heuristic keyword alignment. Clear cases: "small narrow" → h=24, "shallow fast" → depth=1, "large deep" → h=128 d=3.

**Loss neutral vs uncond** — task pipeline matches Phase 3 quality (Δ +0.39 vs random) without degrading zero-shot. Uncond slightly lower loss this run (1.66 vs 1.69) but without task alignment.

**Limitations** — keyword bag is brittle ("efficient" doesn't map); skip keyword hit rate imperfect. Next: richer vocab + more training pairs.

---

### 2026-06-19 — [Milestone] Minimal viable proof-of-concept declared

| Field | Value |
|-------|-------|
| **Phase** | 3 complete → post-MVP |
| **Evidence** | Phase 3 cond v1 + refine runs; `demo_generate_model.py` |
| **Headline** | Variable arch + conditioned weights; zero Δ **+0.37**; FT100 **1.07** |
| **Docs** | `README.md` MVP section, `MVP_LEARNINGS.md`, updated `VISION.md` |
| **Outcome** | **MVP success** at tiny scale |
| **Limitations** | Synthetic task, hypernet weights, no task conditioning |

See Phase 3 refine entry below for full metrics. Next recommended work: Phase 4 task conditioning or tiny transformer scale-up.

---

### 2026-06-19 — [Phase 3] Refine — expanded diversity + conditioning stress test

| Field | Value |
|-------|-------|
| **Phase** | 3 (MVP) |
| **Script** | `experiments/phase3_conditioned_refine.py` |
| **Seed** | 42 |
| **Hyperparameters** | 35-spec grid (h∈24–160, depth 1–3, skip); 140 training weights; CondWeightGenerator 2000 steps; 80 gen / 50 eval; FT100; noise-scale sweep |
| **Key metrics** | See table below |
| **Outcome** | **success** — conditioning holds with more diversity |
| **Artifacts** | `checkpoints/weights/multi_arch_v2/`, `checkpoints/plots/phase3_refine/` |

#### Comparison table

| Metric | **Cond v2** | Cond v1 | Ref hybrid | Random |
|--------|-------------|---------|------------|--------|
| Zero-shot loss | **1.71** | 1.73 | 1.96 | 2.08 |
| Zero Δ vs random | **+0.37** | +0.35 | +0.12 | — |
| FT100 loss | 1.07 | 1.19 | **1.05** | — |

#### Diversity stats (50 eval samples)

| Stat | Value |
|------|-------|
| Unique specs | 26 / 35 grid |
| Depth distribution | d=3: 32, d=2: 13, d=1: 5 |
| Skip enabled | 13 / 50 |
| Grid size | 35 valid specs |

#### Noise-scale sweep (10 held-out grid specs, zero-shot avg loss)

| Scale | 0.25 | 0.5 | 1.0 | 2.0 |
|-------|------|-----|-----|-----|
| Avg loss | **1.64** | 1.64 | 1.65 | 1.69 |

#### Analysis

**Conditioning generalizes** — expanded arch space (deeper, wider, skip connections) does not break the pipeline. Zero-shot **improves slightly** vs v1 (1.71 vs 1.73) despite harder diversity; FT100 closes gap to ref hybrid (1.07 vs 1.05).

**Depth bias** — generator favors depth=3 (32/50); conditioning still works, suggesting hypernet learns cross-depth mapping from 140 training checkpoints.

**Noise sensitivity** — lower latent noise (0.25–0.5) marginally best; default 1.0 is fine.

**MVP declaration:** End-to-end generative tiny-model pipeline is proven at minimal scale. Ready for Phase 4 (joint generation, scaling, or real tasks).

---

### 2026-06-19 — [Phase 3] Architecture-conditioned weight generation

| Field | Value |
|-------|-------|
| **Phase** | 3 |
| **Script** | `experiments/phase3_conditioned.py` |
| **Seed** | 42 |
| **Hyperparameters** | 100 multi-arch weights (10×10 grid); `CondWeightGenerator` 2000 steps; arch embed + latent hypernet; 30 generated archs; FT100 |
| **Key metrics** | See table below |
| **Outcome** | **success** — conditioning fixes variable-arch weight init |
| **Artifacts** | `checkpoints/weights/multi_arch/`, `checkpoints/plots/phase3_conditioned/` |

#### Comparison table

| Metric | **Conditioned** | Uncond (v1) | Ref hybrid | Random |
|--------|-----------------|-------------|------------|--------|
| Zero-shot loss | **1.73** | 2.07 | 1.90 | 2.08 |
| Zero Δ vs random | **+0.35** | +0.01 | +0.18 | — |
| FT100 loss | **1.19** | 1.31 | **0.99** | — |
| FT100 Δ vs random | +0.02 | -0.09 | +0.22 | — |
| Conditioned inits | **30/30** | 1/30 hybrid | — | — |

#### Analysis

**Conditioning works** — `ArchEmbedder` + hypernet weight head trained on multi-arch checkpoints gives every generated topology a tailored init. Zero-shot improves **+0.34 vs unconditioned v1** and **+0.35 vs random** (vs +0.01 before). This confirms the Phase 3 v1 bottleneck was missing arch-aware weights, not bad architectures.

**Ref hybrid** still wins FT100 on fixed h=64,d=1 (0.99 vs 1.19). Conditioned init is best aggregate zero-shot across diverse archs; hybrid remains FT-first for reference topology.

**Similarity plot** — `phase3_cond_v1_sim_vs_zero.png` shows zero-loss vs ref similarity; conditioned weights help across the grid, not only near h=64.

**Next:** Blend cond + hybrid on compatible archs; optional JEPA-latent conditioning instead of pure hypernet MSE.

---

### 2026-06-19 — [Phase 3] Simple architecture generation + weight init (v1)

| Field | Value |
|-------|-------|
| **Phase** | 3 |
| **Script** | `experiments/phase3_arch_gen.py` |
| **Seed** | 42 |
| **Hyperparameters** | GRU arch generator, 500 train steps; 30 sampled archs; hybrid weight init when h=64,d=1; FT100; compare vs fixed ref hybrid/random |
| **Key metrics** | See table below |
| **Outcome** | **partial success** — pipeline works; weak performance signal |
| **Artifacts** | `checkpoints/arch_gen/`, `checkpoints/plots/phase3_arch/` |

#### Comparison table

| Metric | Gen arch+weights | Ref hybrid (fixed) | Ref random | Random baseline |
|--------|------------------|--------------------|------------|-----------------|
| Zero-shot loss | 2.07 | **1.74** | 2.08 | 2.08 |
| Zero Δ vs random | +0.01 | **+0.34** | ~0 | — |
| FT100 loss | 1.37 | **0.94** | 1.24 | — |
| FT100 Δ vs random | -0.15 | **+0.27** | -0.03 | — |

#### Architecture diversity

| Stat | Value |
|------|-------|
| Unique specs / 30 samples | 9 / 10 valid grid |
| JEPA/hybrid-compatible (h=64, d=1) | 1 / 30 |
| Depth=2 fraction | 23 / 30 |
| Arch generator final loss | 0.77 |

#### Analysis

**Pipeline works end-to-end** — autoregressive generator produces valid `ArchSpec` token sequences; `VariableTinyMLP` instantiates without shape errors; eval + FT run on diverse architectures. Sample diagrams saved to `phase3_v1_sample_archs.txt`.

**Performance gap explained** — hybrid/JEPA weights only load when architecture matches Phase 0 reference (9288 params). 29/30 generated archs got random weight init, so aggregate metrics ≈ random (Δ +0.01 zero-shot). The one compatible sample could use hybrid but is drowned out in the mean.

**Arch generator** — learns token grammar (100% valid after fallback); prefers depth=2 and varied hidden sizes — good diversity, bad for current weight pipeline.

**Conclusion:** Phase 3 proves the *shape* of the full vision (gen arch → gen/load weights → eval). Next increment must make weights **architecture-aware** or expand weight training across arch families.

---

### 2026-06-19 — [Phase 2] Hierarchical JEPA (H-JEPA) vs flat JEPA

| Field | Value |
|-------|-------|
| **Phase** | 2 (bridge to 3) |
| **Script** | `experiments/phase2_hierarchical_jepa.py` |
| **Seed** | 42 |
| **Hyperparameters** | 2-level H-JEPA, 2000 train steps, latent_dim=64; compare vs frozen flat JEPA + diffusion; H-JEPA + flat hybrids at refine_t=50; 30 samples; FT 100/200 |
| **Key metrics** | See table below |
| **Outcome** | **inconclusive / negative** — H-JEPA does not beat flat JEPA |
| **Artifacts** | `checkpoints/jepa/weight_hjepa_hjepa_global.pt`, `metrics_hjepa_global.json`, `checkpoints/plots/phase2_hjepa/` |

#### Comparison table

| Metric | Flat JEPA | H-JEPA | Flat hybrid | H hybrid | Diffusion | Random |
|--------|-----------|--------|-------------|----------|-----------|--------|
| Zero-shot loss | **1.84** | 1.89 | 1.84 | 1.89 | 7.37 | 2.08 |
| Zero Δ vs random | **+0.24** | +0.19 | +0.24 | +0.18 | -5.29 | — |
| FT100 loss | 1.03 | 1.12 | **0.99** | 1.06 | **0.87** | 1.22 |
| FT100 Δ vs random | +0.19 | +0.11 | +0.24 | +0.17 | **+0.36** | — |
| FT200 loss | **0.88** | 0.95 | **0.86** | 0.89 | **0.84** | 0.97 |

#### Analysis

**Hierarchy did not help weight generation** — H-JEPA is slightly worse than flat JEPA on zero-shot (-0.06 loss) and FT100/200 (-0.09 / -0.07 vs flat). The global embedding `h` and top-down conditioning add parameters and training objectives without improving sampled weights on this tiny MLP task.

**Hybrids follow the same pattern** — flat hybrid still beats H-hybrid on FT; both preserve near-JEPA zero-shot.

**Structural value for Phase 3** — H-JEPA's global `h` may still be useful as an architecture-level latent even if it doesn't improve flat weight sampling. Proceed to Phase 3 using flat JEPA for weights and H-JEPA concepts for arch tokens.

**Note:** Post-training eval runs ~5 min on CPU with no output unless `--eval_only` (now has tqdm progress bars). Training completing 100% then appearing hung was the silent sampling+eval phase.

---

### 2026-06-19 — [Phase 2] JEPA + diffusion hybrid (refine_t=50)

| Field | Value |
|-------|-------|
| **Phase** | 2 (bridge to 3) |
| **Script** | `experiments/phase2_hybrid.py` |
| **Seed** | 42 |
| **Hyperparameters** | Frozen `weight_jepa_phase2_global.pt` + `weight_denoiser_norm_global.pt`; hybrid = JEPA sample → noise to t=50 → reverse diffuse; 30 samples; FT 100/200 |
| **Key metrics** | See table below |
| **Outcome** | **partial success** — hybrid middle ground; no pareto win |
| **Artifacts** | `checkpoints/hybrid/metrics_hybrid_t50.json`, `checkpoints/plots/phase2_hybrid/` |

#### Comparison table (extends Phase 2 JEPA vs diffusion)

| Metric | JEPA | Diffusion | **Hybrid** | Random |
|--------|------|-----------|------------|--------|
| Zero-shot loss | **1.82** | 7.47 | 1.83 | 2.08 |
| Zero Δ vs random | **+0.26** | -5.39 | **+0.25** | — |
| FT100 loss | 1.02 | **0.88** | 0.99 | 1.22 |
| FT100 Δ vs random | +0.21 | **+0.35** | +0.24 | — |
| FT200 loss | 0.87 | **0.84** | 0.86 | 0.97 |
| FT200 Δ vs random | +0.10 | **+0.13** | +0.11 | — |

#### Analysis

**Zero-shot:** Hybrid preserves JEPA's advantage — loss 1.83 vs JEPA 1.82 (Δ +0.25 vs random, vs JEPA +0.26). Light diffusion refine at t=50 barely perturbs the good JEPA init.

**Fine-tuning:** Hybrid improves over JEPA (FT100: 0.99 vs 1.02, FT200: 0.86 vs 0.87) but still trails diffusion (FT100: 0.88, FT200: 0.84). Diffusion manifold pull helps FT without destroying zero-shot.

**Trade-off:** Hybrid does **not** beat diffusion on FT or JEPA on zero-shot. It is a sensible compromise when you want positive zero-shot *and* better FT than raw JEPA — useful as a default weight pipeline before Phase 3 architecture search.

**Next:** Proceed to Phase 3; carry JEPA chunk structure + optional hybrid refine for generated architectures.

---

### 2026-06-19 — [Phase 2] JEPA vs diffusion (global norm)

| Field | Value |
|-------|-------|
| **Phase** | 2 |
| **Script** | `experiments/phase2_jepa_vs_diffusion.py` |
| **Seed** | 42 |
| **Hyperparameters** | 100 global-norm weights; JEPA 2000 steps, latent_dim=64, 5 layer chunks; diffusion baseline `norm_global` checkpoint (frozen, resampled) |
| **Key metrics** | See table below |
| **Outcome** | **mixed success** — JEPA wins zero-shot; diffusion wins FT |
| **Artifacts** | `checkpoints/jepa/`, `checkpoints/plots/phase2_jepa/` |

#### Comparison table

| Metric | JEPA | Diffusion | Random |
|--------|------|-----------|--------|
| Zero-shot loss | **1.84** | 7.01 | 2.08 |
| Zero Δ vs random | **+0.24** | -4.94 | — |
| FT100 loss | 1.01 | **0.87** | 1.22 |
| FT100 Δ vs random | +0.21 | **+0.35** | — |
| FT200 loss | 0.87 | **0.84** | 0.97 |
| FT200 Δ vs random | +0.09 | **+0.13** | — |

#### Analysis

**JEPA wins zero-shot** — First method to beat random without fine-tuning (Δ=+0.24). Chunk-wise latents + Gaussian sampling produce usable weight vectors directly. Zero-shot loss 1.84 is far closer to collected real weights (~0.83) than diffusion (7.01).

**Diffusion wins fine-tuning** — After 100/200 FT steps, diffusion samples remain better (FT100: 0.87 vs 1.01). Diffusion still best as a learned initializer when brief training is allowed.

**Surprises:** JEPA's simple latent Gaussian sampler dramatically fixes the zero-shot problem that plagued diffusion. The FT gap suggests JEPA decoders may be slightly underfit or latent sampling too conservative.

**Conclusion:** Use JEPA for zero-shot, diffusion for FT-first workflows. Phase 3 (architecture generation) or hybrid JEPA→diffusion are logical next steps.

---

### 2026-06-19 — [Phase 1] Normalization ablation

| Field | Value |
|-------|-------|
| **Phase** | 1 |
| **Script** | `experiments/phase1_norm_ablation.py` |
| **Seed** | 42 |
| **Hyperparameters** | Same 100 raw weights; 3 norm modes; 2000 diffusion steps each; 30 samples; FT 100/200 |
| **Key metrics** | See table below |
| **Outcome** | **success** — global norm wins; FT signal preserved/improved |
| **Artifacts** | `checkpoints/plots/phase1_norm_ablation/` |

#### Comparison table

| Norm | Gen zero loss | Zero Δ | FT100 Δ | FT200 Δ | Gen FT100 loss |
|------|---------------|--------|---------|---------|----------------|
| **global** | 7.97 | -5.89 | **+0.36** | +0.14 | **0.867** |
| layer | 14.21 | -12.13 | +0.33 | +0.14 | 0.897 |
| perdim | 13.24 | -11.16 | +0.26 | +0.14 | 0.963 |
| random (ref) | 2.08 | — | — | — | 1.224 |
| collected (ref) | 0.83 | — | — | — | 0.827 |

*(Positive Δ = generated better than random)*

#### Analysis

**Best for zero-shot:** `global` — loss 7.97 vs 13.24 (perdim) and 14.21 (layer). Still worse than random (2.08), but per-dimension normalization was the main culprit for extreme denormalized values.

**Best for fine-tuning:** `global` — FT100 Δ +0.36 (vs +0.26 perdim), generated FT100 loss 0.867 approaching collected real weights (0.827).

**Layer vs perdim:** Layer-wise is slightly worse than perdim on zero-shot; both share the high-dimensional scaling problem. Layer groups help structure but don't fix low-N noise.

**Did any break FT advantage?** No — all three modes beat random at FT100 and FT200.

**Recommended default:** `global` for all future Phase 1+ work.

**Conclusion:** Phase 1 core proof **complete**. Proceed to Phase 2 (JEPA) with global normalization.

---

### 2026-06-19 — [Phase 1] Full weight diffusion experiment

| Field | Value |
|-------|-------|
| **Phase** | 1 |
| **Script** | `experiments/phase0_train_baselines.py`, `scripts/collect_weights.py`, `experiments/phase1_weight_diffusion.py` |
| **Seed** | 42 (phase1); seeds 0–49 baselines; 0–99 collection |
| **Hyperparameters** | 50 baseline seeds; 100 collected models @ 500 train steps; diffusion 2000 steps, T=200, batch=32, lr=1e-4; 30 eval samples; FT 100 & 200 steps |
| **Key metrics** | gen_zero_loss=13.24, rnd_zero=2.08; **ft100 delta=+0.26**; **ft200 delta=+0.14**; collected_zero=0.83; diffusion_final_loss=0.99 |
| **Outcome** | **partial success** — FT signal yes, zero-shot no |
| **Notes** | See analysis below |
| **Artifacts** | `checkpoints/weights/phase0/`, `checkpoints/diffusion/metrics_phase1_full.json`, `checkpoints/plots/phase1_full_*.png` |

#### Analysis

**What worked**
- Diffusion training converged (final loss ~0.99).
- After **100-step fine-tune**, generated inits beat random by **0.26 eval loss** (~21% relative) and **0.047 accuracy**.
- After **200-step FT**, generated still beats random (+0.14 loss) but margin shrinks — random catches up with more training.
- Generated weights land in a plausible distribution (see `phase1_full_weight_hist.png`) but are not plug-and-play.

**What failed**
- **Zero-shot** generated weights are unusable (loss 13.2 vs random 2.1). Samples likely have extreme values after per-dimension denormalization when N=100 and D=9288 (many dims have tiny estimated std).

**Minimal fixes to try next**
1. Add **global scalar normalization** option (single mean/std over all weights) — likely fixes zero-shot.
2. Try **layer-wise** normalization (group params by tensor).
3. Increase diffusion steps to 5000 or denoiser hidden dim if normalization ablation helps FT but not zero-shot.

**Conclusion:** Basic weight diffusion **works as a learned initializer** for this tiny setup after brief fine-tuning. Phase 1 core proof is **partially validated**. Do not proceed to JEPA until normalization ablation is run.

---

### 2026-06-19 — [Phase 0] Smoke + bootstrap

| Field | Value |
|-------|-------|
| **Phase** | 0 |
| **Script** | N/A (setup) |
| **Seed** | — |
| **Key metrics** | — |
| **Outcome** | smoke: pipeline OK; full run superseded by Phase 1 entry above |
| **Notes** | Initial smoke (10 models, 500 diffusion steps) showed weak FT signal only. |
| **Artifacts** | — |
