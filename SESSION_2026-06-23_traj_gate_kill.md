# Session Notes — 2026-06-23 — GATE-TRAJ Kill Verdict
*ForgeCritic (z-ai/glm-5.2) + Oz infra loop*

---

## TL;DR

**Trajectory prediction is dead.** Three predictor generations (v1 naive, v2 MLP+ceiling, v3 window-conditioned) all failed to beat identity. ForgeCritic's decisive diagnostic — damped per-trajectory linear extrapolation — confirmed the kill: the trajectory is curved and decelerating, direction is right (cosine 0.95) but magnitude overshoots by 53%, and the curvature varies per-model in a way no predictor from trajectory data can capture. The headroom is real but not learnable from trajectory data.

**Paper scope narrows to GATE-GROW** (growth + low-rank correction in aligned space) as primary contribution, with alignment-helps and the negative trajectory result as supporting sections. Everything else (Phases 1–5c, diffusion, JEPA latent, hierarchical progressive, scale-consistency) is cut from the final paper.

---

## GATE-TRAJ v1 (naive predictors) — FAILED

100 trajectories × 11 checkpoints, depth=3 H=64 D=17608, 80 train / 20 test.

**h=200:**
| predictor | CE | delta vs identity |
|---|---|---|
| identity | 1.185 ± 0.363 | — |
| avg_vel_aln | 1.233 ± 0.337 | −0.048 |
| avg_vel_raw | 1.248 ± 0.337 | −0.063 |
| pca_ridge_aln | 1.614 ± 0.339 | −0.429 |
| pca_ridge_raw | 2.498 ± 4.169 | −1.314 |

GATE PASSES: False. Alignment helps: True (both arms).

## GATE-TRAJ v2 (MLP + actual_future ceiling) — FAILED

Added actual_future (evaluate real w_{t+n} = ceiling) and 3-layer MLP in aligned PCA space.

**h=200:**
| predictor | CE | delta vs identity |
|---|---|---|
| identity | 1.185 ± 0.363 | — |
| actual_future | 0.983 ± 0.151 | +0.202 (HEADROOM) |
| avg_vel_aln | 1.233 | −0.048 |
| pca_ridge_aln | 1.614 | −0.429 |
| mlp_aln | 7.812 ± 18.293 | −6.627 (CATASTROPHIC DIVERGENCE) |

**Key signal:** actual_future proves 0.202 CE headroom exists. No predictor captures it. MLP diverges catastrophically (std 18.29). Predictor captures −23.8% of headroom (negative = worse than identity).

**Per-t breakdown (h=100) — the diagnostic that inverted the framing:**
| t | step | identity | actual_f | mlp_aln | headroom | cap% |
|---|---|---|---|---|---|---|
| 0 | 0 | 2.079 | 1.413 | 1.950 | +0.667 | 0% |
| 4 | 400 | 1.083 | 0.999 | 2.544 | +0.083 | −10% |
| 8 | 800 | 0.845 | 0.833 | 4.388 | +0.012 | −232% |

**Headroom shrinks from 0.667 (t=0) to 0.012 (t=8).** Application window is EARLY training, not late. Inverts the earlier "prediction works in late-training" claim.

## GATE-TRAJ v3 (window-conditioned, K=3) — FAILED (decisive)

Window of last 3 checkpoints gives velocity + acceleration. Smaller MLP (128 hidden, 2 layers), weight_decay 1e-2, early stopping, train-only PCA (no leakage).

**h=200:**
| predictor | CE | delta vs identity |
|---|---|---|
| identity | 1.025 ± 0.151 | — |
| actual_future | 0.917 | +0.108 (headroom) |
| window_mlp_aln | 1.752 ± 0.030 | −0.756 |
| window_mlp_raw | 1.873 | −0.873 |

**h=100:**
| predictor | CE | delta vs identity |
|---|---|---|
| identity | 1.002 ± 0.151 | — |
| actual_future | 0.947 | +0.054 (headroom) |
| window_mlp_aln | 1.758 ± 0.032 | −0.756 |
| window_mlp_raw | 1.876 | −0.875 |

GATE PASSES: False. Alignment helps: True. Capture: −1390% of headroom.

**Why it's the kill:** win_aln std is 0.03 — the model collapsed to a degenerate near-constant prediction (output the dataset mean regardless of input). CE is flat ~1.75 across all t. The window did not fix it.

## ForgeCritic's Decisive Diagnostic (the actual kill)

Damped per-trajectory linear extrapolation — simplest predictor using each model's own velocity with damping factor alpha:

- alpha=0.05: CE=1.489 vs identity=1.485 → −1.8% of headroom (worse than doing nothing)
- Every alpha > 0: worse than identity

**Mechanism:** cosine similarity between per-step delta and actual 2-step displacement is 0.95 (direction right) BUT magnitude ratio is 1.53 (overshoots by 53%). Trajectory is curved and decelerating. Curvature varies per-model in a way position/velocity history cannot capture.

**Exhausted predictor space:**
- Cross-trajectory avg velocity → fails (orthogonal divergence, cosine 0.02–0.12)
- PCA + ridge → fails (variance subspace ≠ predictive subspace)
- Snapshot MLP → fails (can't see velocity)
- Window MLP K=3 → fails (collapses to mean, std 0.03)
- Per-trajectory linear extrapolation → fails (curved, overshoots 53%)
- Damped extrapolation → fails (any step wrong-direction hurts)

**The information needed to predict the future is not in the trajectory data. It's in the loss landscape, which the predictors never see.** A JEPA encoder/decoder is the same family as the MLP — it would also collapse to the mean.

---

## Paper Scope (ForgeCritic's verdict)

**Primary contribution: GATE-GROW** (growth + low-rank correction in aligned space, amortizing pretraining compute). Not yet tested — this is the experiment that makes or breaks the thesis.

**Two supporting sections:**
1. **Alignment helps** (confirmed 3×) — variance collapse larger, cross-model prediction less catastrophic, directionally consistent. Enabling technique, not headline.
2. **Negative trajectory result** — headroom exists (0.67 CE at init) but not learnable from trajectory data; headroom shrinks with training. One paragraph + one table: "We investigated trajectory prediction as an alternative to growth-based amortization. Despite measurable headroom, no predictor — linear, learned, or window-conditioned — could capture it. The headroom is real but not learnable from trajectory data alone, motivating our growth-based approach."

**Cut from final paper:** Phases 1–5c, diffusion, JEPA latent model, hierarchical progressive, scale-consistency. Prior work that informed direction but doesn't belong in the paper.

**Three sections: alignment, negative trajectory result, positive growth result.**

---

## Artifacts

- `checkpoints/align/traj_gate/traj_gate_summary.json` — v1 h=200
- `checkpoints/align/traj_gate_h100/traj_gate_summary.json` — v1 h=100
- `checkpoints/align/traj_gate_v2/traj_gate_summary.json` — v2 h=200
- `checkpoints/align/traj_gate_v2_h100/traj_gate_summary.json` — v2 h=100
- `checkpoints/align/traj_gate_v3_h200/traj_gate_v3_summary.json` — v3 h=200
- `checkpoints/align/traj_gate_v3_h100/traj_gate_v3_summary.json` — v3 h=100
- `checkpoints/trajectories/trajectories.pt` — 100 trajectories × 11 checkpoints (pulled local, no re-collection needed)
- `checkpoints/trajectories/meta.json`

## Code (branch align-ablation)

- `3882eb6` — GATE-TRAJ collection + experiment
- `61080f9` — actual_future ceiling + per-checkpoint breakdown
- `5a30aeb` — MLP predictor arm
- `4ec6efe` — v3 window-conditioned predictor

## Next

GATE-GROW — ForgeCritic to spec. The decisive positive experiment.
