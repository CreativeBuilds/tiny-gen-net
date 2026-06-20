---
name: continue
description: >-
  Advance tiny-gen-net one meaningful research increment. Read STATE.md and
  EXPERIMENTS.md, review results, implement the next phase step, update docs.
  Use when the user says /continue or wants to resume the research codebase.
---

# Continue — tiny-gen-net

Advance the research codebase **one meaningful, verifiable increment**. This is the primary workflow for resuming work.

## Step 1 — Read state (mandatory)

Read in order:

1. `STATE.md` — current phase, what exists, next steps
2. `EXPERIMENTS.md` — last runs, metrics, outcomes
3. `README.md` — quick context if needed
4. Relevant code for current phase:
   - Phase 0: `src/models/tiny_mlp.py`, `src/utils/train_loop.py`, `experiments/phase0_train_baselines.py`, `scripts/collect_weights.py`
   - Phase 1: `src/diffusion/simple_diffusion.py`, `experiments/phase1_weight_diffusion.py`, `src/utils/weights.py`

## Step 2 — Assess

Answer internally:

- What phase are we in? What is the **success criterion**?
- Did the last experiment succeed, fail, or was it only a smoke test?
- What is the **single best next step** (not three steps at once)?

## Step 3 — Propose briefly

Tell the user:

- Current phase + status (1–2 sentences)
- Proposed increment (1 concrete action)
- Expected success signal

If blocked (missing deps, no weights collected), fix that first.

## Step 4 — Implement

Rules:

- **Minimal diff** — Karpathy style; don't refactor unrelated code
- **Run the experiment** — don't just write code; execute and capture metrics
- **Plots** — ensure outputs land in `checkpoints/plots/`
- **Phase gates** — don't start JEPA or arch_gen until Phase 1 shows signal

### Phase-specific next steps

| Phase | If not done | If done / partial |
|-------|-------------|-------------------|
| 0 | Run `phase0_train_baselines.py`, then `collect_weights.py --num_models 50` | Proceed to Phase 1 |
| 1 | Run full `phase1_weight_diffusion.py` (50+ models, 2000 steps) | Tune normalization, denoiser, or dataset size; log ablations |
| 2 | Only after Phase 1 success documented | Implement minimal JEPA in `src/jepa/` |

## Step 5 — Update docs (mandatory)

1. Append entry to `EXPERIMENTS.md` with metrics and outcome
2. Update `STATE.md`: phase, key results, open questions, next steps

## Step 6 — Report

Summarize for user:

- What was done
- Key metrics (with comparison to random baseline if Phase 1)
- Plot/artifact paths
- Recommended next `/continue` action

## Default full pipeline (when starting fresh)

```bash
source .venv/bin/activate
pip install -r requirements.txt
python experiments/phase0_train_baselines.py --num_seeds 10
python scripts/collect_weights.py --num_models 50
python experiments/phase1_weight_diffusion.py --weights checkpoints/weights/phase0/weights.pt
```

## Anti-patterns

- Skipping `EXPERIMENTS.md` / `STATE.md` updates
- Jumping to Phase 2/3 without Phase 1 metrics
- Adding heavy dependencies or large refactors in one session
- Claiming success on smoke-test configs (10 models, 500 diffusion steps)
