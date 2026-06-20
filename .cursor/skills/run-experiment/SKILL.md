---
name: run-experiment
description: >-
  Run a tiny-gen-net experiment (Phase 0 or 1), capture metrics and plots,
  log results to EXPERIMENTS.md and update STATE.md. Use for /run-phase0,
  /run-phase1, or explicit experiment requests.
---

# Run Experiment — tiny-gen-net

Execute a standard experiment, capture outputs, update living docs.

## Parse input

| User says | Action |
|-----------|--------|
| phase 0, baselines | Phase 0 workflow |
| phase 1, diffusion | Phase 1 workflow |
| smoke | Reduced config (see below) |
| full | Production config for phase success criteria |

Optional flags from user: `--num_models`, `--diffusion_steps`, `--seed`, `--steps`

## Environment

```bash
cd <repo-root>
source .venv/bin/activate 2>/dev/null || python -m venv .venv && source .venv/bin/activate
pip install -q -r requirements.txt
```

## Phase 0 workflow

```bash
python experiments/phase0_train_baselines.py --num_seeds 10 --steps 500
python scripts/collect_weights.py --num_models 50 --steps 500 --out checkpoints/weights/phase0
```

**Smoke:** `--num_seeds 3 --steps 200`, collect `--num_models 10`

**Success signal:** Loss decreases; avg eval_acc reported; `checkpoints/weights/phase0/weights.pt` exists

**Artifacts:**
- `checkpoints/plots/phase0_baselines.png`
- `checkpoints/weights/phase0/weights.pt` + `meta.json`

## Phase 1 workflow

Requires Phase 0 weights at `checkpoints/weights/phase0/weights.pt`.

```bash
python experiments/phase1_weight_diffusion.py \
  --weights checkpoints/weights/phase0/weights.pt \
  --diffusion_steps 2000 \
  --num_samples 30 \
  --ft_steps 100,200 \
  --tag phase1_full
```

**Smoke:** `--diffusion_steps 500 --num_samples 5 --ft_steps 50`

**Success signal:** `ft100_delta_vs_random` or `ft200_delta_vs_random` positive in metrics JSON; zero-shot optional

**Artifacts:**
- `checkpoints/diffusion/weight_denoiser_{tag}.pt`
- `checkpoints/diffusion/metrics_{tag}.json`
- `checkpoints/plots/{tag}_*.png`

## Log results

Append to `EXPERIMENTS.md`:

```markdown
### YYYY-MM-DD — [Phase N] <title>
| Field | Value |
| **Phase** | N |
| **Script** | `experiments/...` |
| **Seed** | ... |
| **Key metrics** | ... |
| **Outcome** | success / inconclusive / failed |
| **Notes** | smoke vs full, observations |
| **Artifacts** | paths |
```

Update `STATE.md` key results and next steps.

## Report to user

Include:
- Command(s) run
- Summary metrics table
- Whether smoke or full
- Phase 1: zero-shot and FT deltas vs random
- Paths to plots
